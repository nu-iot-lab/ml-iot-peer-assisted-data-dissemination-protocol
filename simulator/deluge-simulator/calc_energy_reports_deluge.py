#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Energy report generator for the Deluge simulator.

Expected per-run files: metadata.csv and node_summary.csv. Deluge stages are
ADV, REQ and DATA. RX split columns are rx_on_adv_ms, rx_on_req_ms,
rx_on_data_ms plus csma_cca_ms and cad_scan_ms.
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
from typing import Dict, Iterable, List
from collections import defaultdict

STAGES=["adv","req","data"]
TX_USED_COLS={"adv":"adv_used_s","req":"req_used_s","data":"data_used_s"}


def read_csv_rows(path: Path) -> List[Dict[str,str]]:
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv_rows(path: Path, rows: Iterable[Dict[str,object]], fieldnames=None):
    rows=list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames=[]
        for r in rows:
            for k in r:
                if k not in fieldnames: fieldnames.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def fnum(v, default=0.0):
    try:
        s=str(v).strip()
        if not s or s.lower() in {'nan','none','null'}: return default
        return float(s)
    except Exception:
        return default

def inum(v, default=0): return int(round(fnum(v, default)))

def load_key_value_csv(path: Path) -> Dict[str,str]:
    if not path.is_file(): return {}
    rows=read_csv_rows(path)
    if not rows: return {}
    if {'key','value'} <= set(rows[0].keys()):
        return {r['key']:r.get('value','') for r in rows if r.get('key')}
    return dict(rows[0])

def discover_run_dirs(root: Path, recursive: bool) -> List[Path]:
    if (root/'metadata.csv').is_file() and (root/'node_summary.csv').is_file(): return [root]
    if not recursive: raise SystemExit(f'No metadata.csv + node_summary.csv found in {root}. Use --recursive.')
    return sorted({p.parent for p in root.rglob('metadata.csv') if (p.parent/'node_summary.csv').is_file()})

def extract_node_count(name: str) -> int:
    nums=[int(x) for x in re.findall(r'(?:^|_)(\d+)(?=_|$)', Path(str(name)).stem)]
    if len(nums)>=2: return nums[-2]
    return nums[0] if nums else 0

def load_energy_profiles(path: Path) -> Dict[str,Dict[str,float]]:
    rows=read_csv_rows(path)
    out={}
    for r in rows:
        key=str(r.get('radio_profile') or r.get('technology') or r.get('profile') or '').strip()
        if not key: continue
        rx=fnum(r.get('rx_current_ma'))
        out[key]={
            'tx_current_ma': fnum(r.get('tx_current_ma')),
            'rx_current_ma': rx,
            'cca_current_ma': fnum(r.get('cca_current_ma'), rx),
            'cad_current_ma': fnum(r.get('cad_current_ma'), rx),
            'voltage': fnum(r.get('voltage', r.get('voltage_v', 3.7)), 3.7),
        }
    return out

def profile_for(metadata, profiles):
    for k in [metadata.get('radio_profile',''), metadata.get('technology','')]:
        if k in profiles: return k, profiles[k]
    if profiles:
        # fall back by canonical prefix for configs named deluge_...
        rp=metadata.get('radio_profile','')
        for k,v in profiles.items():
            if k in rp or rp in k: return k,v
    raise SystemExit(f"No matching energy profile for {metadata.get('radio_profile')!r}")

def run_context(run_dir: Path, metadata: Dict[str,str], node_count: int) -> Dict[str,object]:
    manifest=load_key_value_csv(run_dir/'batch_manifest.csv')
    testbed=Path(manifest.get('testbed_path', metadata.get('testbed',''))).name
    return {
        'run_dir': str(run_dir), 'rep': manifest.get('rep',''),
        'data_size_bytes': inum(manifest.get('data_size_bytes', metadata.get('model_size_bytes',''))),
        'technology': manifest.get('technology', metadata.get('technology', metadata.get('radio_profile',''))),
        'testbed': testbed, 'seed': manifest.get('seed', metadata.get('seed','')),
        'radio_profile': metadata.get('radio_profile',''),
        'node_count': node_count if node_count>0 else extract_node_count(testbed),
    }

def split_rx_seconds(row: Dict[str,str], prefix=''):
    out={s:fnum(row.get(f'{prefix}rx_on_{s}_ms'))/1000.0 for s in STAGES}
    cca=fnum(row.get(f'{prefix}csma_cca_ms'))/1000.0
    cad=fnum(row.get(f'{prefix}cad_scan_ms'))/1000.0
    total=fnum(row.get(f'{prefix}rx_on_ms'))/1000.0
    if total<=0: total=sum(out.values())+cca+cad
    return out, cca, cad, total

def mj(v, ma, sec): return v*ma*sec

def make_node_row(r, ctx, prof_name, prof, mcu_ma, mcu_v):
    v=prof['voltage']; tx_ma=prof['tx_current_ma']; rx_ma=prof['rx_current_ma']; cca_ma=prof['cca_current_ma']; cad_ma=prof['cad_current_ma']
    rx_s, cca_s, cad_s, total_rx_s=split_rx_seconds(r)
    row=dict(ctx); row.update({'energy_profile':prof_name,'nid':inum(r.get('nid')),'x':fnum(r.get('x')),'y':fnum(r.get('y')),'is_gateway':inum(r.get('is_gateway')),'updated':inum(r.get('updated')),'version':inum(r.get('version')),'voltage_v':v,'tx_current_ma':tx_ma,'rx_current_ma':rx_ma,'cca_current_ma':cca_ma,'cad_current_ma':cad_ma,'mcu_current_ma':mcu_ma,'mcu_voltage_v':mcu_v,'rx_stage_method':'direct_logged_deluge_rx_by_stage'})
    total_tx_s=0.0; radio_tx=0.0; radio_rx=0.0; mcu_tx=0.0; mcu_rx=0.0
    for s in STAGES:
        tx_s=fnum(r.get(TX_USED_COLS[s])); total_tx_s += tx_s
        row[f'{s}_tx_s']=tx_s; row[f'{s}_tx_mj']=mj(v,tx_ma,tx_s); row[f'{s}_mcu_tx_mj']=mj(mcu_v,mcu_ma,tx_s)
        row[f'{s}_radio_plus_mcu_tx_mj']=row[f'{s}_tx_mj']+row[f'{s}_mcu_tx_mj']
        row[f'{s}_rx_est_s']=rx_s[s]; row[f'{s}_rx_est_mj']=mj(v,rx_ma,rx_s[s]); row[f'{s}_mcu_rx_est_mj']=mj(mcu_v,mcu_ma,rx_s[s])
        row[f'{s}_radio_plus_mcu_rx_est_mj']=row[f'{s}_rx_est_mj']+row[f'{s}_mcu_rx_est_mj']
        row[f'{s}_total_energy_with_mcu_est_mj']=row[f'{s}_radio_plus_mcu_tx_mj']+row[f'{s}_radio_plus_mcu_rx_est_mj']
        radio_tx += row[f'{s}_tx_mj']; radio_rx += row[f'{s}_rx_est_mj']; mcu_tx += row[f'{s}_mcu_tx_mj']
    cca_mj=mj(v,cca_ma,cca_s); cad_mj=mj(v,cad_ma,cad_s)
    radio_rx += cca_mj + cad_mj; mcu_rx=mj(mcu_v,mcu_ma,total_rx_s)
    row.update({'total_tx_s':total_tx_s,'total_rx_on_s':total_rx_s,'total_rx_busy_s':fnum(r.get('rx_busy_ms'))/1000.0,'total_rx_stage_s':sum(rx_s.values()),'csma_cca_s':cca_s,'cad_scan_s':cad_s,'total_tx_mj':radio_tx,'total_rx_mj':radio_rx,'total_energy_mj':radio_tx+radio_rx,'csma_cca_mj':cca_mj,'cad_scan_mj':cad_mj,'total_mcu_tx_mj':mcu_tx,'total_mcu_rx_mj':mcu_rx,'total_mcu_energy_mj':mcu_tx+mcu_rx,'total_radio_plus_mcu_tx_mj':radio_tx+mcu_tx,'total_radio_plus_mcu_rx_mj':radio_rx+mcu_rx,'total_radio_plus_mcu_energy_mj':radio_tx+radio_rx+mcu_tx+mcu_rx,'total_tx_energy_with_mcu_mj':radio_tx+mcu_tx,'total_rx_energy_with_mcu_mj':radio_rx+mcu_rx,'total_energy_with_mcu_mj':radio_tx+radio_rx+mcu_tx+mcu_rx})
    return row

def process(args):
    profiles=load_energy_profiles(Path(args.energy_profiles))
    run_dirs=discover_run_dirs(Path(args.input), args.recursive)
    out_root=Path(args.output_dir) if args.output_dir else Path(args.input)/'energy_reports_combined'
    node_rows=[]; run_summary=[]; count_groups=defaultdict(list)
    for rd in run_dirs:
        meta=load_key_value_csv(rd/'metadata.csv'); rows=read_csv_rows(rd/'node_summary.csv')
        ctx=run_context(rd, meta, len(rows)); prof_name, prof=profile_for(meta, profiles)
        per=[]
        for r in rows:
            nr=make_node_row(r, ctx, prof_name, prof, args.mcu_current_ma, args.mcu_voltage_v)
            per.append(nr); node_rows.append(nr)
        if not args.skip_per_run_reports:
            write_csv_rows(rd/'energy_node_summary.csv', per)
        mean_total=sum(x['total_energy_with_mcu_mj'] for x in per)/len(per) if per else 0.0
        rs={**ctx,'nodes':len(per),'mean_total_energy_with_mcu_mj':mean_total}
        run_summary.append(rs); count_groups[(ctx['technology'],ctx['data_size_bytes'],ctx['node_count'])].extend(per)
    count_summary=[]
    for (tech,ds,nc), vals in sorted(count_groups.items()):
        count_summary.append({'technology':tech,'data_size_bytes':ds,'node_count':nc,'node_rows':len(vals),'mean_total_energy_with_mcu_mj':sum(v['total_energy_with_mcu_mj'] for v in vals)/len(vals) if vals else 0.0})
    write_csv_rows(out_root/'node_energy_summary.csv', node_rows)
    write_csv_rows(out_root/'run_energy_summary.csv', run_summary)
    write_csv_rows(out_root/'node_count_energy_summary.csv', count_summary)
    print(f'Wrote Deluge energy reports to {out_root}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', default='output_deluge'); ap.add_argument('--recursive', action='store_true')
    ap.add_argument('--energy-profiles', default='energy_profiles.csv'); ap.add_argument('--output-dir', default=None)
    ap.add_argument('--skip-per-run-reports', action='store_true')
    ap.add_argument('--mcu-current-ma', type=float, default=0.0); ap.add_argument('--mcu-voltage-v', type=float, default=3.3)
    process(ap.parse_args())
if __name__=='__main__': main()
