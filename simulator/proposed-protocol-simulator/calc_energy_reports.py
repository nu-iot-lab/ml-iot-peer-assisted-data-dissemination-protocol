#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming energy report generator for the wireless network simulator.

This version is designed for large batch folders. It processes one run directory
at a time, writes per-run reports immediately, and stores only small aggregate
statistics in memory. It does NOT build giant combined per-node/per-day lists by
default.

Expected per-run files:
  - metadata.csv
  - node_summary.csv
Optional:
  - batch_manifest.csv

Energy profile CSV columns:
  radio_profile,tx_current_ma,rx_current_ma,voltage[,cca_current_ma,cad_current_ma,notes]

The current simulator logs RX/listening time by protocol mode in node_summary.csv:
  rx_on_adv_ms, rx_on_upd_ms, rx_on_data_ms, rx_on_miss_ms, rx_on_repair_ms,
  csma_cca_ms, cad_scan_ms, and per-day dX_* variants.
When these columns are present, this script uses them directly. Older summaries
fall back to the old phase-duration split of rx_on_ms.

Energy formula:
  Energy [mJ] = Voltage [V] * Current [mA] * Time [s]

Outputs per run, unless --skip-per-run-reports is set:
  - energy_node_summary.csv
  - energy_per_day_per_node.csv
  - energy_maxima.csv

Batch-level outputs, written to <input>/energy_reports_combined by default:
  - run_energy_summary.csv
  - node_count_energy_summary.csv      concise average end-of-simulation energy by node-count category
  - node_energy_summary.csv            concise end-of-simulation energy per node
  - run_count_audit.csv
  - combined_energy_maxima.csv

Optional huge combined outputs:
  - combined_energy_node_summary.csv       only with --write-combined-node-summary
  - combined_energy_per_day_per_node.csv   only with --write-combined-day-summary
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

STAGES = ["adv", "upd", "data", "miss", "repair"]
TX_USED_COLS = {
    "adv": "adv_used_s",
    "upd": "upd_used_s",
    "data": "data_used_s",
    "miss": "miss_used_s",
    "repair": "repair_used_s",
}
DAY_TX_COUNT_COLS = {
    "adv": "tx_adv",
    "upd": "tx_upd",
    "data": "tx_data",
    "miss": "tx_miss",
    "repair": "tx_repair",
    "end": "tx_end",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def iter_csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


def write_csv_rows(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


class StreamingCsvWriter:
    """CSV writer that opens lazily and writes rows without retaining them."""

    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        self._file = None
        self._writer = None

    def write(self, row: Dict[str, object]) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def fnum(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def inum(value: object, default: int = 0) -> int:
    return int(round(fnum(value, float(default))))


def load_key_value_csv(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    if not ({"key", "value"} <= set(rows[0].keys())):
        return dict(rows[0])
    out: Dict[str, str] = {}
    for r in rows:
        k = str(r.get("key", "")).strip()
        if k:
            out[k] = str(r.get("value", ""))
    return out


def load_energy_profiles(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(path)
    required = {"radio_profile", "tx_current_ma", "rx_current_ma", "voltage"}
    if not rows or not required <= set(rows[0].keys()):
        raise SystemExit(f"Energy profile file must contain columns: {', '.join(sorted(required))}. Got: {path}")
    profiles: Dict[str, Dict[str, float]] = {}
    for r in rows:
        name = str(r.get("radio_profile", "")).strip()
        if not name:
            continue
        rx_ma = fnum(r.get("rx_current_ma"))
        profiles[name] = {
            "tx_current_ma": fnum(r.get("tx_current_ma")),
            "rx_current_ma": rx_ma,
            "cca_current_ma": fnum(r.get("cca_current_ma"), rx_ma),
            "cad_current_ma": fnum(r.get("cad_current_ma"), rx_ma),
            "voltage": fnum(r.get("voltage")),
        }
    return profiles


def profile_value(profile: Dict[str, float], key: str, fallback_key: str = "rx_current_ma") -> float:
    value = profile.get(key)
    if value is None or float(value) <= 0.0:
        return float(profile.get(fallback_key, 0.0))
    return float(value)

def mj(voltage: float, current_ma: float, seconds: float) -> float:
    return voltage * current_ma * seconds


def discover_run_dirs(root: Path, recursive: bool) -> List[Path]:
    if (root / "metadata.csv").is_file() and (root / "node_summary.csv").is_file():
        return [root]
    if not recursive:
        raise SystemExit(f"No metadata.csv + node_summary.csv found in {root}. Use --recursive for a batch tree.")
    out = []
    for meta in root.rglob("metadata.csv"):
        d = meta.parent
        if (d / "node_summary.csv").is_file():
            out.append(d)
    return sorted(set(out))


def run_context(run_dir: Path, metadata: Dict[str, str]) -> Dict[str, object]:
    manifest_path = run_dir / "batch_manifest.csv"
    manifest = load_key_value_csv(manifest_path) if manifest_path.is_file() else {}
    return {
        "run_dir": str(run_dir),
        "rep": manifest.get("rep", ""),
        "data_size_bytes": manifest.get("data_size_bytes", metadata.get("model_size_bytes", "")),
        "technology": manifest.get("technology", metadata.get("radio_profile", "")),
        "testbed": Path(manifest.get("testbed_path", metadata.get("testbed", ""))).name,
        "seed": manifest.get("seed", metadata.get("seed", "")),
        "radio_profile": metadata.get("radio_profile", ""),
    }


def profile_for_run(metadata: Dict[str, str], profiles: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    for c in [metadata.get("radio_profile", ""), metadata.get("lora_radio", "")]:
        c = str(c).strip()
        if c and c in profiles:
            return c, profiles[c]
    raise SystemExit(
        "No matching energy profile for run radio_profile="
        f"{metadata.get('radio_profile')!r}. Available: {', '.join(sorted(profiles))}"
    )


def derive_airtimes(metadata: Dict[str, str], node_rows: List[Dict[str, str]]) -> Dict[str, float]:
    def derive_from_used(stage: str, count_col: str) -> float:
        used = sum(fnum(r.get(TX_USED_COLS[stage])) for r in node_rows)
        cnt = sum(fnum(r.get(count_col)) for r in node_rows)
        return used / cnt if cnt > 0 else 0.0

    data_air = fnum(metadata.get("auto_data_repair_data_airtime_s"))
    if data_air <= 0:
        used = sum(fnum(r.get("data_used_s")) + fnum(r.get("repair_used_s")) for r in node_rows)
        cnt = sum(fnum(r.get("tx_data")) + fnum(r.get("tx_repair")) + fnum(r.get("tx_end")) for r in node_rows)
        data_air = used / cnt if cnt > 0 else 0.0

    miss_air = fnum(metadata.get("auto_miss_packet_airtime_s")) or derive_from_used("miss", "tx_miss")
    adv_air = fnum(metadata.get("auto_adv_packet_airtime_s")) or derive_from_used("adv", "tx_adv")
    upd_air = fnum(metadata.get("auto_upd_packet_airtime_s")) or derive_from_used("upd", "tx_upd") or adv_air
    return {"adv": adv_air, "upd": upd_air, "data": data_air, "miss": miss_air, "repair": data_air, "end": data_air}


def phase_duration_weights(metadata: Dict[str, str]) -> Dict[str, float]:
    durations = {
        "adv": fnum(metadata.get("adv_budget_s")),
        "upd": fnum(metadata.get("upd_listen_s")),
        "data": fnum(metadata.get("data_budget_s")),
        "miss": max(fnum(metadata.get("miss_budget_s")), fnum(metadata.get("miss_listen_s"))),
        "repair": fnum(metadata.get("repair_budget_s")),
    }
    if durations["upd"] <= 0:
        durations["upd"] = fnum(metadata.get("upd_listen_ms")) / 1000.0
    if durations["miss"] <= 0:
        durations["miss"] = fnum(metadata.get("miss_listen_ms")) / 1000.0
    total = sum(max(v, 0.0) for v in durations.values())
    if total <= 0:
        return {s: 1.0 / len(STAGES) for s in STAGES}
    return {s: max(durations[s], 0.0) / total for s in STAGES}


def day_numbers(node_rows: List[Dict[str, str]]) -> List[int]:
    if not node_rows:
        return []
    days = set()
    pat = re.compile(r"^d(\d+)_tx_adv$")
    for col in node_rows[0].keys():
        m = pat.match(col)
        if m:
            days.add(int(m.group(1)))
    return sorted(days)


def split_end_packets(end_count: float, data_count: float, repair_count: float) -> Tuple[float, float]:
    if end_count <= 0:
        return 0.0, 0.0
    denom = data_count + repair_count
    if denom <= 0:
        return end_count, 0.0
    end_data = end_count * (data_count / denom)
    return end_data, end_count - end_data


def safe_int_from_context(value: object) -> int:
    return inum(value, 0)


def extract_node_count_from_name(name: str) -> int:
    """Fallback parser for names like testbed_2km_100_1.csv or .../_200."""
    stem = Path(name).stem
    nums = [int(x) for x in re.findall(r"(?:^|_)(\d+)(?=_|$)", stem)]
    if len(nums) >= 2:
        return nums[-2]
    if len(nums) == 1:
        return nums[0]
    return 0


@dataclass
class GroupStats:
    run_dirs: set[str] = field(default_factory=set)
    reps: set[str] = field(default_factory=set)
    testbeds: set[str] = field(default_factory=set)
    node_count_sum: int = 0
    node_rows: int = 0
    # node-weighted sums across all nodes of all runs
    node_sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    # run-weighted sums of per-run means
    run_mean_sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    max_values: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add_run(self, ctx: Dict[str, object], node_count: int, per_node_sum: Dict[str, float], maxima: Dict[str, float]) -> None:
        self.run_dirs.add(str(ctx.get("run_dir", "")))
        self.reps.add(str(ctx.get("rep", "")))
        self.testbeds.add(str(ctx.get("testbed", "")))
        self.node_count_sum += node_count
        self.node_rows += node_count
        for k, v in per_node_sum.items():
            self.node_sums[k] += v
            self.run_mean_sums[k] += (v / node_count) if node_count > 0 else 0.0
        for k, v in maxima.items():
            self.max_values[k] = max(self.max_values.get(k, 0.0), v)

    @property
    def runs_found(self) -> int:
        return len(self.run_dirs)


def ordered_node_fields(include_stage: bool = True) -> List[str]:
    fields = [
        "run_dir", "rep", "data_size_bytes", "technology", "testbed", "seed", "radio_profile", "energy_profile",
        "nid", "x", "y", "is_gateway", "updated", "version",
        "voltage_v", "tx_current_ma", "rx_current_ma", "cca_current_ma", "cad_current_ma", "mcu_current_ma", "mcu_voltage_v", "rx_stage_method",
    ]
    if include_stage:
        for s in STAGES:
            fields += [
                f"{s}_tx_s", f"{s}_tx_mj", f"{s}_mcu_tx_mj", f"{s}_radio_plus_mcu_tx_mj",
                f"{s}_rx_est_s", f"{s}_rx_est_mj", f"{s}_mcu_rx_est_mj", f"{s}_radio_plus_mcu_rx_est_mj",
                f"{s}_total_energy_with_mcu_est_mj",
            ]
    fields += [
        "total_tx_s", "total_rx_on_s", "total_rx_busy_s", "total_rx_stage_s", "csma_cca_s", "cad_scan_s",
        "total_tx_mj", "total_rx_mj", "total_energy_mj", "csma_cca_mj", "cad_scan_mj",
        "total_mcu_tx_mj", "total_mcu_rx_mj", "total_mcu_energy_mj",
        "total_radio_plus_mcu_tx_mj", "total_radio_plus_mcu_rx_mj", "total_radio_plus_mcu_energy_mj",
        "total_tx_energy_with_mcu_mj", "total_rx_energy_with_mcu_mj", "total_energy_with_mcu_mj",
    ]
    return fields


def ordered_day_fields() -> List[str]:
    fields = [
        "run_dir", "rep", "data_size_bytes", "technology", "testbed", "seed", "radio_profile", "energy_profile",
        "day", "nid", "x", "y", "voltage_v", "tx_current_ma", "rx_current_ma", "cca_current_ma", "cad_current_ma", "mcu_current_ma", "mcu_voltage_v",
        "tx_stage_method", "end_split_method", "rx_stage_method",
    ]
    for s in STAGES:
        fields += [
            f"{s}_tx_count", f"{s}_tx_s", f"{s}_tx_mj", f"{s}_mcu_tx_mj", f"{s}_radio_plus_mcu_tx_mj",
            f"{s}_rx_est_s", f"{s}_rx_est_mj", f"{s}_mcu_rx_est_mj", f"{s}_radio_plus_mcu_rx_est_mj",
            f"{s}_total_energy_with_mcu_est_mj",
        ]
    fields += [
        "end_tx_count_total", "end_tx_count_assigned_to_data", "end_tx_count_assigned_to_repair",
        "total_tx_s", "total_rx_on_s", "total_rx_busy_s", "total_rx_stage_s", "csma_cca_s", "cad_scan_s",
        "total_tx_mj", "total_rx_mj", "total_energy_mj", "csma_cca_mj", "cad_scan_mj",
        "total_mcu_tx_mj", "total_mcu_rx_mj", "total_mcu_energy_mj",
        "total_radio_plus_mcu_tx_mj", "total_radio_plus_mcu_rx_mj", "total_radio_plus_mcu_energy_mj",
        "total_tx_energy_with_mcu_mj", "total_rx_energy_with_mcu_mj", "total_energy_with_mcu_mj",
    ]
    return fields


def energy_metric_names() -> List[str]:
    names = [
        "total_tx_mj", "total_rx_mj", "total_energy_mj", "csma_cca_mj", "cad_scan_mj",
        "total_mcu_tx_mj", "total_mcu_rx_mj", "total_mcu_energy_mj",
        "total_radio_plus_mcu_tx_mj", "total_radio_plus_mcu_rx_mj", "total_radio_plus_mcu_energy_mj",
        "total_energy_with_mcu_mj",
    ]
    for s in STAGES:
        names += [
            f"{s}_tx_mj", f"{s}_rx_est_mj",
            f"{s}_mcu_tx_mj", f"{s}_mcu_rx_est_mj",
            f"{s}_radio_plus_mcu_tx_mj", f"{s}_radio_plus_mcu_rx_est_mj",
            f"{s}_total_energy_with_mcu_est_mj",
        ]
    return names



def has_split_rx_columns(r: Dict[str, str], prefix: str = "") -> bool:
    return any(str(r.get(f"{prefix}rx_on_{stage}_ms", "")).strip() != "" for stage in STAGES)


def split_rx_seconds(r: Dict[str, str], rx_weights: Dict[str, float], prefix: str = "") -> Tuple[Dict[str, float], float, float, str]:
    """Return per-stage RX seconds, CSMA/CCA seconds, CAD seconds, and method.

    Current simulator outputs direct split columns. Older outputs only have rx_on_ms,
    so those are estimated using phase-duration weights.
    """
    total_rx_s = fnum(r.get(f"{prefix}rx_on_ms")) / 1000.0
    if has_split_rx_columns(r, prefix):
        by_stage = {stage: fnum(r.get(f"{prefix}rx_on_{stage}_ms")) / 1000.0 for stage in STAGES}
        csma_cca_s = fnum(r.get(f"{prefix}csma_cca_ms")) / 1000.0
        cad_scan_s = fnum(r.get(f"{prefix}cad_scan_ms")) / 1000.0
        return by_stage, csma_cca_s, cad_scan_s, "direct_logged_rx_by_stage"
    by_stage = {stage: total_rx_s * rx_weights[stage] for stage in STAGES}
    return by_stage, 0.0, 0.0, "estimated_by_global_phase_duration_legacy"


def rx_energy_mj(v: float, rx_ma: float, cca_ma: float, cad_ma: float, stage_rx_s: Dict[str, float], csma_cca_s: float, cad_scan_s: float) -> Tuple[float, float, float, float]:
    stage_rx_mj = sum(mj(v, rx_ma, stage_rx_s[s]) for s in STAGES)
    cca_mj = mj(v, cca_ma, csma_cca_s)
    cad_mj = mj(v, cad_ma, cad_scan_s)
    return stage_rx_mj + cca_mj + cad_mj, stage_rx_mj, cca_mj, cad_mj

def make_aggregate_node_row(
    r: Dict[str, str],
    ctx: Dict[str, object],
    profile_name: str,
    v: float,
    tx_ma: float,
    rx_ma: float,
    cca_ma: float,
    cad_ma: float,
    mcu_ma: float,
    mcu_v: float,
    rx_weights: Dict[str, float],
) -> Dict[str, object]:
    stage_rx_s, csma_cca_s, cad_scan_s, rx_method = split_rx_seconds(r, rx_weights)
    base = dict(ctx)
    base.update({
        "energy_profile": profile_name,
        "nid": inum(r.get("nid")),
        "x": fnum(r.get("x")),
        "y": fnum(r.get("y")),
        "is_gateway": inum(r.get("is_gateway")),
        "updated": inum(r.get("updated")),
        "version": inum(r.get("version")),
        "voltage_v": v,
        "tx_current_ma": tx_ma,
        "rx_current_ma": rx_ma,
        "cca_current_ma": cca_ma,
        "cad_current_ma": cad_ma,
        "mcu_current_ma": mcu_ma,
        "mcu_voltage_v": mcu_v,
        "rx_stage_method": rx_method,
    })
    row = dict(base)
    total_tx_s = 0.0
    for s in STAGES:
        tx_s = fnum(r.get(TX_USED_COLS[s]))
        total_tx_s += tx_s
        row[f"{s}_tx_s"] = tx_s
        row[f"{s}_tx_mj"] = mj(v, tx_ma, tx_s)
        row[f"{s}_mcu_tx_mj"] = mj(mcu_v, mcu_ma, tx_s)
        row[f"{s}_radio_plus_mcu_tx_mj"] = row[f"{s}_tx_mj"] + row[f"{s}_mcu_tx_mj"]

    for s in STAGES:
        row[f"{s}_rx_est_s"] = stage_rx_s[s]
        row[f"{s}_rx_est_mj"] = mj(v, rx_ma, stage_rx_s[s])
        row[f"{s}_mcu_rx_est_mj"] = mj(mcu_v, mcu_ma, stage_rx_s[s])
        row[f"{s}_radio_plus_mcu_rx_est_mj"] = row[f"{s}_rx_est_mj"] + row[f"{s}_mcu_rx_est_mj"]
        row[f"{s}_total_energy_with_mcu_est_mj"] = row[f"{s}_radio_plus_mcu_tx_mj"] + row[f"{s}_radio_plus_mcu_rx_est_mj"]

    total_rx_s = fnum(r.get("rx_on_ms")) / 1000.0
    total_rx_stage_s = sum(stage_rx_s.values())
    # Current simulator's rx_on_ms already includes protocol RX windows + CSMA/CCA + CAD.
    # Prefer the logged total for total time; fall back to reconstructed sum for old/partial files.
    if total_rx_s <= 0.0:
        total_rx_s = total_rx_stage_s + csma_cca_s + cad_scan_s
    radio_rx, _stage_rx_mj, cca_mj, cad_mj = rx_energy_mj(v, rx_ma, cca_ma, cad_ma, stage_rx_s, csma_cca_s, cad_scan_s)
    radio_tx = mj(v, tx_ma, total_tx_s)
    mcu_tx = mj(mcu_v, mcu_ma, total_tx_s)
    mcu_rx_mj = mj(mcu_v, mcu_ma, total_rx_s)
    row.update({
        "total_tx_s": total_tx_s,
        "total_rx_on_s": total_rx_s,
        "total_rx_busy_s": fnum(r.get("rx_busy_ms")) / 1000.0,
        "total_rx_stage_s": total_rx_stage_s,
        "csma_cca_s": csma_cca_s,
        "cad_scan_s": cad_scan_s,
        "total_tx_mj": radio_tx,
        "total_rx_mj": radio_rx,
        "csma_cca_mj": cca_mj,
        "cad_scan_mj": cad_mj,
        "total_energy_mj": radio_tx + radio_rx,
        "total_mcu_tx_mj": mcu_tx,
        "total_mcu_rx_mj": mcu_rx_mj,
        "total_mcu_energy_mj": mcu_tx + mcu_rx_mj,
        "total_radio_plus_mcu_tx_mj": radio_tx + mcu_tx,
        "total_radio_plus_mcu_rx_mj": radio_rx + mcu_rx_mj,
        "total_radio_plus_mcu_energy_mj": radio_tx + radio_rx + mcu_tx + mcu_rx_mj,
        "total_tx_energy_with_mcu_mj": radio_tx + mcu_tx,
        "total_rx_energy_with_mcu_mj": radio_rx + mcu_rx_mj,
        "total_energy_with_mcu_mj": radio_tx + radio_rx + mcu_tx + mcu_rx_mj,
    })
    return row

def make_day_row(
    r: Dict[str, str],
    d: int,
    ctx: Dict[str, object],
    profile_name: str,
    v: float,
    tx_ma: float,
    rx_ma: float,
    cca_ma: float,
    cad_ma: float,
    mcu_ma: float,
    mcu_v: float,
    airt: Dict[str, float],
    rx_weights: Dict[str, float],
) -> Dict[str, object]:
    prefix = f"d{d}_"
    stage_rx_s, csma_cca_s, cad_scan_s, rx_method = split_rx_seconds(r, rx_weights, prefix=prefix)
    base = dict(ctx)
    base.update({
        "energy_profile": profile_name,
        "day": d,
        "nid": inum(r.get("nid")),
        "x": fnum(r.get("x")),
        "y": fnum(r.get("y")),
        "voltage_v": v,
        "tx_current_ma": tx_ma,
        "rx_current_ma": rx_ma,
        "cca_current_ma": cca_ma,
        "cad_current_ma": cad_ma,
        "mcu_current_ma": mcu_ma,
        "mcu_voltage_v": mcu_v,
        "tx_stage_method": "counts_times_metadata_airtime",
        "end_split_method": "proportional_to_data_and_repair_packet_counts",
        "rx_stage_method": rx_method,
    })
    counts = {k: fnum(r.get(f"d{d}_{col}")) for k, col in DAY_TX_COUNT_COLS.items()}
    end_data, end_repair = split_end_packets(counts["end"], counts["data"], counts["repair"])
    tx_s_by_stage = {
        "adv": counts["adv"] * airt["adv"],
        "upd": counts["upd"] * airt["upd"],
        "data": (counts["data"] + end_data) * airt["data"],
        "miss": counts["miss"] * airt["miss"],
        "repair": (counts["repair"] + end_repair) * airt["repair"],
    }
    total_tx_s = sum(tx_s_by_stage.values())
    row = dict(base)
    for s in STAGES:
        row[f"{s}_tx_count"] = counts[s]
        row[f"{s}_tx_s"] = tx_s_by_stage[s]
        row[f"{s}_tx_mj"] = mj(v, tx_ma, tx_s_by_stage[s])
        row[f"{s}_mcu_tx_mj"] = mj(mcu_v, mcu_ma, tx_s_by_stage[s])
        row[f"{s}_radio_plus_mcu_tx_mj"] = row[f"{s}_tx_mj"] + row[f"{s}_mcu_tx_mj"]
        row[f"{s}_rx_est_s"] = stage_rx_s[s]
        row[f"{s}_rx_est_mj"] = mj(v, rx_ma, stage_rx_s[s])
        row[f"{s}_mcu_rx_est_mj"] = mj(mcu_v, mcu_ma, stage_rx_s[s])
        row[f"{s}_radio_plus_mcu_rx_est_mj"] = row[f"{s}_rx_est_mj"] + row[f"{s}_mcu_rx_est_mj"]
        row[f"{s}_total_energy_with_mcu_est_mj"] = row[f"{s}_radio_plus_mcu_tx_mj"] + row[f"{s}_radio_plus_mcu_rx_est_mj"]

    total_rx_s = fnum(r.get(f"d{d}_rx_on_ms")) / 1000.0
    total_rx_stage_s = sum(stage_rx_s.values())
    if total_rx_s <= 0.0:
        total_rx_s = total_rx_stage_s + csma_cca_s + cad_scan_s
    radio_rx, _stage_rx_mj, cca_mj, cad_mj = rx_energy_mj(v, rx_ma, cca_ma, cad_ma, stage_rx_s, csma_cca_s, cad_scan_s)
    radio_tx = mj(v, tx_ma, total_tx_s)
    mcu_tx = mj(mcu_v, mcu_ma, total_tx_s)
    mcu_rx_mj = mj(mcu_v, mcu_ma, total_rx_s)
    row.update({
        "end_tx_count_total": counts["end"],
        "end_tx_count_assigned_to_data": end_data,
        "end_tx_count_assigned_to_repair": end_repair,
        "total_tx_s": total_tx_s,
        "total_rx_on_s": total_rx_s,
        "total_rx_busy_s": fnum(r.get(f"d{d}_rx_busy_ms")) / 1000.0,
        "total_rx_stage_s": total_rx_stage_s,
        "csma_cca_s": csma_cca_s,
        "cad_scan_s": cad_scan_s,
        "total_tx_mj": radio_tx,
        "total_rx_mj": radio_rx,
        "csma_cca_mj": cca_mj,
        "cad_scan_mj": cad_mj,
        "total_energy_mj": radio_tx + radio_rx,
        "total_mcu_tx_mj": mcu_tx,
        "total_mcu_rx_mj": mcu_rx_mj,
        "total_mcu_energy_mj": mcu_tx + mcu_rx_mj,
        "total_radio_plus_mcu_tx_mj": radio_tx + mcu_tx,
        "total_radio_plus_mcu_rx_mj": radio_rx + mcu_rx_mj,
        "total_radio_plus_mcu_energy_mj": radio_tx + radio_rx + mcu_tx + mcu_rx_mj,
        "total_tx_energy_with_mcu_mj": radio_tx + mcu_tx,
        "total_rx_energy_with_mcu_mj": radio_rx + mcu_rx_mj,
        "total_energy_with_mcu_mj": radio_tx + radio_rx + mcu_tx + mcu_rx_mj,
    })
    return row

def maxima_from_rows(ctx: Dict[str, object], rows: List[Dict[str, object]], scope: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for metric in energy_metric_names():
        if not rows:
            continue
        best = max(rows, key=lambda r: fnum(r.get(metric)))
        stage = "total"
        direction = "tx_plus_rx"
        m = re.match(r"^(adv|upd|data|miss|repair)_(.+)$", metric)
        if m:
            stage = m.group(1)
            rest = m.group(2)
            direction = "rx_est" if "rx" in rest else "tx" if "tx" in rest else "tx_plus_rx"
        elif "_tx_" in metric or metric.endswith("_tx_mj"):
            direction = "tx"
        elif "_rx_" in metric or metric.endswith("_rx_mj"):
            direction = "rx"
        row = dict(ctx)
        row.update({
            "scope": scope,
            "direction": direction,
            "stage": stage,
            "metric": metric,
            "max_value_mj": fnum(best.get(metric)),
            "nid": best.get("nid", ""),
            "day": best.get("day", ""),
            "x": best.get("x", ""),
            "y": best.get("y", ""),
        })
        out.append(row)
    return out


def process_run(
    run_dir: Path,
    profiles: Dict[str, Dict[str, float]],
    mcu_current_ma: float,
    mcu_voltage_override: Optional[float],
    write_per_run: bool,
    combined_node_writer: Optional[StreamingCsvWriter],
    combined_day_writer: Optional[StreamingCsvWriter],
    compact_node_writer: Optional[StreamingCsvWriter],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    metadata = load_key_value_csv(run_dir / "metadata.csv")
    node_rows = read_csv_rows(run_dir / "node_summary.csv")
    profile_name, ep = profile_for_run(metadata, profiles)
    ctx = run_context(run_dir, metadata)
    ctx["energy_profile"] = profile_name

    v = ep["voltage"]
    tx_ma = ep["tx_current_ma"]
    rx_ma = ep["rx_current_ma"]
    cca_ma = profile_value(ep, "cca_current_ma")
    cad_ma = profile_value(ep, "cad_current_ma")
    mcu_v = float(mcu_voltage_override) if mcu_voltage_override is not None else v
    airt = derive_airtimes(metadata, node_rows)
    rx_weights = phase_duration_weights(metadata)
    days = day_numbers(node_rows)
    node_count = len(node_rows) or extract_node_count_from_name(str(ctx.get("testbed", "")))

    aggregate_rows: List[Dict[str, object]] = []
    per_day_rows: List[Dict[str, object]] = []

    per_node_sums: Dict[str, float] = defaultdict(float)
    per_node_maxima: Dict[str, float] = defaultdict(float)

    for r in node_rows:
        row = make_aggregate_node_row(r, ctx, profile_name, v, tx_ma, rx_ma, cca_ma, cad_ma, mcu_current_ma, mcu_v, rx_weights)
        aggregate_rows.append(row)
        if combined_node_writer is not None:
            combined_node_writer.write(row)
        if compact_node_writer is not None:
            compact_node_writer.write(make_compact_node_row(row, node_count))
        for metric in energy_metric_names():
            val = fnum(row.get(metric))
            per_node_sums[metric] += val
            per_node_maxima[metric] = max(per_node_maxima.get(metric, 0.0), val)

    for r in node_rows:
        for d in days:
            row = make_day_row(r, d, ctx, profile_name, v, tx_ma, rx_ma, cca_ma, cad_ma, mcu_current_ma, mcu_v, airt, rx_weights)
            per_day_rows.append(row)
            if combined_day_writer is not None:
                combined_day_writer.write(row)

    maxima_rows = maxima_from_rows(ctx, aggregate_rows, "aggregate_per_node") + maxima_from_rows(ctx, per_day_rows, "per_day_per_node")

    if write_per_run:
        write_csv_rows(run_dir / "energy_node_summary.csv", aggregate_rows, ordered_node_fields())
        write_csv_rows(run_dir / "energy_per_day_per_node.csv", per_day_rows, ordered_day_fields())
        write_csv_rows(run_dir / "energy_maxima.csv", maxima_rows)

    run_summary = dict(ctx)
    run_summary.update({
        "node_count": node_count,
        "days_reported": len(days),
        "voltage_v": v,
        "tx_current_ma": tx_ma,
        "rx_current_ma": rx_ma,
        "cca_current_ma": cca_ma,
        "cad_current_ma": cad_ma,
        "mcu_current_ma": mcu_current_ma,
        "mcu_voltage_v": mcu_v,
    })
    for metric in energy_metric_names():
        total = per_node_sums.get(metric, 0.0)
        run_summary[f"sum_{metric}"] = total
        run_summary[f"avg_per_node_{metric}"] = total / node_count if node_count > 0 else 0.0
        run_summary[f"max_node_{metric}"] = per_node_maxima.get(metric, 0.0)

    return run_summary, maxima_rows


def compact_node_fieldnames() -> List[str]:
    """Compact per-node end-of-simulation summary.

    This is intentionally much smaller than combined_energy_node_summary.csv
    and excludes run_dir. It is meant for plotting/statistics over final node
    energy, not per-day energy.
    """
    return [
        "rep",
        "data_size_bytes",
        "technology",
        "testbed",
        "radio_profile",
        "energy_profile",
        "node_count",
        "nid",
        "radio_tx_mj",
        "radio_rx_mj",
        "csma_cca_mj",
        "cad_scan_mj",
        "radio_mj",
        "mcu_tx_mj",
        "mcu_rx_mj",
        "mcu_mj",
        "total_mj",
    ]


def make_compact_node_row(row: Dict[str, object], node_count: int) -> Dict[str, object]:
    """Convert the full aggregate node row into the compact per-node output."""
    return {
        "rep": row.get("rep", ""),
        "data_size_bytes": row.get("data_size_bytes", ""),
        "technology": row.get("technology", ""),
        "testbed": row.get("testbed", ""),
        "radio_profile": row.get("radio_profile", ""),
        "energy_profile": row.get("energy_profile", ""),
        "node_count": node_count,
        "nid": row.get("nid", ""),
        "radio_tx_mj": row.get("total_tx_mj", 0.0),
        "radio_rx_mj": row.get("total_rx_mj", 0.0),
        "csma_cca_mj": row.get("csma_cca_mj", 0.0),
        "cad_scan_mj": row.get("cad_scan_mj", 0.0),
        "radio_mj": row.get("total_energy_mj", 0.0),
        "mcu_tx_mj": row.get("total_mcu_tx_mj", 0.0),
        "mcu_rx_mj": row.get("total_mcu_rx_mj", 0.0),
        "mcu_mj": row.get("total_mcu_energy_mj", 0.0),
        "total_mj": row.get("total_energy_with_mcu_mj", 0.0),
    }


def summary_fieldnames() -> List[str]:
    """Small high-level summary requested for comparing node-count categories.

    The averages are computed from aggregate end-of-simulation per-node rows,
    not from per-day rows. This avoids per-day averages being dominated by days
    where many nodes were asleep / did not wake up.
    """
    return [
        "data_size_bytes",
        "technology",
        "radio_profile",
        "energy_profile",
        "node_count",
        "runs_found",
        "total_node_rows",
        "avg_radio_tx_mj",
        "avg_radio_rx_mj",
        "avg_csma_cca_mj",
        "avg_cad_scan_mj",
        "avg_mcu_mj",
        "avg_radio_plus_mcu_mj",
    ]


def build_group_summary_rows(groups: Dict[Tuple[str, str, str, str, int], GroupStats], expected_runs: int) -> List[Dict[str, object]]:
    """Build concise node-count summary rows.

    For each data size / technology / node-count group, each value is the
    node-weighted average over all nodes from all runs in the group:

        average = sum(per-node final energy) / total_node_rows

    Since runs in one group should all have the same node_count, this is
    normally equivalent to averaging each run's per-node average. If a run is
    incomplete or has a different row count, total_node_rows makes that visible.
    """
    rows: List[Dict[str, object]] = []
    for (size, tech, radio, energy_profile, node_count), gs in sorted(groups.items(), key=lambda x: (safe_int_from_context(x[0][0]), x[0][1], x[0][4])):
        denom = gs.node_rows if gs.node_rows > 0 else 1
        rows.append({
            "data_size_bytes": size,
            "technology": tech,
            "radio_profile": radio,
            "energy_profile": energy_profile,
            "node_count": node_count,
            "runs_found": gs.runs_found,
            "total_node_rows": gs.node_rows,
            # Radio-only TX energy from aggregate per-node energy_node_summary logic.
            "avg_radio_tx_mj": gs.node_sums.get("total_tx_mj", 0.0) / denom,
            # Radio-only RX/listening energy from aggregate per-node energy_node_summary logic.
            "avg_radio_rx_mj": gs.node_sums.get("total_rx_mj", 0.0) / denom,
            "avg_csma_cca_mj": gs.node_sums.get("csma_cca_mj", 0.0) / denom,
            "avg_cad_scan_mj": gs.node_sums.get("cad_scan_mj", 0.0) / denom,
            # MCU-only TX + RX/listening energy.
            "avg_mcu_mj": gs.node_sums.get("total_mcu_energy_mj", 0.0) / denom,
            # Final combined radio + MCU energy.
            "avg_radio_plus_mcu_mj": gs.node_sums.get("total_energy_with_mcu_mj", 0.0) / denom,
        })
    return rows


def audit_fieldnames() -> List[str]:
    return [
        "data_size_bytes", "technology", "radio_profile", "energy_profile", "node_count",
        "runs_found", "expected_runs", "runs_ok", "unique_reps", "unique_testbeds", "observed_testbeds", "observed_reps",
    ]


def build_audit_rows(groups: Dict[Tuple[str, str, str, str, int], GroupStats], expected_runs: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (size, tech, radio, energy_profile, node_count), gs in sorted(groups.items(), key=lambda x: (safe_int_from_context(x[0][0]), x[0][1], x[0][4])):
        rows.append({
            "data_size_bytes": size,
            "technology": tech,
            "radio_profile": radio,
            "energy_profile": energy_profile,
            "node_count": node_count,
            "runs_found": gs.runs_found,
            "expected_runs": expected_runs,
            "runs_ok": int(gs.runs_found == expected_runs) if expected_runs > 0 else "",
            "unique_reps": len({x for x in gs.reps if x != ""}),
            "unique_testbeds": len({x for x in gs.testbeds if x != ""}),
            "observed_testbeds": ";".join(sorted({x for x in gs.testbeds if x != ""})),
            "observed_reps": ";".join(sorted({x for x in gs.reps if x != ""}, key=lambda z: safe_int_from_context(z))),
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate memory-efficient energy CSV reports from simulator outputs.")
    ap.add_argument("--input", default=".", help="Single run directory or batch output root. Default: current directory.")
    ap.add_argument("--energy-profiles", default="energy_profiles.csv", help="CSV with radio_profile,tx_current_ma,rx_current_ma,voltage[,cca_current_ma,cad_current_ma].")
    ap.add_argument("--recursive", action="store_true", help="Search recursively for run directories containing metadata.csv and node_summary.csv.")
    ap.add_argument("--combined-out", default=None, help="Directory for batch-level reports. Default: <input>/energy_reports_combined.")
    ap.add_argument("--mcu-current-ma", type=float, default=0.0, help="MCU current during TX and RX-on/listening time. Default: 0.")
    ap.add_argument("--mcu-voltage", type=float, default=None, help="MCU voltage. Default: selected radio energy profile voltage.")
    ap.add_argument("--expected-runs-per-node-count", type=int, default=50, help="Expected run folders per data size / technology / node count. Default: 50.")
    ap.add_argument("--skip-per-run-reports", action="store_true", help="Do not write energy_*.csv inside each run folder. Saves disk I/O.")
    ap.add_argument("--write-combined-node-summary", action="store_true", help="Write one huge combined per-node file. Off by default to save RAM/disk.")
    ap.add_argument("--write-combined-day-summary", action="store_true", help="Write one huge combined per-day per-node file. Off by default to save RAM/disk.")
    args = ap.parse_args(argv)

    root = Path(args.input).resolve()
    profile_path = Path(args.energy_profiles).resolve()
    profiles = load_energy_profiles(profile_path)
    run_dirs = discover_run_dirs(root, args.recursive)
    if not run_dirs:
        raise SystemExit(f"No run directories found under {root}")

    combined = Path(args.combined_out).resolve() if args.combined_out else (root / "energy_reports_combined")
    combined.mkdir(parents=True, exist_ok=True)

    node_writer = StreamingCsvWriter(combined / "combined_energy_node_summary.csv", ordered_node_fields()) if args.write_combined_node_summary else None
    day_writer = StreamingCsvWriter(combined / "combined_energy_per_day_per_node.csv", ordered_day_fields()) if args.write_combined_day_summary else None
    compact_node_writer = StreamingCsvWriter(combined / "node_energy_summary.csv", compact_node_fieldnames())
    run_summary_writer = StreamingCsvWriter(combined / "run_energy_summary.csv", [
        "run_dir", "rep", "data_size_bytes", "technology", "testbed", "seed", "radio_profile", "energy_profile",
        "node_count", "days_reported", "voltage_v", "tx_current_ma", "rx_current_ma", "cca_current_ma", "cad_current_ma", "mcu_current_ma", "mcu_voltage_v",
        *[f"avg_per_node_{m}" for m in energy_metric_names()],
        *[f"max_node_{m}" for m in energy_metric_names()],
        *[f"sum_{m}" for m in energy_metric_names()],
    ])
    maxima_writer = StreamingCsvWriter(combined / "combined_energy_maxima.csv", [
        "run_dir", "rep", "data_size_bytes", "technology", "testbed", "seed", "radio_profile", "energy_profile",
        "scope", "direction", "stage", "metric", "max_value_mj", "nid", "day", "x", "y",
    ])

    groups: Dict[Tuple[str, str, str, str, int], GroupStats] = defaultdict(GroupStats)

    print(f"Found {len(run_dirs)} run director(y/ies).")
    try:
        for idx, rd in enumerate(run_dirs, start=1):
            run_summary, maxima_rows = process_run(
                rd,
                profiles,
                mcu_current_ma=float(args.mcu_current_ma),
                mcu_voltage_override=args.mcu_voltage,
                write_per_run=not args.skip_per_run_reports,
                combined_node_writer=node_writer,
                combined_day_writer=day_writer,
                compact_node_writer=compact_node_writer,
            )
            run_summary_writer.write(run_summary)
            for row in maxima_rows:
                maxima_writer.write(row)

            node_count = inum(run_summary.get("node_count"))
            key = (
                str(run_summary.get("data_size_bytes", "")),
                str(run_summary.get("technology", "")),
                str(run_summary.get("radio_profile", "")),
                str(run_summary.get("energy_profile", "")),
                node_count,
            )
            per_node_sum = {m: fnum(run_summary.get(f"sum_{m}")) for m in energy_metric_names()}
            maxima = {m: fnum(run_summary.get(f"max_node_{m}")) for m in energy_metric_names()}
            groups[key].add_run(run_summary, node_count, per_node_sum, maxima)

            if idx == 1 or idx % 100 == 0 or idx == len(run_dirs):
                print(f"Processed {idx}/{len(run_dirs)}: {rd}", flush=True)
    finally:
        if node_writer:
            node_writer.close()
        if day_writer:
            day_writer.close()
        compact_node_writer.close()
        run_summary_writer.close()
        maxima_writer.close()

    summary_rows = build_group_summary_rows(groups, args.expected_runs_per_node_count)
    audit_rows = build_audit_rows(groups, args.expected_runs_per_node_count)
    write_csv_rows(combined / "node_count_energy_summary.csv", summary_rows, summary_fieldnames())
    write_csv_rows(combined / "run_count_audit.csv", audit_rows, audit_fieldnames())

    print(f"Wrote batch-level reports in: {combined}")
    print("Main summary: node_count_energy_summary.csv")
    print("Compact per-node summary: node_energy_summary.csv")
    print("Run-count audit: run_count_audit.csv")
    if not args.write_combined_node_summary and not args.write_combined_day_summary:
        print("Skipped huge combined per-node/per-day files by default. Use --write-combined-node-summary or --write-combined-day-summary only if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
