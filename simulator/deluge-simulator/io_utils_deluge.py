# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
from typing import List, Optional

from models_deluge import Node, Packet


class OutputMixin:
    def write_metadata(self, testbed_path: str):
        def _arg(name: str, default=""):
            return getattr(self.args, name, default)
        def _join(name: str):
            return " ".join(_arg(name, []) or [])
        fields = [
            ("protocol", "deluge"),
            ("testbed", testbed_path),
            ("seed", _arg("seed")),
            ("radio_profile", _arg("radio_profile")),
            ("lora_radio", _arg("lora_radio")),
            ("sf_adv", _arg("sf_adv")), ("sf_data", _arg("sf_data")),
            ("bw_hz", _arg("bw_hz")), ("cr_denom", _arg("cr_denom")),
            ("tx_dbm", _arg("tx_dbm")),
            ("pl_ref", _arg("pl_ref")), ("pl_ref_m", _arg("pl_ref_m")),
            ("pl_ref_loss_db", _arg("pl_ref_loss_db")), ("pl_gamma", _arg("pl_gamma")),
            ("pl_40m_lpld0", _arg("pl_40m_lpld0")), ("pl_40m_gamma", _arg("pl_40m_gamma")),
            ("pl_sigma", _arg("pl_sigma")), ("fading", _arg("fading")),
            ("rayleigh_min_db", _arg("rayleigh_min_db")), ("rayleigh_max_db", _arg("rayleigh_max_db")),
            ("collision_th_db", _arg("collision_th_db")),
            ("adv_pl_bytes", _arg("adv_pl_bytes")), ("req_pl_bytes", _arg("req_pl_bytes", _arg("miss_pl_bytes"))),
            ("data_pl_bytes", _arg("data_pl_bytes")), ("data_header_bytes", _arg("data_header_bytes")),
            ("data_packet_bytes", _arg("data_packet_bytes")), ("data_session_id_bytes", _arg("data_session_id_bytes")),
            ("halow_mac_overhead_bytes", _arg("halow_mac_overhead_bytes")),
            ("radio_max_payload_bytes", _arg("radio_max_payload_bytes")),
            ("model_size_bytes", _arg("model_size_bytes")), ("complete_need", _arg("complete_need")),
            ("version_start", _arg("version_start")),
            ("deluge_page_packets", _arg("deluge_page_packets")),
            ("deluge_page_count", _arg("deluge_page_count")),
            ("deluge_adv_window_ms", _arg("deluge_adv_window_ms")),
            ("deluge_req_window_ms", _arg("deluge_req_window_ms")),
            ("deluge_data_gap_ms", _arg("deluge_data_gap_ms")),
            ("deluge_round_gap_ms", _arg("deluge_round_gap_ms")),
            ("deluge_sender_selection", _arg("deluge_sender_selection")),
            ("deluge_page_selection", _arg("deluge_page_selection")),
            ("deluge_packet_order", _arg("deluge_packet_order")),
            ("deluge_immediate_adv_on_page_complete", int(bool(_arg("deluge_immediate_adv_on_page_complete", True)))),
            ("area_m2", _arg("area_m2")), ("effective_area_m2", _arg("effective_area_m2")),
            ("comm_radius_m", _arg("comm_radius_m")), ("communication_radius_m", _arg("communication_radius_m")),
            ("expected_neighbors", _arg("expected_neighbors")), ("adv_probability", _arg("adv_probability")),
            ("auto_control_windows", int(bool(_arg("auto_control_windows", False)))),
            ("auto_window_rule", _arg("auto_window_rule")),
            ("reply_collision_rho", _arg("reply_collision_rho")),
            ("num_channels", _arg("num_channels")), ("chan_adv", _arg("chan_adv")),
            ("chan_upd", _arg("chan_upd")), ("chan_data", _arg("chan_data")),
            ("phy_repetitions", _arg("phy_repetitions")), ("phy_repeat_gap_ms", _arg("phy_repeat_gap_ms")),
            ("phy_repeat_jitter_ms", _arg("phy_repeat_jitter_ms")), ("phy_repeat_stages", _join("phy_repeat_stages")),
            ("phy_repeat_mode", _arg("phy_repeat_mode")), ("phy_repeat_channel_count", _arg("phy_repeat_channel_count")),
            ("phy_repeat_rx_scan", int(bool(_arg("phy_repeat_rx_scan", True)))),
            ("csma_ca", int(bool(_arg("csma_ca", False)))), ("csma_stages", _join("csma_stages")),
            ("csma_carrier_sense_thresh_dbm", _arg("csma_carrier_sense_thresh_dbm")),
            ("csma_busy_wait_airtime_factor", _arg("csma_busy_wait_airtime_factor")),
            ("csma_busy_wait_airtime_max_factor", _arg("csma_busy_wait_airtime_max_factor")),
            ("csma_busy_wait_jitter", _arg("csma_busy_wait_jitter")), ("csma_max_deferrals", _arg("csma_max_deferrals")),
            ("cad_csma", int(bool(_arg("cad_csma", False)))), ("cad_csma_stages", _join("cad_csma_stages")),
            ("cad_symbol_num", _arg("cad_symbol_num")), ("cad_detect_margin_db", _arg("cad_detect_margin_db")),
            ("cad_busy_wait_airtime_factor", _arg("cad_busy_wait_airtime_factor")),
            ("cad_busy_wait_airtime_max_factor", _arg("cad_busy_wait_airtime_max_factor")),
            ("cad_busy_wait_jitter", _arg("cad_busy_wait_jitter")), ("cad_max_deferrals", _arg("cad_max_deferrals")),
            ("duty_cycle_limit_s", _arg("duty_cycle_limit_s")),
            ("duty_cycle_data_fraction", _arg("duty_cycle_data_fraction")),
            ("duty_cycle_include_adv_in_data", int(bool(_arg("duty_cycle_include_adv_in_data", True)))),
            ("max_days", _arg("max_days")), ("max_cycles", _arg("max_cycles", _arg("max_days"))),
            ("until_complete", int(bool(_arg("until_complete", False)))),
            ("no_packet_log", int(bool(_arg("no_packet_log", False)))),
        ]
        path = os.path.join(self.out_dir, "metadata.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow([k for k,_ in fields]); w.writerow([v for _,v in fields])

    def write_node_summary(self):
        path = os.path.join(self.out_dir, "node_summary.csv")
        max_days = max((len(n.day_tx_adv) for n in self.nodes), default=0)
        base_cols = [
            "nid","x","y","is_gateway","updated","version","rx_ok","tx_ok","coll",
            "tx_adv","tx_req","tx_data",
            "tx_upd","tx_miss","tx_end","tx_repair",
            "adv_used_s","req_used_s","data_used_s",
            "upd_used_s","miss_used_s","repair_used_s",
            "rx_on_ms","rx_on_adv_ms","rx_on_req_ms","rx_on_data_ms",
            "rx_on_upd_ms","rx_on_miss_ms","rx_on_repair_ms",
            "rx_busy_ms","csma_cca_ms","cad_scan_ms","csma_sense_ms",
            "pages_complete_currentver","have_chunks_currentver",
        ]
        day_cols=[]
        for d in range(1,max_days+1):
            day_cols.extend([
                f"d{d}_tx_adv", f"d{d}_tx_req", f"d{d}_tx_data",
                f"d{d}_tx_upd", f"d{d}_tx_miss", f"d{d}_tx_end", f"d{d}_tx_repair",
                f"d{d}_rx_on_ms", f"d{d}_rx_on_adv_ms", f"d{d}_rx_on_req_ms", f"d{d}_rx_on_data_ms",
                f"d{d}_rx_on_upd_ms", f"d{d}_rx_on_miss_ms", f"d{d}_rx_on_repair_ms",
                f"d{d}_rx_busy_ms", f"d{d}_csma_cca_ms", f"d{d}_cad_scan_ms", f"d{d}_csma_sense_ms",
            ])
        with open(path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(base_cols+day_cols)
            for n in self.nodes:
                have_cur=len(n.have_chunks.get(n.version,set()))
                pages_cur=len(n.pages_have.get(n.version,set()))
                row=[n.nid,n.x,n.y,int(n.is_gateway),int(n.updated),n.version,n.rx_ok,n.tx_ok,n.coll,
                     n.tx_adv,n.tx_req,n.tx_data,n.tx_upd,n.tx_miss,n.tx_end,n.tx_repair,
                     round(n.adv_used_s,6),round(n.req_used_s,6),round(n.data_used_s,6),
                     round(n.upd_used_s,6),round(n.miss_used_s,6),round(n.repair_used_s,6),
                     round(n.rx_on_s*1000,3),round(n.rx_on_adv_s*1000,3),round(n.rx_on_req_s*1000,3),round(n.rx_on_data_s*1000,3),
                     round(n.rx_on_upd_s*1000,3),round(n.rx_on_miss_s*1000,3),round(n.rx_on_repair_s*1000,3),
                     round(n.rx_busy_s*1000,3),round(n.csma_cca_s*1000,3),round(n.cad_scan_s*1000,3),round(n.csma_sense_s*1000,3),
                     pages_cur,have_cur]
                def pad(lst, fill=0): return lst + [fill]*(max_days-len(lst))
                vals=[]
                d_adv=pad(n.day_tx_adv); d_req=pad(n.day_tx_req); d_data=pad(n.day_tx_data)
                d_upd=pad(n.day_tx_upd); d_miss=pad(n.day_tx_miss); d_end=pad(n.day_tx_end); d_rep=pad(n.day_tx_repair)
                d_rx=pad(n.day_rx_on_ms,0.0); d_ra=pad(n.day_rx_on_adv_ms,0.0); d_rr=pad(n.day_rx_on_req_ms,0.0); d_rd=pad(n.day_rx_on_data_ms,0.0)
                d_ru=pad(n.day_rx_on_upd_ms,0.0); d_rm=pad(n.day_rx_on_miss_ms,0.0); d_rp=pad(n.day_rx_on_repair_ms,0.0)
                d_busy=pad(n.day_rx_busy_ms,0.0); d_cca=pad(n.day_csma_cca_ms,0.0); d_cad=pad(n.day_cad_scan_ms,0.0); d_sense=pad(n.day_csma_sense_ms,0.0)
                for i in range(max_days):
                    vals.extend([d_adv[i],d_req[i],d_data[i],d_upd[i],d_miss[i],d_end[i],d_rep[i],d_rx[i],d_ra[i],d_rr[i],d_rd[i],d_ru[i],d_rm[i],d_rp[i],d_busy[i],d_cca[i],d_cad[i],d_sense[i]])
                w.writerow(row+vals)

    def write_node_state(self):
        path=os.path.join(self.out_dir,"node_state.csv")
        with open(path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["nid","updated","is_gateway","version","pages_complete_currentver","have_chunks_currentver"])
            for n in self.nodes:
                w.writerow([n.nid,int(n.updated),int(n.is_gateway),n.version,len(n.pages_have.get(n.version,set())),len(n.have_chunks.get(n.version,set()))])

    def write_packet_log_row(self, p: Packet, rx_node: Optional[int], is_duplicate: bool=False):
        if self.args.no_packet_log: return
        with open(self.packet_log_path,"a",newline="") as f:
            w=csv.writer(f)
            w.writerow([self.day_idx+1,p.ptype,p.src,"" if p.dst is None else p.dst,p.sf,p.bw_hz,p.cr_denom,
                        round(p.start_s,6),round(p.airtime_s,6),p.power_dbm,p.chan,p.version,
                        -1 if p.chunk_idx is None else p.chunk_idx,-1 if rx_node is None else rx_node,
                        "" if p.logical_id is None else p.logical_id,p.rep_idx,p.rep_count,"" if p.base_chan is None else p.base_chan,
                        "" if p.phase is None else p.phase,p.end_seq,int(bool(is_duplicate)),getattr(p,"csma_deferrals",0),round(float(getattr(p,"csma_delay_s",0.0) or 0.0),6),
                        "" if getattr(p,"page_idx",None) is None else p.page_idx,
                        "" if getattr(p,"page_packet_idx",None) is None else p.page_packet_idx,
                        "" if getattr(p,"requested_page",None) is None else p.requested_page,
                        0 if getattr(p,"requested_indices",None) is None else len(p.requested_indices),getattr(p,"session_id",0)])


def load_testbed(path: str) -> List[Node]:
    nodes=[]
    with open(path,"r",newline="") as f:
        r=csv.DictReader(f); cols=set(r.fieldnames or [])
        for row in r:
            nid=int(row.get("id", row.get("nid")))
            x=float(row["x"]); y=float(row["y"])
            val=row.get("is_gateway", row.get("gateway", "0"))
            is_gw=bool(int(str(val).strip() or "0"))
            nodes.append(Node(nid=nid,x=x,y=y,is_gateway=is_gw))
    return nodes
