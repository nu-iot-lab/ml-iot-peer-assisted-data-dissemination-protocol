# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
from typing import List, Optional

from models import Node, Packet


class OutputMixin:
    def write_metadata(self, testbed_path: str):
        """Write one-row simulation metadata for reproducibility.

        Keep this list intentionally explicit instead of dumping vars(args):
        it preserves a stable column order and records the fields that affect
        radio/profile choice, protocol timing, payload sizes, channels, and run
        controls. Missing attributes are written as an empty string so older
        argument sets remain compatible.
        """
        def _arg(name: str, default=""):
            return getattr(self.args, name, default)

        def _pair(name: str, idx: int, default=""):
            value = getattr(self.args, name, None)
            if value is None:
                return default
            try:
                return value[idx]
            except (TypeError, IndexError):
                return default

        fields = [
            # Inputs / selected radio profile
            ("testbed", testbed_path),
            ("seed", _arg("seed")),
            ("radio_profile", _arg("radio_profile")),
            # Kept for backward compatibility with old result files/args
            ("lora_radio", _arg("lora_radio")),

            # PHY / radio settings
            ("sf_adv", _arg("sf_adv")),
            ("sf_data", _arg("sf_data")),
            ("bw_hz", _arg("bw_hz")),
            ("cr_denom", _arg("cr_denom")),
            ("tx_dbm", _arg("tx_dbm")),
            ("pl_ref", _arg("pl_ref")),
            ("pl_ref_m", _arg("pl_ref_m")),
            ("pl_ref_loss_db", _arg("pl_ref_loss_db")),
            ("pl_gamma", _arg("pl_gamma")),
            # Legacy column names kept for backward compatibility. They now mirror
            # the effective generic PL(d0)/gamma even when d0 is 100m.
            ("pl_40m_lpld0", _arg("pl_40m_lpld0")),
            ("pl_40m_gamma", _arg("pl_40m_gamma")),
            ("pl_sigma", _arg("pl_sigma")),
            ("fading", _arg("fading")),
            ("rayleigh_min_db", _arg("rayleigh_min_db")),
            ("rayleigh_max_db", _arg("rayleigh_max_db")),
            ("collision_th_db", _arg("collision_th_db")),

            # Payload / model sizing
            ("adv_pl_bytes", _arg("adv_pl_bytes")),
            ("upd_pl_bytes", _arg("upd_pl_bytes")),
            ("data_pl_bytes", _arg("data_pl_bytes")),
            ("data_header_bytes", _arg("data_header_bytes")),
            ("data_session_id_bytes", _arg("data_session_id_bytes")),
            ("halow_mac_overhead_bytes", _arg("halow_mac_overhead_bytes")),
            ("data_packet_bytes", _arg("data_packet_bytes")),
            ("radio_max_payload_bytes", _arg("radio_max_payload_bytes")),
            ("model_size_bytes", _arg("model_size_bytes")),
            ("miss_pl_bytes", _arg("miss_pl_bytes")),
            ("miss_len_mode", _arg("miss_len_mode")),
            ("miss_encoding", _arg("miss_encoding")),
            ("miss_hdr_bytes", _arg("miss_hdr_bytes")),
            ("miss_header_bytes", _arg("miss_header_bytes")),
            ("miss_max_packet_bytes", _arg("miss_max_packet_bytes")),
            ("miss_bitmap_capacity_bytes", _arg("miss_bitmap_capacity_bytes")),
            ("miss_bitmap_capacity_chunks", _arg("miss_bitmap_capacity_chunks")),
            ("data_chunks_per_round", _arg("data_chunks_per_round")),
            ("complete_need", _arg("complete_need")),
            ("version_start", _arg("version_start")),

            # Phase budgets / timing
            ("adv_budget_s", _arg("adv_budget_s")),
            ("data_budget_s", _arg("data_budget_s")),
            ("miss_budget_s", _arg("miss_budget_s")),
            ("repair_budget_s", _arg("repair_budget_s")),
            ("auto_data_repair_windows", int(bool(_arg("auto_data_repair_windows", False)))),
            ("dynamic_data_repair", int(bool(_arg("dynamic_data_repair", True)))),
            ("data_inactivity_airtimes", _arg("data_inactivity_airtimes")),
            ("repair_inactivity_airtimes", _arg("repair_inactivity_airtimes")),
            ("owner_miss_extra_airtimes", _arg("owner_miss_extra_airtimes")),
            ("repair_chunks_per_round", _arg("repair_chunks_per_round")),
            ("data_repair_window_margin", _arg("data_repair_window_margin")),
            ("auto_data_repair_data_airtime_s", _arg("auto_data_repair_data_airtime_s")),
            ("auto_data_repair_total_packets", _arg("auto_data_repair_total_packets")),
            ("auto_data_repair_tx_train_s", _arg("auto_data_repair_tx_train_s")),
            ("auto_data_repair_data_backoff_max_s", _arg("auto_data_repair_data_backoff_max_s")),
            ("auto_data_repair_computed_budget_s", _arg("auto_data_repair_computed_budget_s")),
            ("phase_gap_ms", _arg("phase_gap_ms")),
            ("phase_gap_s", _arg("phase_gap_s")),
            ("upd_listen_ms", _arg("upd_listen_ms")),
            ("upd_listen_s", _arg("upd_listen_s")),
            ("miss_listen_ms", _arg("miss_listen_ms")),
            ("miss_listen_s", _arg("miss_listen_s")),
            ("proc_gap_ms", _arg("proc_gap_ms")),
            ("proc_gap_s", _arg("proc_gap_s")),
            ("tx_to_rx_s", _arg("tx_to_rx_s")),
            ("rx_to_tx_s", _arg("rx_to_tx_s")),

            # Density / mathematical control-window model
            ("area_m2", _arg("area_m2")),
            ("effective_area_m2", _arg("effective_area_m2")),
            ("comm_radius_m", _arg("comm_radius_m")),
            ("communication_radius_m", _arg("communication_radius_m")),
            ("expected_neighbors", _arg("expected_neighbors")),
            ("adv_probability", _arg("adv_probability")),
            ("auto_control_windows", int(bool(_arg("auto_control_windows", False)))),
            ("auto_window_rule", _arg("auto_window_rule")),
            ("control_window_node_factor", _arg("control_window_node_factor")),
            ("control_window_node_offset", _arg("control_window_node_offset")),
            ("cad_control_window_node_factor", _arg("cad_control_window_node_factor")),
            ("cad_control_window_node_offset", _arg("cad_control_window_node_offset")),
            ("effective_control_window_nodes", _arg("effective_control_window_nodes")),
            ("reply_collision_rho", _arg("reply_collision_rho")),
            ("min_control_window_s", _arg("min_control_window_s")),
            ("max_control_window_s", _arg("max_control_window_s")),
            ("auto_adv_packet_airtime_s", _arg("auto_adv_packet_airtime_s")),
            ("auto_upd_packet_airtime_s", _arg("auto_upd_packet_airtime_s")),
            ("auto_miss_packet_airtime_s", _arg("auto_miss_packet_airtime_s")),
            ("auto_miss_packet_bytes", _arg("auto_miss_packet_bytes")),
            ("auto_adv_window_s", _arg("auto_adv_window_s")),
            ("auto_upd_window_s", _arg("auto_upd_window_s")),
            ("auto_miss_raw_window_s", _arg("auto_miss_raw_window_s")),
            ("auto_miss_raw_select_s", _arg("auto_miss_raw_select_s")),
            ("auto_miss_data_channel_count", _arg("auto_miss_data_channel_count")),
            ("auto_miss_window_scale_factor", _arg("auto_miss_window_scale_factor")),
            ("auto_miss_scaled_select_s", _arg("auto_miss_scaled_select_s")),
            ("auto_miss_window_s", _arg("auto_miss_window_s")),

            # Backoff ranges
            ("adv_backoff_ms_min", _pair("adv_backoff_ms", 0)),
            ("adv_backoff_ms_max", _pair("adv_backoff_ms", 1)),
            ("upd_backoff_ms_min", _pair("upd_backoff_ms", 0)),
            ("upd_backoff_ms_max", _pair("upd_backoff_ms", 1)),
            ("data_backoff_ms_min", _pair("data_backoff_ms", 0)),
            ("data_backoff_ms_max", _pair("data_backoff_ms", 1)),
            ("miss_backoff_ms_min", _pair("miss_backoff_ms", 0)),
            ("miss_backoff_ms_max", _pair("miss_backoff_ms", 1)),
            ("repair_backoff_ms_min", _pair("repair_backoff_ms", 0)),
            ("repair_backoff_ms_max", _pair("repair_backoff_ms", 1)),

            # Channel model
            ("num_channels", _arg("num_channels")),
            ("chan_adv", _arg("chan_adv")),
            ("chan_upd", _arg("chan_upd")),
            ("chan_data", _arg("chan_data")),

            # Physical repetition / BLE Mesh-like advertising retransmission layer
            ("phy_repetitions", _arg("phy_repetitions")),
            ("phy_repeat_gap_ms", _arg("phy_repeat_gap_ms")),
            ("phy_repeat_jitter_ms", _arg("phy_repeat_jitter_ms")),
            ("phy_repeat_stages", " ".join(_arg("phy_repeat_stages", []) or [])),
            ("phy_repeat_mode", _arg("phy_repeat_mode")),
            ("phy_repeat_channel_count", _arg("phy_repeat_channel_count")),
            ("phy_repeat_rx_scan", int(bool(_arg("phy_repeat_rx_scan", True)))),

            # CSMA/CA-like channel access layer
            ("csma_ca", int(bool(_arg("csma_ca", False)))),
            ("csma_stages", " ".join(_arg("csma_stages", []) or [])),
            ("csma_carrier_sense_thresh_dbm", _arg("csma_carrier_sense_thresh_dbm")),
            ("csma_busy_wait_airtime_factor", _arg("csma_busy_wait_airtime_factor")),
            ("csma_busy_wait_airtime_max_factor", _arg("csma_busy_wait_airtime_max_factor")),
            ("csma_busy_wait_jitter", _arg("csma_busy_wait_jitter")),
            ("csma_max_deferrals", _arg("csma_max_deferrals")),

            # LoRa CAD-CSMA channel access layer
            ("cad_csma", int(bool(_arg("cad_csma", False)))),
            ("cad_csma_stages", " ".join(_arg("cad_csma_stages", []) or [])),
            ("cad_symbol_num", _arg("cad_symbol_num")),
            ("cad_detect_margin_db", _arg("cad_detect_margin_db")),
            ("cad_busy_wait_airtime_factor", _arg("cad_busy_wait_airtime_factor")),
            ("cad_busy_wait_airtime_max_factor", _arg("cad_busy_wait_airtime_max_factor")),
            ("cad_busy_wait_jitter", _arg("cad_busy_wait_jitter")),
            ("cad_max_deferrals", _arg("cad_max_deferrals")),

            # Duty-cycle limiting
            ("duty_cycle_limit_s", _arg("duty_cycle_limit_s")),
            ("duty_cycle_data_fraction", _arg("duty_cycle_data_fraction")),
            ("duty_cycle_include_adv_in_data", int(bool(_arg("duty_cycle_include_adv_in_data", True)))),

            # Run controls / output options
            ("max_days", _arg("max_days")),
            ("until_complete", int(bool(_arg("until_complete", False)))),
            ("no_packet_log", int(bool(_arg("no_packet_log", False)))),
            ("animate", int(bool(_arg("animate", False)))),
            ("anim_fps", _arg("anim_fps")),
        ]

        path = os.path.join(self.out_dir, "metadata.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([name for name, _value in fields])
            w.writerow([value for _name, value in fields])

    def write_node_summary(self):
        path = os.path.join(self.out_dir, "node_summary.csv")
        # Determine how many days of per-day deltas we have
        max_days = 0
        for n in self.nodes:
            max_days = max(max_days, len(n.day_tx_adv))
        base_cols = [
            "nid","x","y","is_gateway","updated","version",
            "rx_ok","tx_ok","coll",
            "tx_adv","tx_upd","tx_data","tx_miss","tx_end","tx_repair",
            "adv_used_s","upd_used_s","data_used_s","miss_used_s","repair_used_s",
            "rx_on_ms",
            "rx_on_adv_ms","rx_on_upd_ms","rx_on_data_ms","rx_on_miss_ms","rx_on_repair_ms",
            "rx_busy_ms","csma_cca_ms","cad_scan_ms","csma_sense_ms","have_chunks_currentver",
        ]
        day_cols = []
        for d in range(1, max_days + 1):
            day_cols.extend([
                f"d{d}_tx_adv",
                f"d{d}_tx_upd",
                f"d{d}_tx_data",
                f"d{d}_tx_miss",
                f"d{d}_tx_end",
                f"d{d}_tx_repair",
                f"d{d}_rx_on_ms",
                f"d{d}_rx_on_adv_ms",
                f"d{d}_rx_on_upd_ms",
                f"d{d}_rx_on_data_ms",
                f"d{d}_rx_on_miss_ms",
                f"d{d}_rx_on_repair_ms",
                f"d{d}_rx_busy_ms",
                f"d{d}_csma_cca_ms",
                f"d{d}_cad_scan_ms",
                f"d{d}_csma_sense_ms",
            ])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(base_cols + day_cols)

        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            for n in self.nodes:
                have_cur = len(n.have_chunks.get(n.version, set()))
                row = [
                    n.nid,
                    n.x,
                    n.y,
                    int(n.is_gateway),
                    int(n.updated),
                    n.version,
                    n.rx_ok,
                    n.tx_ok,
                    n.coll,
                    n.tx_adv,
                    n.tx_upd,
                    n.tx_data,
                    n.tx_miss,
                    n.tx_end,
                    n.tx_repair,
                    round(n.adv_used_s, 3),
                    round(n.upd_used_s, 3),
                    round(n.data_used_s, 3),
                    round(n.miss_used_s, 3),
                    round(n.repair_used_s, 3),
                    round(n.rx_on_s * 1000.0, 3),
                    round(n.rx_on_adv_s * 1000.0, 3),
                    round(n.rx_on_upd_s * 1000.0, 3),
                    round(n.rx_on_data_s * 1000.0, 3),
                    round(n.rx_on_miss_s * 1000.0, 3),
                    round(n.rx_on_repair_s * 1000.0, 3),
                    round(n.rx_busy_s * 1000.0, 3),
                    round(n.csma_cca_s * 1000.0, 3),
                    round(n.cad_scan_s * 1000.0, 3),
                    round(n.csma_sense_s * 1000.0, 3),
                    have_cur,
                ]

                # Per-day deltas (pad to max_days)
                def _pad(lst, fill=0):
                    return lst + [fill] * (max_days - len(lst))

                d_adv = _pad(n.day_tx_adv, 0)
                d_upd = _pad(n.day_tx_upd, 0)
                d_data = _pad(n.day_tx_data, 0)
                d_miss = _pad(n.day_tx_miss, 0)
                d_end = _pad(n.day_tx_end, 0)
                d_rep = _pad(n.day_tx_repair, 0)
                d_rx_on = _pad(n.day_rx_on_ms, 0.0)
                d_rx_on_adv = _pad(n.day_rx_on_adv_ms, 0.0)
                d_rx_on_upd = _pad(n.day_rx_on_upd_ms, 0.0)
                d_rx_on_data = _pad(n.day_rx_on_data_ms, 0.0)
                d_rx_on_miss = _pad(n.day_rx_on_miss_ms, 0.0)
                d_rx_on_repair = _pad(n.day_rx_on_repair_ms, 0.0)
                d_rx_busy = _pad(n.day_rx_busy_ms, 0.0)
                d_csma_cca = _pad(n.day_csma_cca_ms, 0.0)
                d_cad_scan = _pad(n.day_cad_scan_ms, 0.0)
                d_csma_sense = _pad(n.day_csma_sense_ms, 0.0)

                for i in range(max_days):
                    row.extend([
                        d_adv[i], d_upd[i], d_data[i], d_miss[i], d_end[i], d_rep[i],
                        round(float(d_rx_on[i]), 6),
                        round(float(d_rx_on_adv[i]), 6),
                        round(float(d_rx_on_upd[i]), 6),
                        round(float(d_rx_on_data[i]), 6),
                        round(float(d_rx_on_miss[i]), 6),
                        round(float(d_rx_on_repair[i]), 6),
                        round(float(d_rx_busy[i]), 6),
                        round(float(d_csma_cca[i]), 6),
                        round(float(d_cad_scan[i]), 6),
                        round(float(d_csma_sense[i]), 6),
                    ])

                w.writerow(row)


    def write_node_state(self):
        path = os.path.join(self.out_dir, "node_state.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "nid",
                    "updated",
                    "is_gateway",
                    "version",
                    "have_chunks_currentver",
                ]
            )
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            for n in self.nodes:
                w.writerow(
                    [
                        n.nid,
                        int(n.updated),
                        int(n.is_gateway),
                        n.version,
                        len(n.have_chunks.get(n.version, set())),
                    ]
                )

    def write_packet_log_row(self, p: Packet, rx_node: Optional[int], is_duplicate: bool = False):
        if self.args.no_packet_log:
            return
        with open(self.packet_log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    self.day_idx + 1,
                    p.ptype,
                    p.src,
                    p.dst if p.dst is not None else "",
                    p.sf,
                    p.bw_hz,
                    p.cr_denom,
                    round(p.start_s, 6),
                    round(p.airtime_s, 6),
                    p.power_dbm,
                    p.chan,
                    p.version,
                    -1 if p.chunk_idx is None else p.chunk_idx,
                    -1 if rx_node is None else rx_node,
                    "" if p.logical_id is None else p.logical_id,
                    p.rep_idx,
                    p.rep_count,
                    "" if p.base_chan is None else p.base_chan,
                    "" if p.phase is None else p.phase,
                    p.end_seq,
                    int(bool(is_duplicate)),
                    getattr(p, "csma_deferrals", 0),
                    round(float(getattr(p, "csma_delay_s", 0.0) or 0.0), 6),
                    getattr(p, "session_id", 0),
                ]
            )



def load_testbed(path: str) -> List[Node]:
    nodes: List[Node] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        for row in r:
            nid = int(row["id"])
            x = float(row["x"])
            y = float(row["y"])
            is_gw = False
            if "is_gateway" in cols:
                val = row.get("is_gateway", "0").strip()
                is_gw = bool(int(val)) if val else False
            nodes.append(Node(nid=nid, x=x, y=y, is_gateway=is_gw))
    return nodes


