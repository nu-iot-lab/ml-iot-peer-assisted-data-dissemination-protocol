#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import random
from typing import Dict, List, Optional, Set, Tuple, DefaultDict
from collections import defaultdict

import simpy

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


from radio_profiles import RadioProfile, get_radio_profile
from models import Node, Packet, _splitmix64
from config import parse_args
from io_utils import OutputMixin, load_testbed
from channel import ChannelMixin

# Current selected radio profile, initialized in main().
RADIO_PROFILE: Optional[RadioProfile] = None

def _radio_profile() -> RadioProfile:
    if RADIO_PROFILE is None:
        raise RuntimeError("RADIO_PROFILE is not initialized. Call main()/parse args first.")
    return RADIO_PROFILE


def _miss_payload_length(
    missing_indices: List[int],
    chunks: int,
    method: str,
    hdr_bytes: int,
) -> int:
    """Compute MISS payload length."""
    if method == "list":
        return hdr_bytes + len(missing_indices)
    elif method == "bitmap":
        return hdr_bytes + math.ceil(chunks / 8.0)
    else:
        raise ValueError(f"Unknown miss_encoding method={method!r}")


class Sim(ChannelMixin, OutputMixin):
    def __init__(self, args, nodes: List[Node], rng: random.Random):
        self.args = args
        self.nodes = nodes
        self.rng = rng
        self.radio = _radio_profile()
        # Per-node RNG streams (independent across devices, reproducible from --seed)
        # Use these for node-local randomized decisions like backoff selection.
        self.node_rng: Dict[int, random.Random] = {}
        base_seed = int(getattr(args, "seed", 0) or 0)
        for n in self.nodes:
            # Mix global seed with node id to derive a stable per-node seed
            mixed = _splitmix64((base_seed ^ (n.nid * 0xD1B54A32D192ED03)) & 0xFFFFFFFFFFFFFFFF)
            # random.Random expects a Python int; keep full 64-bit value
            self.node_rng[n.nid] = random.Random(mixed)
        self.day_idx = 0
        self.out_dir = args.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.env: Optional[simpy.Environment] = None
        self.active_packets: List[Packet] = []
        # History of all transmissions that have started (used for correct overlap/collision checks)
        self.tx_history: List[Packet] = []
        self.by_id: Dict[int, Node] = {n.nid: n for n in self.nodes}
        self.reg_followers_day: DefaultDict[int, Set[int]] = defaultdict(set)
        # For each owner id -> set of (version, chunk_idx) tuples to retransmit in REPAIR
        self.accepted_repair_day: DefaultDict[int, Set[Tuple[int, int]]] = defaultdict(set)
        self.sent_upd_today: Set[Tuple[int, int]] = set()
        self.next_logical_packet_id = 0

        # Packet log
        self.packet_log_path = os.path.join(self.out_dir, "packet_log.csv")
        if not args.no_packet_log:
            with open(self.packet_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "day",
                        "ptype",
                        "src",
                        "dst",
                        "sf",
                        "bw_hz",
                        "cr_denom",
                        "start_s",
                        "airtime_s",
                        "tx_dbm",
                        "chan",
                        "version",
                        "chunk_idx",
                        "rx_node",
                        "logical_id",
                        "rep_idx",
                        "rep_count",
                        "base_chan",
                        "phase",
                        "end_seq",
                        "is_duplicate",
                        "csma_deferrals",
                        "csma_delay_s",
                        "session_id",
                    ]
                )

        # Initialize source gateways: if any are provided in the testbed, use them;
        # otherwise pick one random node. A source gateway is an external/full source,
        # so it must start with the complete object in its chunk buffer.
        any_gw = any(n.is_gateway for n in self.nodes)
        if any_gw:
            for n in self.nodes:
                if n.is_gateway:
                    self._grant_full_update(n, args.version_start)
        else:
            gw = self.rng.choice(self.nodes)
            self._grant_full_update(gw, args.version_start)


    def _is_complete_holder(self, n: Node, version: Optional[int] = None) -> bool:
        """True only when a node actually has all chunks for a version."""
        v = n.version if version is None else version
        return (v > 0) and (len(n.have_chunks.get(v, set())) >= int(self.args.complete_need))

    def _grant_full_update(self, n: Node, version: int) -> None:
        """Mark a node as a full source/updated holder for a version."""
        n.is_gateway = True
        n.updated = True
        n.version = version
        n.have_chunks[version] = set(range(int(self.args.complete_need)))


    def testbed_area_m2(self) -> float:
        """Return deployment area used by density/auto-window formulas."""
        if getattr(self.args, "area_m2", None) is not None:
            return max(float(self.args.area_m2), 1.0)
        if not self.nodes:
            return 1.0
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1.0)

    def communication_radius_m(self) -> float:
        """Return communication radius R used by density/auto-window formulas."""
        return self.radio.communication_radius_m(self.args, self.args.sf_adv)

    def effective_control_window_node_count(self) -> float:
        """Physical device count; no hidden-terminal half-N shortcut.

        CAD/CSMA use a separate reply-airtime window-sizing approximation,
        not an artificially smaller neighborhood. This count also controls
        sender-activation probability through expected_neighbors().
        """
        return float(len(self.nodes))

    def uses_carrier_sensing(self) -> bool:
        return bool(getattr(self.args, "cad_csma", False) or getattr(self.args, "csma_ca", False))

    def expected_neighbors(self) -> float:
        """Expected number of neighbors inside the communication disk."""
        n_eff = self.effective_control_window_node_count()
        if n_eff <= 1.0:
            return 0.0
        area = self.testbed_area_m2()
        radius = self.communication_radius_m()
        return (n_eff - 1.0) * ((math.pi * radius * radius) / area)

    def density_adv_probability(self) -> float:
        """Compute ADV probability from channel count, density and radius.

        p = min(1, (c - 1) / (((N - 1) * ((pi * R^2) / A) * 0.5)))
        """
        expected = self.expected_neighbors()
        if expected <= 0.0:
            return 1.0
        c = int(getattr(self.args, "num_channels", 1) or 1)
        data_channel_count = max(1, c - 1)
        denom = expected * 0.5
        if denom <= 0.0:
            return 1.0
        return min(1.0, data_channel_count / denom)

    def _miss_len_for_control_window(self) -> int:
        """Estimate MISS packet size for control-window calculation.

        For bitmap mode this is the actual full-object bitmap length.
        For list mode this is a worst-case full missing-list length.
        Fixed mode uses --miss-pl-bytes.
        """
        args = self.args
        if getattr(args, "miss_len_mode", "auto") == "fixed":
            return int(args.miss_pl_bytes)

        chunks = max(1, int(getattr(args, "complete_need", 1)))
        hdr = int(getattr(args, "miss_hdr_bytes", getattr(args, "miss_header_bytes", 6)))
        if getattr(args, "miss_encoding", "list") == "bitmap":
            length = hdr + int(math.ceil(chunks / 8.0))
        else:
            length = hdr + chunks

        # We assume the selected MISS packet format/capacity is sufficient for the
        # configured model size. Do not cap the timing length here: if the bitmap/list
        # is larger, the airtime/window calculation should reflect the actual MISS
        # description size rather than silently shortening it.
        return int(length)

    def control_window_s(self, packet_airtime_s: float) -> float:
        """Size control window using MAC-specific approximation.

        Raw/ALOHA: collision-target approximation with physical N, p_R and rho.
        CAD/CSMA: 4.5 * expected_local_replies * packet airtime.
        The 4.5 factor is a planning heuristic (transmission + three backoff
        stages); actual CAD/CSMA channel sensing/backoff is modeled separately.
        Packet airtime refers to the specific ADV/UPD/MISS packet being sized.
        """
        args = self.args
        n = float(len(self.nodes))
        t_pkt = max(float(packet_airtime_s), 1e-12)
        area = self.testbed_area_m2()
        radius = self.communication_radius_m()
        p_reach = min(1.0, math.pi * radius * radius / area)

        if self.uses_carrier_sensing():
            expected_replies = max(0.0, n - 1.0) * p_reach
            raw = 4.5 * expected_replies * t_pkt
        elif n <= 2.0:
            raw = t_pkt
        else:
            rho = min(max(float(getattr(args, "reply_collision_rho", 0.05)), 1e-12), 1.0 - 1e-12)
            denom_term = 1.0 - ((1.0 - rho) ** (1.0 / (n - 2.0)))
            raw = t_pkt + 2.0 * t_pkt * p_reach / denom_term if denom_term > 0.0 else t_pkt

        min_w = float(getattr(args, "min_control_window_s", 0.0) or 0.0)
        max_w = float(getattr(args, "max_control_window_s", 0.0) or 0.0)
        if min_w > 0.0:
            raw = max(raw, min_w)
        if max_w > 0.0:
            raw = min(raw, max_w)
        return max(raw, t_pkt)

    def apply_auto_control_windows(self) -> None:
        """Override ADV/UPD/MISS timing/backoffs using the mathematical model.

        The maximum random backoff is t_window - t_pkt. For UPD/MISS, rx_to_tx_s
        is also included in the listening/budget window because packets are sent
        after RX-to-TX turnaround.
        """
        args = self.args

        # Always expose these in metadata.
        args.effective_area_m2 = self.testbed_area_m2()
        args.communication_radius_m = self.communication_radius_m()
        args.effective_control_window_nodes = self.effective_control_window_node_count()
        args.auto_window_rule = "4.5_expected_replies_airtime" if self.uses_carrier_sensing() else "aloha_collision_target"
        args.expected_neighbors = self.expected_neighbors()
        args.adv_probability = self.density_adv_probability()

        if not getattr(args, "auto_control_windows", False):
            return

        adv_pkt_s = self.radio.airtime_seconds_for_args(
            args, args.sf_adv, args.bw_hz, args.cr_denom, args.adv_pl_bytes
        )
        upd_pkt_s = self.radio.airtime_seconds_for_args(
            args, args.sf_adv, args.bw_hz, args.cr_denom, args.upd_pl_bytes
        )
        miss_len = self._miss_len_for_control_window()
        miss_pkt_s = self.radio.airtime_seconds_for_args(
            args, args.sf_adv, args.bw_hz, args.cr_denom, miss_len
        )

        adv_logic_s = self.logical_airtime_s("ADV", adv_pkt_s)
        upd_logic_s = self.logical_airtime_s("UPD", upd_pkt_s)
        miss_logic_s = self.logical_airtime_s("MISS", miss_pkt_s)

        adv_window_s = self.control_window_s(adv_logic_s)
        upd_window_s = self.control_window_s(upd_logic_s)

        # First compute the ordinary MISS window from the same collision model.
        # Then scale only the *selectable start interval* by the number of DATA
        # channels, because MISS packets are distributed over DATA channels rather
        # than all contending on one control channel.
        #
        # raw_miss_window_s = raw selectable interval + miss_pkt_s
        # scaled_select_s    = (raw_miss_window_s - miss_pkt_s) / (c - 1)
        # final_miss_window  = scaled_select_s + miss_pkt_s
        raw_miss_window_s = self.control_window_s(miss_logic_s)
        data_channel_count = max(1, int(getattr(args, "num_channels", 1) or 1) - 1)
        raw_miss_select_s = max(0.0, raw_miss_window_s - miss_logic_s)
        miss_scale = max(0.0, float(getattr(args, "miss_window_scale_factor", 1.0) or 1.0))
        scaled_miss_select_s = (raw_miss_select_s / float(data_channel_count)) * miss_scale
        miss_window_s = scaled_miss_select_s + miss_logic_s

        args.auto_adv_packet_airtime_s = adv_logic_s
        args.auto_upd_packet_airtime_s = upd_logic_s
        args.auto_miss_packet_airtime_s = miss_logic_s
        args.auto_adv_single_airtime_s = adv_pkt_s
        args.auto_upd_single_airtime_s = upd_pkt_s
        args.auto_miss_single_airtime_s = miss_pkt_s
        args.auto_miss_packet_bytes = miss_len
        args.auto_adv_window_s = adv_window_s
        args.auto_upd_window_s = upd_window_s
        # Raw MISS window from the formula before channel scaling.
        args.auto_miss_raw_window_s = raw_miss_window_s
        args.auto_miss_raw_select_s = raw_miss_select_s
        # Number of data channels used for MISS scaling.
        args.auto_miss_data_channel_count = data_channel_count
        args.auto_miss_window_scale_factor = miss_scale
        # Scaled selectable interval and final MISS window actually used.
        args.auto_miss_scaled_select_s = scaled_miss_select_s
        args.auto_miss_window_s = miss_window_s

        args.adv_budget_s = adv_window_s
        args.adv_backoff_ms = [0.0, max(0.0, adv_window_s - adv_logic_s) * 1000.0]

        # UPD starts after rx_to_tx_s, so keep owners listening for rx_to_tx_s + modeled window.
        args.upd_listen_s = args.rx_to_tx_s + upd_window_s
        args.upd_listen_ms = args.upd_listen_s * 1000.0
        args.upd_backoff_ms = [0.0, max(0.0, upd_window_s - upd_logic_s) * 1000.0]

        # MISS also starts after rx_to_tx_s. Its backoff max is the scaled
        # selectable interval, so packet start is in [0, miss_window_s - t_pkt].
        args.miss_budget_s = args.rx_to_tx_s + miss_window_s
        args.miss_listen_s = args.miss_budget_s
        args.miss_listen_ms = args.miss_listen_s * 1000.0
        args.miss_backoff_ms = [0.0, scaled_miss_select_s * 1000.0]

    # ---------- Dynamic owner-local DATA/MISS/REPAIR helpers ----------
    def _data_packet_airtime_s(self) -> float:
        return self.radio.airtime_seconds_for_args(
            self.args,
            self.args.sf_data,
            self.args.bw_hz,
            self.args.cr_denom,
            self.args.data_packet_bytes,
        )

    def _end_wait_s(self, p: Packet) -> float:
        """END1 waits two END logical airtimes, END2 waits one, END3 exits immediately."""
        seq = int(getattr(p, "end_seq", 0) or 0)
        end_logic = self.logical_airtime_s("END", p.airtime_s)
        if seq <= 1:
            return 2.0 * end_logic
        if seq == 2:
            return 1.0 * end_logic
        return 0.0

    def _data_inactivity_timeout_s(self) -> float:
        return max(0.0, float(getattr(self.args, "data_inactivity_airtimes", 10.0) or 10.0)) * self.logical_airtime_s("DATA", self._data_packet_airtime_s())

    def _repair_inactivity_timeout_s(self) -> float:
        return max(0.0, float(getattr(self.args, "repair_inactivity_airtimes", 10.0) or 10.0)) * self.logical_airtime_s("REPAIR", self._data_packet_airtime_s())

    def _schedule_data_inactivity_timeout(self, n: Node, version: int, owner: int, from_t: float) -> None:
        if self.env is None:
            return
        n.data_timeout_token += 1
        token = n.data_timeout_token
        timeout_s = self._data_inactivity_timeout_s()

        def proc():
            yield self.env.timeout(max(0.0, float(from_t) + timeout_s - self.env.now))
            if n.data_timeout_token != token:
                return
            if n.updated:
                return
            # No DATA/END activity for the inactivity timeout: cache received chunks and sleep until next cycle.
            self._stop_dynamic_listen(n, n.data_listen_idx, self.env.now)
            n.data_listen_idx = None

        self.env.process(proc())

    def _schedule_repair_inactivity_timeout(self, n: Node, version: int, owner: int, from_t: float) -> None:
        if self.env is None:
            return
        n.repair_timeout_token += 1
        token = n.repair_timeout_token
        timeout_s = self._repair_inactivity_timeout_s()

        def proc():
            yield self.env.timeout(max(0.0, float(from_t) + timeout_s - self.env.now))
            if n.repair_timeout_token != token:
                return
            self._stop_dynamic_listen(n, n.repair_listen_idx, self.env.now)
            n.repair_listen_idx = None
            n.repair_done_today = True

        self.env.process(proc())

    def _on_data_payload_rx(self, n: Node, p: Packet) -> None:
        """Reset DATA inactivity timer after accepted unique DATA packet."""
        if n.owner_for_day != p.src:
            return
        self._schedule_data_inactivity_timeout(n, p.version, p.src, p.start_s + p.airtime_s)

    def _on_repair_payload_rx(self, n: Node, p: Packet) -> None:
        """Reset REPAIR inactivity timer after accepted unique REPAIR packet."""
        if n.owner_for_day != p.src:
            return
        self._schedule_repair_inactivity_timeout(n, p.version, p.src, p.start_s + p.airtime_s)

    def _on_end_rx(self, n: Node, p: Packet) -> None:
        """Handle END1/END2/END3 for dynamic DATA/REPAIR stages."""
        if n.nid == p.src:
            return
        phase = str(getattr(p, "phase", "") or "").upper()
        wait_s = self._end_wait_s(p)
        transition_t = (p.start_s + p.airtime_s) + wait_s
        if phase == "DATA":
            if n.owner_for_day != p.src:
                return
            # END cancels the inactivity timeout and causes a transition to MISS after END wait.
            n.data_timeout_token += 1
            self._stop_dynamic_listen(n, n.data_listen_idx, transition_t)
            if not n.data_miss_scheduled:
                n.data_miss_scheduled = True
                self._schedule_miss_for_receiver(n, p, transition_t)
        elif phase == "REPAIR":
            if n.owner_for_day != p.src:
                return
            n.repair_timeout_token += 1
            self._stop_dynamic_listen(n, n.repair_listen_idx, transition_t)
            n.repair_done_today = True

    def _schedule_miss_for_receiver(self, n: Node, end_packet: Packet, ready_t: float) -> None:
        """Schedule one owner-local MISS packet after DATA END if chunks are missing."""
        if self.env is None:
            return
        args = self.args
        owner = end_packet.src
        version = end_packet.version

        def proc():
            yield self.env.timeout(max(0.0, float(ready_t) - self.env.now))
            if n.updated:
                return

            expected_chunks = args.complete_need
            if version in n.last_chunk_idx:
                expected_chunks = min(args.complete_need, n.last_chunk_idx[version] + 1)
            have = n.have_chunks.get(version, set())
            missing_idxs = [i for i in range(expected_chunks) if i not in have]
            if not missing_idxs:
                return

            self.day_ctr['miss_candidates'] += 1
            miss_len = _miss_payload_length(
                missing_indices=missing_idxs,
                chunks=expected_chunks,
                method=args.miss_encoding,
                hdr_bytes=args.miss_hdr_bytes,
            )
            airt = self.radio.airtime_seconds_for_args(
                args, args.sf_adv, args.bw_hz, args.cr_denom, miss_len
            )
            miss_logic = self.logical_airtime_s("MISS", airt)
            miss_min_ms, miss_max_ms = args.miss_backoff_ms
            lo = min(miss_min_ms, miss_max_ms) / 1000.0
            hi = max(miss_min_ms, miss_max_ms) / 1000.0
            bo = self.node_rng[n.nid].uniform(max(0.0, lo), max(0.0, hi)) if hi > 0.0 or lo > 0.0 else 0.0
            start = self.env.now + args.rx_to_tx_s + bo
            miss_chan = n.listen_chan_data
            if miss_chan is None:
                owner_node = self.by_id[owner]
                miss_chan = owner_node.data_chan if owner_node.data_chan is not None else args.chan_data

            p_miss = Packet(
                "MISS",
                src=n.nid,
                dst=owner,
                sf=args.sf_adv,
                bw_hz=args.bw_hz,
                cr_denom=args.cr_denom,
                start_s=start,
                airtime_s=airt,
                power_dbm=args.tx_dbm,
                chan=miss_chan,
                version=version,
                _requested_indices=missing_idxs,
                _miss_payload_bytes=miss_len,
                phase="MISS",
            )
            self.tx_logical_packet(p_miss, "miss_used_s")

            # Start repair listening. It is dynamically shortened by REPAIR END, completion, or timeout.
            owner_repair_start = self.owner_repair_start_est.get(owner, start + miss_logic)
            listen_start = start + miss_logic
            listen_until = max(getattr(self, "day_safety_end_t", listen_start), owner_repair_start + self._repair_inactivity_timeout_s())
            repair_chan = miss_chan
            n.repair_listen_chan = repair_chan
            n.repair_listen_idx = self._start_dynamic_listen(n, listen_start, listen_until, repair_chan, "REPAIR")
            # Timeout is measured from the expected owner-local repair start, so a node does not sleep during MISS collection.
            self._schedule_repair_inactivity_timeout(n, version, owner, max(listen_start, owner_repair_start))

        self.env.process(proc())

    def _duty_cycle_enabled(self) -> bool:
        return float(getattr(self.args, "duty_cycle_limit_s", 0.0) or 0.0) > 0.0

    def _duty_data_budget_s(self, owner: Node) -> float:
        """DATA-side TX airtime budget for this owner in the current cycle.

        With a 1% per-hour duty-cycle abstraction this is usually 18 s when
        --duty-cycle-limit-s=36 and --duty-cycle-data-fraction=0.5. If enabled,
        the owner's ADV airtime already spent in this cycle is subtracted from
        this DATA-side budget.
        """
        limit = float(getattr(self.args, "duty_cycle_limit_s", 0.0) or 0.0)
        if limit <= 0.0:
            return float("inf")
        frac = min(1.0, max(0.0, float(getattr(self.args, "duty_cycle_data_fraction", 0.5) or 0.5)))
        budget = limit * frac
        if bool(getattr(self.args, "duty_cycle_include_adv_in_data", True)):
            start_adv = getattr(self, "day_start_adv_used_s", {}).get(owner.nid, owner.adv_used_s)
            adv_today = max(0.0, float(owner.adv_used_s) - float(start_adv))
            budget -= adv_today
        return max(0.0, budget)

    def _duty_repair_budget_s(self, owner: Node) -> float:
        """REPAIR-side TX airtime budget for this owner in the current cycle."""
        limit = float(getattr(self.args, "duty_cycle_limit_s", 0.0) or 0.0)
        if limit <= 0.0:
            return float("inf")
        frac = min(1.0, max(0.0, float(getattr(self.args, "duty_cycle_data_fraction", 0.5) or 0.5)))
        return max(0.0, limit * (1.0 - frac))

    def _max_payload_packets_for_budget(self, budget_s: float, per_payload_time: float, per_end_time: float) -> int:
        """Maximum DATA/REPAIR payload packets that fit with 3 END packets.

        The protocol needs END1/END2/END3 after a non-empty train so receivers can
        transition without timing out. If there is not enough budget for one
        payload packet plus three END packets, return 0 and skip this train.
        """
        if math.isinf(float(budget_s)):
            return 10**12
        available_for_payload = float(budget_s) - 3.0 * float(per_end_time)
        if available_for_payload < float(per_payload_time):
            return 0
        return max(0, int(math.floor(available_for_payload / float(per_payload_time))))

    def _send_owner_data_train(self, o: Node, start_t: float, data_airt: float, per_data_time: float, per_end_time: float) -> float:
        """Schedule DATA train and END1/END2/END3 for one owner. Return finish time.

        If duty-cycle limiting is enabled, the owner sends only the portion of
        the update that fits in the DATA-side duty-cycle budget for this cycle.
        Progress is remembered in o.data_next_chunk_by_version so the next cycle
        continues from the remaining chunks instead of restarting from chunk 0.
        """
        args = self.args
        if not self.reg_followers_day.get(o.nid, set()):
            return start_t
        owner_chunks = sorted(o.have_chunks.get(o.version, set()))
        if len(owner_chunks) < args.complete_need:
            return start_t

        configured_limit = min(int(args.data_chunks_per_round), len(owner_chunks))
        if configured_limit <= 0:
            return start_t

        duty_budget_s = self._duty_data_budget_s(o)
        duty_limit_chunks = self._max_payload_packets_for_budget(duty_budget_s, per_data_time, per_end_time)
        chunks_to_send = min(configured_limit, len(owner_chunks), duty_limit_chunks)
        if chunks_to_send <= 0:
            self.day_ctr["duty_data_skips"] += 1
            return start_t

        data_chan = o.data_chan if o.data_chan is not None else args.chan_data
        cursor = start_t

        # Continue from where this owner stopped in the previous cycle. If the
        # pointer reaches the end, wrap around so older missed chunks can be
        # repeated in later cycles.
        start_pos = int(o.data_next_chunk_by_version.get(o.version, 0) or 0)
        if start_pos < 0 or start_pos >= len(owner_chunks):
            start_pos = 0
        send_indices = owner_chunks[start_pos:start_pos + chunks_to_send]
        if len(send_indices) < chunks_to_send:
            # Wrap within the same cycle only if the configured/duty budget allows
            # more chunks than remain at the tail. This keeps long-running owners
            # useful after one full pass through the update.
            send_indices.extend(owner_chunks[:chunks_to_send - len(send_indices)])

        for chunk_idx in send_indices:
            p = Packet(
                "DATA",
                src=o.nid,
                dst=None,
                sf=args.sf_data,
                bw_hz=args.bw_hz,
                cr_denom=args.cr_denom,
                start_s=cursor,
                airtime_s=data_airt,
                power_dbm=args.tx_dbm,
                chan=data_chan,
                version=o.version,
                chunk_idx=chunk_idx,
                session_id=o.nid & 0xFFFF,
                phase="DATA",
            )
            self.tx_logical_packet(p, "data_used_s")
            cursor += per_data_time

        next_pos = (start_pos + len(send_indices)) % len(owner_chunks)
        o.data_next_chunk_by_version[o.version] = next_pos
        self.first_round_sent[o.nid] = (o.version, len(send_indices))
        self.day_ctr["duty_data_chunks_scheduled"] += len(send_indices)

        last_idx = send_indices[-1]
        for seq in (1, 2, 3):
            p_end = Packet(
                "END",
                src=o.nid,
                dst=None,
                sf=args.sf_data,
                bw_hz=args.bw_hz,
                cr_denom=args.cr_denom,
                start_s=cursor,
                airtime_s=data_airt,
                power_dbm=args.tx_dbm,
                chan=data_chan,
                version=o.version,
                chunk_idx=last_idx,
                session_id=o.nid & 0xFFFF,
                phase="DATA",
                end_seq=seq,
            )
            self.tx_logical_packet(p_end, "data_used_s")
            cursor += per_end_time
        return cursor

    def _send_owner_repair_train(self, oid: int, start_t: float, repair_airt: float, per_repair_time: float, per_end_time: float) -> float:
        """Schedule REPAIR train and END1/END2/END3 for one owner. Return finish time.

        When duty-cycle limiting is enabled, REPAIR uses the repair-side budget
        reserved by --duty-cycle-data-fraction.
        """
        args = self.args
        reqset = self.accepted_repair_day.get(oid, set())
        if not reqset:
            return start_t
        owner_node = self.by_id[oid]
        repair_chan = owner_node.data_chan if owner_node.data_chan is not None else args.chan_data
        cursor = start_t
        count = 0
        sent_indices: List[Tuple[int, int]] = []

        max_repairs = int(getattr(args, "repair_chunks_per_round", 0) or 0)
        if max_repairs <= 0:
            max_repairs = len(reqset)

        duty_budget_s = self._duty_repair_budget_s(owner_node)
        duty_limit_repairs = self._max_payload_packets_for_budget(duty_budget_s, per_repair_time, per_end_time)
        max_repairs = min(max_repairs, duty_limit_repairs)
        if max_repairs <= 0:
            self.day_ctr["duty_repair_skips"] += 1
            return start_t

        for (v, idx) in sorted(reqset, key=lambda t: (t[0], t[1])):
            if count >= max_repairs:
                break
            if idx not in owner_node.have_chunks.get(v, set()):
                continue
            p = Packet(
                "REPAIR",
                src=oid,
                dst=None,
                sf=args.sf_data,
                bw_hz=args.bw_hz,
                cr_denom=args.cr_denom,
                start_s=cursor,
                airtime_s=repair_airt,
                power_dbm=args.tx_dbm,
                chan=repair_chan,
                version=v,
                chunk_idx=idx,
                session_id=oid & 0xFFFF,
                phase="REPAIR",
            )
            self.tx_logical_packet(p, "repair_used_s")
            cursor += per_repair_time
            count += 1
            sent_indices.append((v, idx))

        self.day_ctr["duty_repair_chunks_scheduled"] += len(sent_indices)
        if sent_indices:
            v_last, max_idx = max(sent_indices, key=lambda t: (t[0], t[1]))
            for seq in (1, 2, 3):
                p_end = Packet(
                    "END",
                    src=oid,
                    dst=None,
                    sf=args.sf_data,
                    bw_hz=args.bw_hz,
                    cr_denom=args.cr_denom,
                    start_s=cursor,
                    airtime_s=repair_airt,
                    power_dbm=args.tx_dbm,
                    chan=repair_chan,
                    version=v_last,
                    session_id=oid & 0xFFFF,
                    chunk_idx=max_idx,
                    phase="REPAIR",
                    end_seq=seq,
                )
                self.tx_logical_packet(p_end, "repair_used_s")
                cursor += per_end_time
        return cursor

    def _owner_local_data_repair_process(self, o: Node, data_start: float, data_airt: float, per_data_time: float, per_end_time: float, miss_collect_s: float, repair_airt: float, per_repair_time: float, per_repair_end_time: float):
        """Owner-local DATA -> MISS listen -> REPAIR process."""
        if self.env is None:
            return
        yield self.env.timeout(max(0.0, data_start - self.env.now))
        data_finish = self._send_owner_data_train(o, data_start, data_airt, per_data_time, per_end_time)
        data_chan = o.data_chan if o.data_chan is not None else self.args.chan_data
        miss_listen_start = data_finish
        miss_listen_end = data_finish + miss_collect_s
        if miss_listen_end > miss_listen_start:
            o.owner_miss_listen_idx = self._start_dynamic_listen(o, miss_listen_start, miss_listen_end, data_chan, "MISS")
        # Remember this for receivers that sent MISS and need to wait for REPAIR.
        self.owner_repair_start_est[o.nid] = miss_listen_end
        yield self.env.timeout(max(0.0, miss_listen_end - self.env.now))
        self._stop_dynamic_listen(o, o.owner_miss_listen_idx, self.env.now)
        repair_finish = self._send_owner_repair_train(o.nid, self.env.now, repair_airt, per_repair_time, per_repair_end_time)
        self.owner_repair_finish_est[o.nid] = repair_finish
    def run_day(self):
        args = self.args
        self.env = simpy.Environment()
        self.active_packets.clear()
        self.tx_history.clear()
        self.clear_radio_states()
        self.day_ctr = defaultdict(int)
        self.reg_followers_day.clear()
        self.accepted_repair_day.clear()
        self.sent_upd_today.clear()

        # Snapshot per-node counters for per-day deltas
        start_stats = {}
        # Cumulative used_s at the start of the cycle, used by duty-cycle accounting.
        self.day_start_adv_used_s = {n.nid: n.adv_used_s for n in self.nodes}
        self.day_start_data_used_s = {n.nid: n.data_used_s for n in self.nodes}
        self.day_start_repair_used_s = {n.nid: n.repair_used_s for n in self.nodes}
        for n in self.nodes:
            start_stats[n.nid] = (
                n.tx_adv,
                n.tx_upd,
                n.tx_data,
                n.tx_miss,
                n.tx_end,
                n.tx_repair,
                n.rx_on_s,
                n.rx_busy_s,
                n.rx_on_adv_s,
                n.rx_on_upd_s,
                n.rx_on_data_s,
                n.rx_on_miss_s,
                n.rx_on_repair_s,
                n.csma_cca_s,
                n.cad_scan_s,
                n.csma_sense_s,
            )

        # Only complete update holders are allowed to advertise as owners/senders.
        owners = [n for n in self.nodes if self._is_complete_holder(n)]

        # ---------------- PHASE TIMES ----------------
        t0 = 0.0
        adv_min, adv_max = args.adv_backoff_ms
        adv_end_t = t0 + args.adv_budget_s
        # Allow extra UPD collection time after the ADV phase.
        data_start_t = adv_end_t + max(args.phase_gap_s, args.upd_listen_s)
        # DATA/REPAIR are now owner-local and receiver-timeout driven. These legacy
        # phase boundaries are kept as safety caps and for backward-compatible metadata/plots.
        data_end_t = data_start_t + args.data_budget_s
        miss_start_t = data_end_t + args.phase_gap_s
        miss_window_s = max(args.miss_budget_s, args.miss_listen_s)
        miss_end_t = miss_start_t + miss_window_s
        repair_start_t = miss_end_t + args.phase_gap_s
        repair_end_t = repair_start_t + args.repair_budget_s

        # Expose phase times for use in _evaluate_rx_for_packet
        self.adv_start_t = t0
        self.adv_end_t = adv_end_t
        self.data_start_t = data_start_t
        self.data_end_t = data_end_t
        self.miss_start_t = miss_start_t
        self.miss_end_t = miss_end_t
        self.repair_start_t = repair_start_t
        self.repair_end_t = repair_end_t

        # ---------------- DATA CHANNEL CHOICES ----------------
        # Reserve one control channel (chan_adv). DATA/MISS/REPAIR may use all remaining c - 1 channels.
        # UPD may have its own channel, but phases are separated, so it does not have to be excluded here.
        control_chan = args.chan_adv
        if args.num_channels >= 2:
            data_channels = [c for c in range(args.num_channels) if c != control_chan]
            if not data_channels:
                raise ValueError(
                    f"No available DATA channels: num_channels={args.num_channels}, "
                    f"control_chan={control_chan}. Increase --num-channels."
                )
            for n in self.nodes:
                n.data_chan = self.node_rng[n.nid].choice(data_channels)
        else:
            for n in self.nodes:
                n.data_chan = args.chan_data

        # ---------------- TUNING PLANS (ADV) ----------------
        # All nodes listen on ADV/UPD channel during entire ADV window
        for n in self.nodes:
            self._plan_channel(n, t0, adv_end_t, args.chan_adv, "ADV")

        # ---------------- ADV (all nodes advertise their current version) ----------------
        adv_pkts: List[Packet] = []
        latest_version = max(n.version for n in self.nodes)
        for n in self.nodes:
            # Only nodes that fully hold the update may advertise/send. Partial
            # receivers keep chunks, but they are not sources until complete.
            if not self._is_complete_holder(n):
                continue

            # Density/channel-based probabilistic ADV.
            p_adv = self.density_adv_probability()
            if self.node_rng[n.nid].random() > p_adv:
                continue

            # Beta-distributed backoff skewed toward newer versions
            version_gap = max(0, latest_version - n.version)
            beta_param = 1.0 + 2.0 / (1.0 + version_gap)  # newer version -> beta closer to 3
            airt = self.radio.airtime_seconds_for_args(
                args, args.sf_adv, args.bw_hz, args.cr_denom, args.adv_pl_bytes
            )
            u = self.node_rng[n.nid].betavariate(1.0, beta_param)
            bo = adv_min / 1000.0 + u * max(0.0, (adv_max - adv_min) / 1000.0)
            # Max random start is t_window - t_pkt so the packet fits inside the ADV window.
            adv_logic = self.logical_airtime_s("ADV", airt)
            start = t0 + min(bo, max(0.0, args.adv_budget_s - adv_logic))
            p = Packet(
                "ADV",
                src=n.nid,
                dst=None,
                sf=args.sf_adv,
                bw_hz=args.bw_hz,
                cr_denom=args.cr_denom,
                start_s=start,
                airtime_s=airt,
                power_dbm=args.tx_dbm,
                chan=args.chan_adv,
                version=n.version,
                data_chan=n.data_chan,
            )
            self.tx_logical_packet(p, "adv_used_s")
            adv_pkts.append(p)

        # Owners: after THEIR OWN ADV ends, they listen on UPD channel until DATA starts (for UPD)
        owner_ids = {o.nid for o in owners}
        for p in adv_pkts:
            if p.src in owner_ids:
                o = self.by_id[p.src]
                owner_upd_listen_start = p.start_s + self.logical_airtime_s("ADV", p.airtime_s)
                owner_upd_listen_end = data_start_t
                if owner_upd_listen_end > owner_upd_listen_start:
                    self._plan_channel(
                        o, owner_upd_listen_start, owner_upd_listen_end, args.chan_upd, "UPD"
                    )

        # Owner MISS-listening and receiver DATA/REPAIR listening are dynamic and owner-local.
        # Run until end of ADV window (ADV and immediate UPDs)
        self.env.run(until=adv_end_t)

        # ---------------- OWNER-LOCAL DATA -> MISS -> REPAIR ----------------
        DATA_CHUNKS_PER_ROUND = args.data_chunks_per_round
        data_airt = self.radio.airtime_seconds_for_args(
            args, args.sf_data, args.bw_hz, args.cr_denom, args.data_packet_bytes
        )
        data_logic_airt = self.logical_airtime_s("DATA", data_airt)
        end_logic_airt = self.logical_airtime_s("END", data_airt)
        per_data_time = data_logic_airt + args.proc_gap_s
        per_end_time = end_logic_airt + args.proc_gap_s

        repair_airt = data_airt
        repair_logic_airt = self.logical_airtime_s("REPAIR", repair_airt)
        repair_end_logic_airt = self.logical_airtime_s("END", repair_airt)
        per_repair_time = repair_logic_airt + args.proc_gap_s
        per_repair_end_time = repair_end_logic_airt + args.proc_gap_s

        # MISS listen collection time per owner after DATA END3. It must cover:
        # receiver END1 delay (2 END airtimes), RX->TX turnaround, random MISS backoff, and MISS airtime.
        miss_len_for_timing = self._miss_len_for_control_window()
        miss_airt_for_timing = self.radio.airtime_seconds_for_args(
            args, args.sf_adv, args.bw_hz, args.cr_denom, miss_len_for_timing
        )
        miss_logic_for_timing = self.logical_airtime_s("MISS", miss_airt_for_timing)
        miss_backoff_max_s = max(float(args.miss_backoff_ms[0]), float(args.miss_backoff_ms[1])) / 1000.0
        miss_collect_s = (
            2.0 * end_logic_airt
            + args.rx_to_tx_s
            + miss_backoff_max_s
            + miss_logic_for_timing
            + max(0.0, float(getattr(args, "owner_miss_extra_airtimes", 2.0) or 0.0)) * miss_logic_for_timing
        )

        self.first_round_sent: Dict[int, Tuple[int, int]] = {}
        self.owner_repair_start_est: Dict[int, float] = {}
        self.owner_repair_finish_est: Dict[int, float] = {}

        # Safety horizon: dynamic timeouts should make nodes sleep, but we still cap the day.
        data_bo_max_s = max(float(args.data_backoff_ms[0]), float(args.data_backoff_ms[1])) / 1000.0
        max_data_chunks_for_safety = int(args.data_chunks_per_round)
        if float(getattr(args, "duty_cycle_limit_s", 0.0) or 0.0) > 0.0:
            max_data_chunks_for_safety = min(
                max_data_chunks_for_safety,
                self._max_payload_packets_for_budget(
                    float(args.duty_cycle_limit_s) * float(args.duty_cycle_data_fraction),
                    per_data_time,
                    per_end_time,
                ),
            )
        max_data_train_s = max_data_chunks_for_safety * per_data_time + (3 * per_end_time if max_data_chunks_for_safety > 0 else 0.0)

        max_repair_chunks = int(getattr(args, "repair_chunks_per_round", 0) or 0)
        if max_repair_chunks <= 0:
            max_repair_chunks = int(args.complete_need)
        if float(getattr(args, "duty_cycle_limit_s", 0.0) or 0.0) > 0.0:
            max_repair_chunks = min(
                max_repair_chunks,
                self._max_payload_packets_for_budget(
                    float(args.duty_cycle_limit_s) * (1.0 - float(args.duty_cycle_data_fraction)),
                    per_repair_time,
                    per_repair_end_time,
                ),
            )
        max_repair_train_s = max_repair_chunks * per_repair_time + (3 * per_repair_end_time if max_repair_chunks > 0 else 0.0)
        owner_stagger_s = max(per_data_time, 1e-9)

        # Only owners that actually received at least one UPD should consume an
        # owner-local DATA slot. Otherwise late active owners get delayed behind
        # inactive owners, while their receivers may time out before DATA begins.
        active_owners = [o for o in owners if self.reg_followers_day.get(o.nid, set())]

        day_safety = (
            data_start_t
            + data_bo_max_s
            + max(1, len(active_owners)) * owner_stagger_s
            + max_data_train_s
            + miss_collect_s
            + max_repair_train_s
            + self._repair_inactivity_timeout_s()
            + 1.0
        )
        self.day_safety_end_t = day_safety
        self.data_end_t = day_safety
        self.repair_end_t = day_safety
        # ADV receptions happened before this safety horizon was known. Extend dynamic DATA listens to the new cap.
        for n in self.nodes:
            if n.data_listen_idx is not None and 0 <= n.data_listen_idx < len(n.dynamic_tune_plan):
                n.dynamic_tune_plan[n.data_listen_idx][1] = max(n.dynamic_tune_plan[n.data_listen_idx][1], day_safety)

        # Compute deterministic owner DATA start estimates. Active owners are
        # staggered among active owners only. Inactive owners still get a nominal
        # start estimate so receivers whose UPD was not accepted do not listen
        # until day_safety; they time out after the expected no-traffic interval.
        self.owner_data_start_est: Dict[int, float] = {}
        owner_data_backoff: Dict[int, float] = {}
        data_bo_min_ms, data_bo_max_ms = args.data_backoff_ms
        for o in owners:
            data_bo = 0.0
            if data_bo_max_ms > 0.0 or data_bo_min_ms > 0.0:
                lo = min(data_bo_min_ms, data_bo_max_ms)
                hi = max(data_bo_min_ms, data_bo_max_ms)
                data_bo = self.node_rng[o.nid].uniform(lo, hi) / 1000.0
            owner_data_backoff[o.nid] = data_bo

        active_idx_map = {o.nid: idx for idx, o in enumerate(active_owners)}
        for o in owners:
            if o.nid in active_idx_map:
                start_t = data_start_t + owner_data_backoff[o.nid] + active_idx_map[o.nid] * owner_stagger_s
            else:
                start_t = data_start_t + owner_data_backoff[o.nid]
            self.owner_data_start_est[o.nid] = start_t

            # Arm initial DATA inactivity timeout from this owner's estimated DATA
            # start, not from the global DATA start. Accepted DATA packets reset it.
            for n in self.nodes:
                if n.owner_for_day == o.nid and n.data_listen_idx is not None:
                    self._schedule_data_inactivity_timeout(n, o.version, o.nid, start_t)

        # Start owner-local DATA->MISS->REPAIR only for active owners.
        for o in active_owners:
            start_t = self.owner_data_start_est[o.nid]
            self.env.process(
                self._owner_local_data_repair_process(
                    o,
                    start_t,
                    data_airt,
                    per_data_time,
                    per_end_time,
                    miss_collect_s,
                    repair_airt,
                    per_repair_time,
                    per_repair_end_time,
                )
            )

        self.env.run(until=day_safety)

        # End-of-day promotion: any node that has full model for a newer version becomes updated/owner next day
        for n in self.nodes:
            for v, chunks in n.have_chunks.items():
                if v > n.version and len(chunks) >= args.complete_need:
                    n.version = v
                    n.adv_cycles = 0
                    n.version_changed_today = True
                    n.updated = True
                    n.is_gateway = True

        # Mark partial receivers for visualization:
        # nodes that are not updated yet but received at least one DATA chunk this day
        for n in self.nodes:
            gained_today = len(n.day_rx_chunks)
            n.last_day_partial = (not n.updated) and (gained_today > 0)

        # Save animation frame if requested
        if self.args.animate:
            self.save_day_snapshot(self.day_idx)

        # ADV probability is density/channel based now, not cycle based.
        # adv_cycles/version_changed_today are kept only for backward-compatible state/visualization.

        # Record per-day deltas per node for energy accounting
        for n in self.nodes:
            # First compute RX-on time for this day, split by protocol mode.
            day_rx_by_mode = self._compute_rx_on_by_mode_for_day(n)
            day_rx_adv_s = day_rx_by_mode.get("ADV", 0.0)
            day_rx_upd_s = day_rx_by_mode.get("UPD", 0.0)
            day_rx_data_s = day_rx_by_mode.get("DATA", 0.0)
            day_rx_miss_s = day_rx_by_mode.get("MISS", 0.0)
            day_rx_repair_s = day_rx_by_mode.get("REPAIR", 0.0)
            day_rx_window_s = day_rx_adv_s + day_rx_upd_s + day_rx_data_s + day_rx_miss_s + day_rx_repair_s + day_rx_by_mode.get("UNKNOWN", 0.0)

            default_stats = (0,0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
            (
                sa, su, sd, sm, se, sr,
                srx_on, srx_busy,
                srx_adv, srx_upd, srx_data, srx_miss, srx_repair,
                scsma_cca, scad_scan, scsma_sense,
            ) = start_stats.get(n.nid, default_stats)

            day_csma_cca_s = max(0.0, n.csma_cca_s - scsma_cca)
            day_cad_scan_s = max(0.0, n.cad_scan_s - scad_scan)
            day_csma_sense_s = max(0.0, n.csma_sense_s - scsma_sense)
            # Backward compatibility: if only aggregate was updated, preserve it.
            if day_csma_sense_s <= 0.0:
                day_csma_sense_s = day_csma_cca_s + day_cad_scan_s

            # Accumulate split RX-on windows.
            n.rx_on_adv_s += day_rx_adv_s
            n.rx_on_upd_s += day_rx_upd_s
            n.rx_on_data_s += day_rx_data_s
            n.rx_on_miss_s += day_rx_miss_s
            n.rx_on_repair_s += day_rx_repair_s
            # RX-on energy includes ordinary tuning windows plus CSMA CCA and CAD scan time.
            n.rx_on_s += day_rx_window_s + day_csma_cca_s + day_cad_scan_s

            n.day_tx_adv.append(n.tx_adv - sa)
            n.day_tx_upd.append(n.tx_upd - su)
            n.day_tx_data.append(n.tx_data - sd)
            n.day_tx_miss.append(n.tx_miss - sm)
            n.day_tx_end.append(n.tx_end - se)
            n.day_tx_repair.append(n.tx_repair - sr)
            n.day_rx_on_ms.append(round((n.rx_on_s - srx_on) * 1000.0, 6))
            n.day_rx_on_adv_ms.append(round((n.rx_on_adv_s - srx_adv) * 1000.0, 6))
            n.day_rx_on_upd_ms.append(round((n.rx_on_upd_s - srx_upd) * 1000.0, 6))
            n.day_rx_on_data_ms.append(round((n.rx_on_data_s - srx_data) * 1000.0, 6))
            n.day_rx_on_miss_ms.append(round((n.rx_on_miss_s - srx_miss) * 1000.0, 6))
            n.day_rx_on_repair_ms.append(round((n.rx_on_repair_s - srx_repair) * 1000.0, 6))
            n.day_rx_busy_ms.append(round((n.rx_busy_s - srx_busy) * 1000.0, 6))
            n.day_csma_cca_ms.append(round(day_csma_cca_s * 1000.0, 6))
            n.day_cad_scan_ms.append(round(day_cad_scan_s * 1000.0, 6))
            n.day_csma_sense_ms.append(round(day_csma_sense_s * 1000.0, 6))

        self.day_idx += 1

        if args.debug:
            # Additional diagnostics to understand why MISS/REPAIR may be inactive.
            # rx_* counters are successful decodes, tx_* counters are transmissions scheduled.
            # miss_candidates counts nodes that (a) latched onto an owner and (b) have partial chunks and thus *could* send MISS.
            accepted_chunks = sum(len(s) for s in self.accepted_repair_day.values())
            accepted_owners = sum(1 for s in self.accepted_repair_day.values() if s)
            # Partial chunk holders for the newest owner version observed (roughly)
            partial_any = 0
            any_chunks = 0
            for n in self.nodes:
                for v, hs in n.have_chunks.items():
                    if v > n.version and len(hs) > 0:
                        any_chunks += 1
                        if len(hs) < args.complete_need:
                            partial_any += 1
                        break

            print(
                f"[DEBUG] Day {self.day_idx} RX: ADV={self.day_ctr['rx_adv']}  UPD={self.day_ctr['rx_upd']}  "
                f"DATA={self.day_ctr['rx_data']}  MISS={self.day_ctr['rx_miss']}"
            )
            print(
                f"[DEBUG] Day {self.day_idx} TX: ADV={self.day_ctr['tx_adv']}  UPD={self.day_ctr['tx_upd']}  "
                f"DATA={self.day_ctr['tx_data']}  MISS={self.day_ctr['tx_miss']}  END={self.day_ctr['tx_end']}"
            )
            print(
                f"[DEBUG] Day {self.day_idx} MISS candidates={self.day_ctr['miss_candidates']}  "
                f"nodes_with_any_new_chunks={any_chunks}  nodes_with_partial_new_chunks={partial_any}"
            )
            print(
                f"[DEBUG] Day {self.day_idx} accepted MISS owners={accepted_owners}  "
                f"accepted requested chunks={accepted_chunks}"
            )
            if bool(getattr(args, "csma_ca", False)):
                csma_sense_ms = sum((n.day_csma_sense_ms[-1] if n.day_csma_sense_ms else 0.0) for n in self.nodes)
                print(
                    f"[DEBUG] Day {self.day_idx} CSMA: deferrals={self.day_ctr['csma_deferrals']} "
                    f"drops={self.day_ctr['csma_drops']} sense_ms={csma_sense_ms:.3f}"
                )
            if float(getattr(args, "duty_cycle_limit_s", 0.0) or 0.0) > 0.0:
                print(
                    f"[DEBUG] Day {self.day_idx} duty: DATA scheduled={self.day_ctr['duty_data_chunks_scheduled']} "
                    f"DATA skips={self.day_ctr['duty_data_skips']} REPAIR scheduled={self.day_ctr['duty_repair_chunks_scheduled']} "
                    f"REPAIR skips={self.day_ctr['duty_repair_skips']}"
                )

    def save_day_snapshot(self, day_idx: int):
        if plt is None:
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        # Regular nodes (non-owners)
        xs_idle, ys_idle = [], []
        xs_partial, ys_partial = [], []
        xs_updated, ys_updated = [], []

        # Owners (gateways) colored by cycles-on-version (adv_cycles)
        xs_owner, ys_owner, owner_cycle_vals = [], [], []

        for n in self.nodes:
            if n.is_gateway:
                xs_owner.append(n.x)
                ys_owner.append(n.y)
                cyc = getattr(n, 'adv_cycles', 0)
                cyc_bucket = min(max(cyc + 1, 1), 5)  # 1..5+ bucket
                owner_cycle_vals.append(cyc_bucket)
                continue

            if n.updated:
                xs_updated.append(n.x)
                ys_updated.append(n.y)
            else:
                if n.last_day_partial:
                    xs_partial.append(n.x)
                    ys_partial.append(n.y)
                else:
                    xs_idle.append(n.x)
                    ys_idle.append(n.y)

        # Plot non-owners
        if xs_idle:
            ax.scatter(xs_idle, ys_idle, s=10, label='idle', c='#1f77b4')
        if xs_partial:
            ax.scatter(xs_partial, ys_partial, s=14, label='partial', c='#f39c12')
        if xs_updated:
            ax.scatter(xs_updated, ys_updated, s=20, label='updated', c='#2ecc71')

        # Owners with graded colors (green → orange → purple)
        legend_handles = []
        try:
            from matplotlib.colors import LinearSegmentedColormap, Normalize
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            legend_handles.extend([
                Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#1f77b4', markeredgecolor='none', markersize=6, label='idle'),
                Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#f39c12', markeredgecolor='none', markersize=7, label='partial'),
                Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#2ecc71', markeredgecolor='none', markersize=7, label='updated'),
            ])

            if xs_owner:
                cmap = LinearSegmentedColormap.from_list(
                    'owner_cycles',
                    ['#2ecc71', '#f39c12', '#8e44ad'],
                    N=256,
                )
                norm = Normalize(vmin=1, vmax=5)
                owner_colors = [cmap(norm(v)) for v in owner_cycle_vals]
                ax.scatter(
                    xs_owner,
                    ys_owner,
                    s=70,
                    c=owner_colors,
                    edgecolors='black',
                    linewidths=0.8,
                    zorder=5,
                )

                for v, lab in [
                    (1, 'owner cycle 1'),
                    (2, 'owner cycle 2'),
                    (3, 'owner cycle 3'),
                    (4, 'owner cycle 4'),
                    (5, 'owner cycle ≥5'),
                ]:
                    legend_handles.append(Patch(facecolor=cmap(norm(v)), edgecolor='black', label=lab))
        except Exception:
            # Fallback if matplotlib extras aren't available
            if xs_owner:
                ax.scatter(xs_owner, ys_owner, s=60, facecolors='none', edgecolors='black', linewidths=1.2, label='owners')

        total = len(self.nodes)
        upd = sum(1 for _n in self.nodes if _n.updated)
        ax.set_title(f"Day {day_idx+1}: updated={upd}/{total}")
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.6)

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
        else:
            ax.legend(loc='upper right', fontsize=8)

        out = os.path.join(self.out_dir, f"day_{day_idx+1:03d}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)



def _phy_repetitions_for_ptype(args, ptype: str) -> int:
    reps = int(getattr(args, "phy_repetitions", 1) or 1)
    if reps <= 1:
        return 1
    stages = {str(x).upper() for x in getattr(args, "phy_repeat_stages", [])}
    return reps if str(ptype).upper() in stages else 1


def _logical_airtime_s_for_args(args, ptype: str, single_airtime_s: float) -> float:
    reps = _phy_repetitions_for_ptype(args, ptype)
    if reps <= 1:
        return single_airtime_s
    gap_s = (float(getattr(args, "phy_repeat_gap_ms", 0.0) or 0.0) + float(getattr(args, "phy_repeat_jitter_ms", 0.0) or 0.0)) / 1000.0
    return reps * single_airtime_s + (reps - 1) * max(0.0, gap_s)


def _apply_auto_data_repair_windows(args, radio: RadioProfile) -> None:
    """Compute DATA and REPAIR phase durations from model size and packet airtime.

    DATA duration includes:
      - all DATA chunks (`complete_need`)
      - three END packets after DATA
      - processing gaps between packets
      - maximum DATA random backoff, so the latest possible sender still fits

    REPAIR duration is set equal to DATA duration as a conservative worst-case
    assumption: collectively, listeners may request up to the whole model.
    """
    if not getattr(args, "auto_data_repair_windows", False):
        return

    chunks = max(0, int(getattr(args, "complete_need", 0)))
    end_packets = 3
    total_packets = chunks + end_packets

    data_airt = radio.airtime_seconds_for_args(
        args,
        args.sf_data,
        args.bw_hz,
        args.cr_denom,
        args.data_packet_bytes,
    )

    proc_gap_s = float(getattr(args, "proc_gap_s", 0.0))
    data_logic_airt = _logical_airtime_s_for_args(args, "DATA", data_airt)
    end_logic_airt = _logical_airtime_s_for_args(args, "END", data_airt)
    if total_packets <= 0:
        tx_train_s = 0.0
    else:
        tx_train_s = chunks * data_logic_airt + end_packets * end_logic_airt + max(0, total_packets - 1) * proc_gap_s

    data_backoff_ms = getattr(args, "data_backoff_ms", [0.0, 0.0])
    try:
        data_backoff_max_s = max(float(data_backoff_ms[0]), float(data_backoff_ms[1])) / 1000.0
    except Exception:
        data_backoff_max_s = 0.0

    margin = max(0.0, float(getattr(args, "data_repair_window_margin", 1.0) or 1.0))
    computed_s = (tx_train_s + data_backoff_max_s) * margin
    computed_s = max(computed_s, max(data_logic_airt, end_logic_airt))

    args.data_budget_s = computed_s
    args.repair_budget_s = computed_s

    args.auto_data_repair_data_airtime_s = data_logic_airt
    args.auto_data_repair_single_data_airtime_s = data_airt
    args.auto_data_repair_end_airtime_s = end_logic_airt
    args.auto_data_repair_total_packets = total_packets
    args.auto_data_repair_tx_train_s = tx_train_s
    args.auto_data_repair_data_backoff_max_s = data_backoff_max_s
    args.auto_data_repair_computed_budget_s = computed_s

def main():
    args = parse_args()
    global RADIO_PROFILE
    RADIO_PROFILE = get_radio_profile(args)

    # If model size is specified, derive complete_need and data_chunks_per_round from it
    if getattr(args, "model_size_bytes", 0) and args.model_size_bytes > 0:
        total_chunks = math.ceil(args.model_size_bytes / args.data_pl_bytes)
        args.complete_need = total_chunks
        args.data_chunks_per_round = total_chunks

    # Optional dynamic DATA/REPAIR phase sizing. Must run after complete_need is derived.
    _apply_auto_data_repair_windows(args, RADIO_PROFILE)

    if args.animate and plt is None:
        print("[WARN] --animate requested but matplotlib not available; disabling.")
        args.animate = False
    if args.seed is None:
        args.seed = random.randrange(1 << 31)
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    nodes = load_testbed(args.testbed)
    sim = Sim(args, nodes, rng)
    sim.apply_auto_control_windows()
    sim.write_metadata(args.testbed)
    for _ in range(args.max_days):
        print(
            f"\n=== Day {sim.day_idx+1}: {sum(1 for n in nodes if n.is_gateway)} owners ==="
        )
        sim.run_day()
        total_updated = sum(1 for n in nodes if n.updated)
        print(
            f"Day {sim.day_idx} summary: CUMULATIVE updated {total_updated}/{len(nodes)}"
        )
        if args.until_complete and total_updated == len(nodes):
            break
    sim.write_node_summary()
    sim.write_node_state()
    print("Wrote node_summary.csv, node_state.csv, metadata.csv")


if __name__ == "__main__":
    main()
