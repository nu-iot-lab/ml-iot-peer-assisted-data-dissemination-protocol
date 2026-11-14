#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deluge-style page-based data dissemination simulator.

This simulator keeps the same radio/profile/channel assumptions as the original
`dist_lora_sim.py`, but replaces the protocol with a simplified Deluge-like
ADV -> REQ -> DATA cycle. Nodes may forward completed pages before they have the
full object, but `updated=True` is set only when the full object is complete.
"""
from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import simpy

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from radio_profiles import RadioProfile, get_radio_profile
from models_deluge import Node, Packet, _splitmix64
from config_deluge import parse_args
from io_utils_deluge import OutputMixin, load_testbed
from channel_deluge import ChannelMixin

RADIO_PROFILE: Optional[RadioProfile] = None


def _radio_profile() -> RadioProfile:
    if RADIO_PROFILE is None:
        raise RuntimeError("RADIO_PROFILE is not initialized")
    return RADIO_PROFILE


class AdvInfo:
    def __init__(self, src: int, version: int, pages: List[int], data_chan: int, t: float):
        self.src = int(src)
        self.version = int(version)
        self.pages = sorted(set(int(x) for x in pages))
        self.data_chan = int(data_chan)
        self.t = float(t)


class Sim(ChannelMixin, OutputMixin):
    def __init__(self, args, nodes: List[Node], rng: random.Random):
        self.args = args
        self.nodes = nodes
        self.rng = rng
        self.radio = _radio_profile()
        self.node_rng: Dict[int, random.Random] = {}
        base_seed = int(getattr(args, "seed", 0) or 0)
        for n in nodes:
            mixed = _splitmix64((base_seed ^ (n.nid * 0xD1B54A32D192ED03)) & 0xFFFFFFFFFFFFFFFF)
            self.node_rng[n.nid] = random.Random(mixed)
        self.day_idx = 0
        self.out_dir = args.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.env: Optional[simpy.Environment] = None
        self.active_packets: List[Packet] = []
        self.tx_history: List[Packet] = []
        self.by_id = {n.nid: n for n in nodes}
        self.next_logical_packet_id = 0
        self.day_ctr = defaultdict(int)
        self.requests_by_sender_page: DefaultDict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.requesters_by_sender_page: DefaultDict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.day_safety_end_t = 0.0
        self.packet_log_path = os.path.join(self.out_dir, "packet_log.csv")
        if not args.no_packet_log:
            with open(self.packet_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "day","ptype","src","dst","sf","bw_hz","cr_denom","start_s","airtime_s","tx_dbm","chan",
                    "version","chunk_idx","rx_node","logical_id","rep_idx","rep_count","base_chan","phase","end_seq",
                    "is_duplicate","csma_deferrals","csma_delay_s","page_idx","page_packet_idx","requested_page","requested_indices_count","session_id",
                ])
        for n in nodes:
            n.initial_source = False
        any_gw = any(n.is_gateway for n in nodes)
        if any_gw:
            for n in nodes:
                if n.is_gateway:
                    n.initial_source = True
                    self._grant_full_update(n, args.version_start)
        else:
            gw = self.rng.choice(nodes)
            gw.initial_source = True
            self._grant_full_update(gw, args.version_start)

    # ---------- sizing/state helpers ----------
    def page_count(self) -> int:
        return int(math.ceil(int(self.args.complete_need) / float(self.args.deluge_page_packets)))

    def page_for_chunk(self, chunk_idx: int) -> int:
        return int(chunk_idx) // int(self.args.deluge_page_packets)

    def chunk_range_for_page(self, page_idx: int) -> range:
        start = int(page_idx) * int(self.args.deluge_page_packets)
        end = min(int(self.args.complete_need), start + int(self.args.deluge_page_packets))
        return range(start, end)

    def page_packet_idx_for_chunk(self, chunk_idx: int) -> int:
        return int(chunk_idx) % int(self.args.deluge_page_packets)

    def is_page_complete(self, n: Node, version: int, page_idx: int) -> bool:
        chunks = set(self.chunk_range_for_page(page_idx))
        return chunks.issubset(n.have_chunks.get(version, set()))

    def _refresh_completed_pages(self, n: Node, version: int) -> None:
        for p in range(self.page_count()):
            if self.is_page_complete(n, version, p):
                n.pages_have[version].add(p)

    def _is_complete_holder(self, n: Node, version: Optional[int] = None) -> bool:
        v = n.version if version is None else version
        return v > 0 and len(n.have_chunks.get(v, set())) >= int(self.args.complete_need)

    def _grant_full_update(self, n: Node, version: int) -> None:
        n.is_gateway = True
        n.updated = True
        n.version = int(version)
        n.have_chunks[version] = set(range(int(self.args.complete_need)))
        n.pages_have[version] = set(range(self.page_count()))
        for p in range(self.page_count()):
            n.page_packets_have[(version, p)] = {self.page_packet_idx_for_chunk(c) for c in self.chunk_range_for_page(p)}

    def useful_pages(self, n: Node) -> List[int]:
        if n.version <= 0:
            return []
        self._refresh_completed_pages(n, n.version)
        return sorted(n.pages_have.get(n.version, set()))

    def testbed_area_m2(self) -> float:
        if getattr(self.args, "area_m2", None) is not None:
            return max(float(self.args.area_m2), 1.0)
        xs = [n.x for n in self.nodes]; ys = [n.y for n in self.nodes]
        return max((max(xs)-min(xs))*(max(ys)-min(ys)), 1.0) if self.nodes else 1.0

    def communication_radius_m(self) -> float:
        return self.radio.communication_radius_m(self.args, self.args.sf_adv)

    def effective_control_window_node_count(self) -> float:
        """Physical device count for control-window density and sender activation."""
        return float(len(self.nodes))

    def uses_carrier_sensing(self) -> bool:
        return bool(getattr(self.args, "cad_csma", False) or getattr(self.args, "csma_ca", False))

    def expected_neighbors(self) -> float:
        n_eff = self.effective_control_window_node_count()
        if n_eff <= 1.0:
            return 0.0
        return (n_eff - 1.0) * ((math.pi * self.communication_radius_m() ** 2) / self.testbed_area_m2())

    def density_adv_probability(self) -> float:
        expected = self.expected_neighbors()
        if expected <= 0.0:
            return 1.0
        c = int(getattr(self.args, "num_channels", 1) or 1)
        denom = expected * 0.5
        return min(1.0, max(1, c - 1) / denom) if denom > 0 else 1.0

    def control_window_s(self, packet_airtime_s: float, effective_n: Optional[float] = None) -> float:
        """Size ADV/REQ windows with the same rules as the main simulator.

        CAD/CSMA: 4.5 * (physical neighbors expected to reply) * packet airtime.
        Raw/ALOHA: collision-target window using rho and physical node count.
        """
        args = self.args
        n = float(len(self.nodes) if effective_n is None else effective_n)
        t_pkt = max(float(packet_airtime_s), 1e-12)
        p_reach = min(1.0, math.pi * self.communication_radius_m()**2 / self.testbed_area_m2())
        if self.uses_carrier_sensing():
            raw = 4.5 * max(0.0, n - 1.0) * p_reach * t_pkt
        elif n <= 2.0:
            raw = t_pkt
        else:
            rho = min(max(float(getattr(args, "reply_collision_rho", 0.05) or 0.05), 1e-12), 1.0 - 1e-12)
            denom_term = 1.0 - ((1.0 - rho) ** (1.0 / (n - 2.0)))
            raw = t_pkt + 2.0 * t_pkt * p_reach / denom_term if denom_term > 0.0 else t_pkt
        min_w = float(getattr(args, "min_control_window_s", 0.0) or 0.0)
        max_w = float(getattr(args, "max_control_window_s", 0.0) or 0.0)
        if min_w > 0.0:
            raw = max(raw, min_w)
        if max_w > 0.0:
            raw = min(raw, max_w)
        return max(raw, t_pkt)

    def _adv_logical_airtime_s(self) -> float:
        args = self.args
        airt = self.radio.airtime_seconds_for_args(args, args.sf_adv, args.bw_hz, args.cr_denom, args.adv_pl_bytes)
        return self.logical_airtime_s("ADV", airt)

    def _req_logical_airtime_s_for_window(self) -> float:
        """Conservative REQ logical airtime for automatic REQ window sizing."""
        args = self.args
        requested_count = min(max(1, int(getattr(args, "deluge_page_packets", 1) or 1)), max(1, int(getattr(args, "complete_need", 1) or 1)))
        plen = self._req_payload_len(requested_count)
        airt = self.radio.airtime_seconds_for_args(args, args.sf_adv, args.bw_hz, args.cr_denom, plen)
        return self.logical_airtime_s("REQ", airt)

    def apply_auto_control_windows(self) -> None:
        """Automatically size Deluge ADV and REQ windows/backoffs.

        Deluge does not prescribe one fixed contention-window formula, but in
        this simulator ADV and REQ are random-access control packets. Therefore
        they should scale with the expected local number of contenders rather
        than stay fixed for 20, 100 and 300 node deployments.
        """
        args = self.args
        args.effective_area_m2 = self.testbed_area_m2()
        args.communication_radius_m = self.communication_radius_m()
        args.effective_control_window_nodes = self.effective_control_window_node_count()
        args.auto_window_rule = "4.5_expected_replies_airtime" if self.uses_carrier_sensing() else "aloha_collision_target"
        args.expected_neighbors = self.expected_neighbors()
        args.adv_probability = self.density_adv_probability()

        # Always expose the packet airtimes used by the automatic model.
        adv_logic_s = self._adv_logical_airtime_s()
        req_logic_s = self._req_logical_airtime_s_for_window()
        args.auto_adv_packet_airtime_s = adv_logic_s
        args.auto_req_packet_airtime_s = req_logic_s

        # Deluge always uses density/node-count based control windows.
        # The original simulator has parser defaults for fixed windows when
        # --auto-control-windows is not passed; for Deluge we avoid that
        # behavior so ADV/REQ contention scales by default with deployment
        # size, range, packet airtime, and the configured rho/clamps.
        args.auto_control_windows = True

        adv_window_s = self.control_window_s(adv_logic_s)
        req_window_s = self.control_window_s(req_logic_s)

        args.auto_adv_window_s = adv_window_s
        args.auto_req_window_s = req_window_s
        args.deluge_adv_window_s = adv_window_s
        args.deluge_adv_window_ms = adv_window_s * 1000.0
        args.adv_budget_s = adv_window_s
        args.adv_backoff_ms = [0.0, max(0.0, adv_window_s - adv_logic_s) * 1000.0]

        args.deluge_req_window_s = req_window_s
        args.deluge_req_window_ms = req_window_s * 1000.0
        args.deluge_req_backoff_ms = [0.0, max(0.0, req_window_s - req_logic_s) * 1000.0]

    def _req_payload_len(self, requested_count: int) -> int:
        # Compact request header + missing-packet bitmap/list approximation.
        return min(int(self.args.radio_max_payload_bytes), 6 + max(1, int(requested_count)))

    def _adv_is_useful_for_node(self, n: Node, adv: AdvInfo) -> bool:
        """Return True if this ADV can lead n to request missing data.

        This is used for the Deluge request decision.  In this discrete Deluge
        model, ADV RX/listening energy is charged for the fixed ADV window
        computed by the control-window formula; receiving a useful ADV does not
        shorten the ADV listening window.

        Important: a node that is complete for its current version must still
        treat an ADV for a newer version as useful.  Otherwise a fully updated
        network would never discover a later update.
        """
        if adv.src == n.nid:
            return False
        if adv.version < n.version:
            return False
        # Same-version complete holders do not need that ADV.  Newer-version
        # ADVs remain useful even if the node is complete for the old version.
        if adv.version == n.version and self._is_complete_holder(n, adv.version):
            return False
        have = n.have_chunks.get(adv.version, set())
        for page in adv.pages:
            if any(c not in have for c in self.chunk_range_for_page(page)):
                return True
        return False

    # ---------- protocol RX callback from channel_deluge ----------
    def _handle_packet_rx(self, n: Node, p: Packet) -> None:
        """Handle successful Deluge packet receptions.

        Version 5 uses a local FCFS cycle: a useful ADV immediately causes the
        receiver to stop ADV scanning and schedule a local REQ to that ADV
        sender.  There is no global REQ phase and no global DATA phase.
        """
        if p.ptype == "ADV":
            pages = list(getattr(p, "adv_pages", None) or [])
            if not pages:
                return
            adv = AdvInfo(
                p.src,
                p.version,
                pages,
                p.data_chan if p.data_chan is not None else self.args.chan_data,
                p.start_s,
            )
            n.last_adv_by_sender[p.src] = adv
            if self._adv_is_useful_for_node(n, adv):
                # First useful ADV still makes the node a requester for this cycle,
                # but it no longer stops the node's ADV-listening window and no
                # longer cancels the node's own pending ADV.  This keeps Deluge
                # comparable to the main simulator setting where every device
                # wakes for the full ADV window each cycle, because an even newer
                # version may appear at any time.  Half-duplex TX/RX overlap is
                # still handled by the channel layer and subtracted from RX energy.
                if getattr(n, "deluge_role", "IDLE") == "IDLE":
                    n.deluge_role = "REQUESTER"
                    n.selected_adv = adv
                    self._schedule_local_req(n, adv, p.start_s + p.airtime_s)
        elif p.ptype == "REQ":
            if p.dst is None or n.nid != p.dst:
                return
            page = int(p.requested_page if p.requested_page is not None else -1)
            if page < 0:
                return
            reqs = p.requested_indices or p._requested_indices or []
            key = (p.dst, p.version, page)
            for idx in reqs:
                self.requests_by_sender_page[key].add(int(idx))
            self.requesters_by_sender_page[key].add(p.src)
        elif p.ptype == "DATA":
            if p.chunk_idx is None or p.version <= 0:
                return
            if p.version > n.version:
                n.version = p.version
            n.have_chunks[p.version].add(int(p.chunk_idx))
            page = self.page_for_chunk(int(p.chunk_idx))
            pp = self.page_packet_idx_for_chunk(int(p.chunk_idx))
            n.page_packets_have[(p.version, page)].add(pp)
            n.day_rx_chunks.add((p.version, int(p.chunk_idx)))
            if self.is_page_complete(n, p.version, page):
                if page not in n.pages_have[p.version]:
                    n.pages_have[p.version].add(page)
                    n.completed_pages_today.add(page)
            if len(n.have_chunks[p.version]) >= int(self.args.complete_need):
                n.updated = True
                n.is_gateway = True
                n.version = p.version
                self._refresh_completed_pages(n, p.version)

    def _missing_for_adv(self, n: Node, adv: AdvInfo) -> Optional[Tuple[int, List[int]]]:
        """Choose one page and missing chunk list from an ADV for a receiver."""
        for page in adv.pages:
            missing = [c for c in self.chunk_range_for_page(page) if c not in n.have_chunks.get(adv.version, set())]
            if missing:
                return int(page), missing
        return None

    def _schedule_local_req(self, n: Node, adv: AdvInfo, adv_rx_end_t: float) -> None:
        """Schedule an immediate/local REQ after a useful ADV reception."""
        if self.env is None:
            return
        args = self.args
        chosen = self._missing_for_adv(n, adv)
        if chosen is None:
            return
        page, missing = chosen
        plen = self._req_payload_len(len(missing))
        airt = self.radio.airtime_seconds_for_args(args, args.sf_adv, args.bw_hz, args.cr_denom, plen)
        logic = self.logical_airtime_s("REQ", airt)
        rmin, rmax = args.deluge_req_backoff_ms
        lo = min(rmin, rmax) / 1000.0
        hi = max(rmin, rmax) / 1000.0
        bo = self.node_rng[n.nid].uniform(max(0.0, lo), max(0.0, hi)) if (hi > 0.0 or lo > 0.0) else 0.0
        start = float(adv_rx_end_t) + float(args.rx_to_tx_s) + bo
        # The sender's local REQ window opens after its ADV TX ends. The REQ
        # must finish inside that local window; CSMA/CAD deferrals past this
        # deadline drop the REQ instead of transmitting too late.
        deadline = float(adv_rx_end_t) + float(args.deluge_req_window_s)
        latest_start = deadline - logic
        if start > latest_start:
            start = latest_start
        if start < adv_rx_end_t:
            start = adv_rx_end_t
        if start + logic > deadline + 1e-12:
            self.day_ctr["req_deadline_drops"] += 1
            return
        p = Packet(
            "REQ",
            src=n.nid,
            dst=adv.src,
            sf=args.sf_adv,
            bw_hz=args.bw_hz,
            cr_denom=args.cr_denom,
            start_s=start,
            airtime_s=airt,
            power_dbm=args.tx_dbm,
            chan=args.chan_adv,
            version=adv.version,
            requested_page=page,
            requested_indices=missing,
            _requested_indices=missing,
            phase="REQ",
        )
        p.phase_deadline_s = deadline
        self.tx_logical_packet(p, "req_used_s")

    def _on_packet_tx_started(self, p: Packet) -> None:
        """Callback from the channel layer when the actual post-CSMA TX starts."""
        if p.ptype != "ADV":
            return
        sender = self.by_id.get(p.src)
        if sender is None:
            return
        sender.deluge_role = "SENDER_CANDIDATE"
        req_start = float(p.start_s) + float(p.airtime_s)
        req_end = req_start + float(self.args.deluge_req_window_s)
        # Sender-local REQ RX window. It is tied to the actual ADV TX time, so
        # CSMA/CAD deferral does not make receivers transmit into a stale global
        # window.
        self._plan_channel(sender, req_start, req_end, self.args.chan_adv, "REQ")
        if p.src not in getattr(self, "sender_data_process_scheduled", set()):
            self.sender_data_process_scheduled.add(p.src)
            self.env.process(self._sender_data_after_req_window(p.src, req_end))

    def _sender_data_after_req_window(self, sender_id: int, req_end: float):
        """After a sender's local REQ window closes, send one requested page train."""
        yield self.env.timeout(max(0.0, float(req_end) - self.env.now))
        self._schedule_sender_data(sender_id, float(req_end) + float(self.args.deluge_round_gap_s))

    def _schedule_sender_data(self, sender_id: int, data_start: float) -> float:
        """Serve all requested pages sequentially within this sender's cycle budget.

        Each requester still sends at most one REQ after a useful ADV. The sender
        collects REQs during its existing local REQ window, then sends one DATA
        train per requested page, ordered by the number of requesters (ties: lower
        page index). There are no new ADV/REQ rounds or changes to per-cycle roles.
        """
        args = self.args
        sender = self.by_id.get(sender_id)
        if sender is None:
            return data_start

        candidates = []
        for (sid, version, page), requested in self.requests_by_sender_page.items():
            if sid != sender_id:
                continue
            requesters = self.requesters_by_sender_page.get((sid, version, page), set())
            if requested and requesters:
                candidates.append((len(requesters), -int(page), int(version), int(page),
                                   set(requested), set(requesters)))
        if not candidates:
            return data_start
        # Preserve the old one-page priority, then serve lower-priority pages
        # if they fit. Each (version, page) is included at most once per cycle.
        candidates.sort(reverse=True)

        data_airt = self.radio.airtime_seconds_for_args(
            args, args.sf_data, args.bw_hz, args.cr_denom, args.data_packet_bytes
        )
        data_logic = self.logical_airtime_s("DATA", data_airt)
        per_data = data_logic + max(0.0, args.deluge_data_gap_s)
        chan = sender.data_chan if sender.data_chan is not None else args.chan_data
        cursor = max(float(data_start), self.env.now)
        # tx_logical_packet schedules a *future* TX. Until that event executes,
        # sender.data_used_s does not include its airtime. Explicitly reserve
        # scheduled page airtime to prevent aggregate duty-cycle overbooking.
        reserved_data_tx_s = 0.0

        for _count, _neg_page, version, page, requested, requester_ids in candidates:
            if page not in sender.pages_have.get(version, set()):
                continue
            chunks = [c for c in self.chunk_range_for_page(page)
                      if c in sender.have_chunks.get(version, set()) and c in requested]
            if args.deluge_packet_order == "random":
                self.node_rng[sender_id].shuffle(chunks)
            else:
                chunks = sorted(chunks)
            if not chunks:
                continue

            train_tx_airtime_s = len(chunks) * data_logic
            remaining_budget_s = self._duty_data_remaining_s(sender) - reserved_data_tx_s
            if train_tx_airtime_s > remaining_budget_s + 1e-12:
                self.day_ctr["duty_data_page_skips"] += 1
                self.day_ctr["duty_data_packets_deferred"] += len(chunks)
                continue

            train_start = cursor
            train_end = train_start + len(chunks) * per_data
            if train_end > float(self.day_safety_end_t) + 1e-12:
                self.day_ctr["data_deadline_drops"] += len(chunks)
                continue

            # A successful REQ is required to follow this page's DATA train.
            # Receivers for a different page are not tuned to this train.
            for n in self.nodes:
                if n.nid == sender_id or n.nid not in requester_ids:
                    continue
                if any(c not in n.have_chunks.get(version, set()) for c in chunks):
                    self._plan_channel(n, train_start, train_end, chan, "DATA")

            for chunk in chunks:
                p = Packet(
                    "DATA", src=sender_id, dst=None, sf=args.sf_data,
                    bw_hz=args.bw_hz, cr_denom=args.cr_denom,
                    start_s=cursor, airtime_s=data_airt, power_dbm=args.tx_dbm,
                    chan=chan, version=version, chunk_idx=chunk,
                    page_idx=page, page_packet_idx=self.page_packet_idx_for_chunk(chunk),
                    phase="DATA", session_id=sender_id & 0xFFFF,
                )
                p.phase_deadline_s = self.day_safety_end_t
                self.tx_logical_packet(p, "data_used_s")
                cursor += per_data
            reserved_data_tx_s += train_tx_airtime_s
            self.day_ctr["duty_data_pages_scheduled"] += 1
            self.day_ctr["duty_data_packets_scheduled"] += len(chunks)
        return cursor

    # ---------- cycle logic ----------
    def _schedule_adv_phase(self, t0: float, adv_end: float) -> None:
        args = self.args
        airt = self.radio.airtime_seconds_for_args(args, args.sf_adv, args.bw_hz, args.cr_denom, args.adv_pl_bytes)
        logic = self.logical_airtime_s("ADV", airt)
        p_adv = self.density_adv_probability()
        for n in self.nodes:
            # All devices wake at cycle start and listen for ADV. This is a
            # dynamic interval because a useful ADV makes a requester stop ADV
            # scanning immediately and send a local REQ.
            n.adv_listen_idx = self._start_dynamic_listen(n, t0, adv_end, args.chan_adv, "ADV")
        for n in self.nodes:
            pages = self.useful_pages(n)
            if not pages:
                continue
            if self.node_rng[n.nid].random() > p_adv:
                continue
            bo = self.node_rng[n.nid].uniform(0.0, max(0.0, (adv_end - t0) - logic))
            start = t0 + bo
            p = Packet("ADV", src=n.nid, dst=None, sf=args.sf_adv, bw_hz=args.bw_hz, cr_denom=args.cr_denom,
                       start_s=start, airtime_s=airt, power_dbm=args.tx_dbm, chan=args.chan_adv,
                       version=n.version, data_chan=n.data_chan, adv_pages=pages, phase="ADV")
            # ADV must finish inside the ADV window. CSMA/CAD deferrals beyond
            # this deadline are dropped by the channel layer.
            p.phase_deadline_s = adv_end

            def adv_proc(node=n, pkt=p):
                yield self.env.timeout(max(0.0, float(pkt.start_s) - self.env.now))
                # Do not cancel scheduled ADV just because the node became a
                # requester earlier in the ADV window.  A node with completed
                # pages/full update remains an advertiser in every cycle.
                self.tx_logical_packet(pkt, "adv_used_s")
            self.env.process(adv_proc())

    def _choose_requests(self) -> List[Tuple[Node, AdvInfo, int, List[int]]]:
        out = []
        for n in self.nodes:
            candidates = sorted(n.last_adv_by_sender.values(), key=lambda a: a.t, reverse=True)
            chosen = None
            for adv in candidates:
                if adv.src == n.nid:
                    continue
                if adv.version < n.version:
                    continue
                for page in adv.pages:
                    missing = [c for c in self.chunk_range_for_page(page) if c not in n.have_chunks.get(adv.version, set())]
                    if missing:
                        chosen = (adv, page, missing)
                        break
                if chosen:
                    break
            if chosen:
                adv, page, missing = chosen
                out.append((n, adv, page, missing))
        return out

    def _schedule_req_phase(self, reqs: List[Tuple[Node, AdvInfo, int, List[int]]], req_start: float, req_end: float) -> None:
        args = self.args
        # Only potential senders that were actually selected by at least one
        # requester need to listen during the REQ collection window.  The older
        # version planned REQ RX for every node, which made rx_on_req_ms grow as
        # N * req_window for every cycle and heavily inflated Deluge energy.
        # This mirrors the main simulator's owner-local UPD listening: receivers
        # transmit their request, while advertised page holders listen for REQs.
        req_listener_ids = {int(adv.src) for _n, adv, _page, _missing in reqs}
        for nid in req_listener_ids:
            listener = self.by_id.get(nid)
            if listener is not None:
                self._plan_channel(listener, req_start, req_end, args.chan_adv, "REQ")
        rmin, rmax = args.deluge_req_backoff_ms
        for n, adv, page, missing in reqs:
            plen = self._req_payload_len(len(missing))
            airt = self.radio.airtime_seconds_for_args(args, args.sf_adv, args.bw_hz, args.cr_denom, plen)
            logic = self.logical_airtime_s("REQ", airt)
            bo = self.node_rng[n.nid].uniform(min(rmin, rmax), max(rmin, rmax)) / 1000.0
            start = req_start + min(max(0.0, bo), max(0.0, req_end - req_start - logic))
            p = Packet("REQ", src=n.nid, dst=adv.src, sf=args.sf_adv, bw_hz=args.bw_hz, cr_denom=args.cr_denom,
                       start_s=start, airtime_s=airt, power_dbm=args.tx_dbm, chan=args.chan_adv,
                       version=adv.version, requested_page=page, requested_indices=missing, _requested_indices=missing, phase="REQ")
            self.tx_logical_packet(p, "req_used_s")

    # ---------- duty-cycle helpers ----------
    def _duty_cycle_enabled(self) -> bool:
        return float(getattr(self.args, "duty_cycle_limit_s", 0.0) or 0.0) > 0.0

    def _duty_data_budget_s(self, sender: Node) -> float:
        """DATA-side TX airtime budget for this sender in the current cycle.

        Deluge has no REPAIR phase, but we keep the same configuration knob as
        the main simulator: --duty-cycle-data-fraction selects the share of the
        per-cycle duty-cycle limit that may be used for DATA-page trains.  If
        --duty-cycle-include-adv-in-data is enabled, ADV airtime spent by this
        sender in the current cycle is subtracted from the DATA-side budget.
        """
        limit = float(getattr(self.args, "duty_cycle_limit_s", 0.0) or 0.0)
        if limit <= 0.0:
            return float("inf")
        frac = min(1.0, max(0.0, float(getattr(self.args, "duty_cycle_data_fraction", 1.0) or 0.0)))
        budget = limit * frac
        if bool(getattr(self.args, "duty_cycle_include_adv_in_data", True)):
            start_adv = getattr(self, "day_start_adv_used_s", {}).get(sender.nid, sender.adv_used_s)
            adv_today = max(0.0, float(sender.adv_used_s) - float(start_adv))
            budget -= adv_today
        return max(0.0, budget)

    def _duty_data_remaining_s(self, sender: Node) -> float:
        """Remaining DATA TX airtime budget for this sender in the current cycle."""
        budget = self._duty_data_budget_s(sender)
        if math.isinf(float(budget)):
            return float("inf")
        start_data = getattr(self, "day_start_data_used_s", {}).get(sender.nid, sender.data_used_s)
        data_today = max(0.0, float(sender.data_used_s) - float(start_data))
        return max(0.0, float(budget) - data_today)

    def _schedule_data_phase(self, data_start: float) -> float:
        args = self.args
        data_airt = self.radio.airtime_seconds_for_args(args, args.sf_data, args.bw_hz, args.cr_denom, args.data_packet_bytes)
        data_logic = self.logical_airtime_s("DATA", data_airt)
        per_data = data_logic + max(0.0, args.deluge_data_gap_s)
        cursor = data_start
        for (sender_id, version, page), requested in sorted(self.requests_by_sender_page.items(), key=lambda kv: (kv[0][0], kv[0][2])):
            sender = self.by_id.get(sender_id)
            if sender is None or page not in sender.pages_have.get(version, set()):
                continue

            chunks = [c for c in self.chunk_range_for_page(page) if c in sender.have_chunks.get(version, set()) and c in requested]
            if args.deluge_packet_order == "random":
                chunks = list(chunks); self.node_rng[sender_id].shuffle(chunks)
            else:
                chunks = sorted(chunks)
            if not chunks:
                continue

            # Duty-cycle limiting is applied at Deluge page-train granularity, not
            # at individual chunk granularity.  A sender either transmits the full
            # requested set for this page in this cycle, or skips it and lets the
            # receivers request the missing page packets again in a later cycle.
            train_tx_airtime_s = len(chunks) * data_logic
            remaining_budget_s = self._duty_data_remaining_s(sender)
            if train_tx_airtime_s > remaining_budget_s + 1e-12:
                self.day_ctr["duty_data_page_skips"] += 1
                self.day_ctr["duty_data_packets_deferred"] += len(chunks)
                continue

            chan = sender.data_chan if sender.data_chan is not None else args.chan_data
            train_start = cursor
            train_end = train_start + len(chunks) * per_data
            # Only nodes whose REQ was successfully received by this sender follow
            # this DATA train. Other nodes may be in radio range, but Deluge receivers
            # must first hear an ADV and successfully request the page.
            requester_ids = set(self.requesters_by_sender_page.get((sender_id, version, page), set()))
            for n in self.nodes:
                if n.nid == sender_id or n.nid not in requester_ids:
                    continue
                if any(c not in n.have_chunks.get(version, set()) for c in chunks):
                    self._plan_channel(n, train_start, train_end, chan, "DATA")

            for chunk in chunks:
                p = Packet("DATA", src=sender_id, dst=None, sf=args.sf_data, bw_hz=args.bw_hz, cr_denom=args.cr_denom,
                           start_s=cursor, airtime_s=data_airt, power_dbm=args.tx_dbm, chan=chan, version=version,
                           chunk_idx=chunk, session_id=sender_id & 0xFFFF, page_idx=page, page_packet_idx=self.page_packet_idx_for_chunk(chunk), phase="DATA")
                self.tx_logical_packet(p, "data_used_s")
                cursor += per_data
            self.day_ctr["duty_data_pages_scheduled"] += 1
            self.day_ctr["duty_data_packets_scheduled"] += len(chunks)
        return cursor

    def _snapshot_stats(self):
        return {n.nid: (n.tx_adv,n.tx_req,n.tx_data,n.tx_upd,n.tx_miss,n.tx_end,n.tx_repair,
                        n.rx_on_s,n.rx_on_adv_s,n.rx_on_req_s,n.rx_on_data_s,n.rx_on_upd_s,n.rx_on_miss_s,n.rx_on_repair_s,
                        n.rx_busy_s,n.csma_cca_s,n.cad_scan_s,n.csma_sense_s) for n in self.nodes}

    def _append_day_stats(self, start_stats):
        for n in self.nodes:
            by_mode = self._compute_rx_on_by_mode_for_day(n)
            adv = by_mode.get("ADV",0.0); req = by_mode.get("REQ",0.0); data = by_mode.get("DATA",0.0)
            ordinary = adv + req + data + by_mode.get("UNKNOWN",0.0)
            total_rx_on = ordinary + (n.csma_cca_s - start_stats[n.nid][15]) + (n.cad_scan_s - start_stats[n.nid][16])
            n.rx_on_s += total_rx_on
            n.rx_on_adv_s += adv; n.rx_on_req_s += req; n.rx_on_data_s += data
            st = start_stats[n.nid]
            n.day_tx_adv.append(n.tx_adv-st[0]); n.day_tx_req.append(n.tx_req-st[1]); n.day_tx_data.append(n.tx_data-st[2])
            n.day_tx_upd.append(n.tx_upd-st[3]); n.day_tx_miss.append(n.tx_miss-st[4]); n.day_tx_end.append(n.tx_end-st[5]); n.day_tx_repair.append(n.tx_repair-st[6])
            n.day_rx_on_ms.append(round(total_rx_on*1000,3)); n.day_rx_on_adv_ms.append(round(adv*1000,3)); n.day_rx_on_req_ms.append(round(req*1000,3)); n.day_rx_on_data_ms.append(round(data*1000,3))
            n.day_rx_on_upd_ms.append(0.0); n.day_rx_on_miss_ms.append(0.0); n.day_rx_on_repair_ms.append(0.0)
            n.day_rx_busy_ms.append(round((n.rx_busy_s-st[14])*1000,3)); n.day_csma_cca_ms.append(round((n.csma_cca_s-st[15])*1000,3)); n.day_cad_scan_ms.append(round((n.cad_scan_s-st[16])*1000,3)); n.day_csma_sense_ms.append(round((n.csma_sense_s-st[17])*1000,3))

    def clear_radio_states(self):
        super().clear_radio_states()
        for n in self.nodes:
            n.last_adv_by_sender.clear()
            n.completed_pages_today.clear()
            n.adv_listen_idx = None
            n.deluge_role = "IDLE"
            n.selected_adv = None

    def save_day_snapshot(self, day_idx: int):
        """Save one Deluge dissemination snapshot as a PNG frame.

        Visual states are Deluge-specific:
          - idle: no packets for the current version
          - partial object: has at least one chunk but no complete page yet
          - page relay: has one or more completed pages but not the full object
          - fully updated: has the complete object
          - initial source: original/external gateway source
        """
        if plt is None:
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        version = int(getattr(self.args, "version_start", 1) or 1)
        total_chunks = max(1, int(getattr(self.args, "complete_need", 1) or 1))
        page_total = max(1, self.page_count())

        xs_idle, ys_idle = [], []
        xs_partial, ys_partial = [], []
        xs_page, ys_page, page_vals = [], [], []
        xs_updated, ys_updated = [], []
        xs_source, ys_source = [], []

        for n in self.nodes:
            have = len(n.have_chunks.get(version, set()))
            completed_pages = len(n.pages_have.get(version, set()))
            is_full = self._is_complete_holder(n, version)

            # Nodes that started as full sources have is_gateway=True from initialization.
            # After Deluge completes, receivers are also set is_gateway=True so they can relay;
            # keep true full sources visually distinct only when they had the full object from the start.
            if getattr(n, "initial_source", False):
                xs_source.append(n.x)
                ys_source.append(n.y)
            elif is_full or n.updated:
                xs_updated.append(n.x)
                ys_updated.append(n.y)
            elif completed_pages > 0:
                xs_page.append(n.x)
                ys_page.append(n.y)
                page_vals.append(min(page_total, completed_pages))
            elif have > 0:
                xs_partial.append(n.x)
                ys_partial.append(n.y)
            else:
                xs_idle.append(n.x)
                ys_idle.append(n.y)

        legend_handles = []
        try:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            from matplotlib.colors import LinearSegmentedColormap, Normalize

            if xs_idle:
                ax.scatter(xs_idle, ys_idle, s=12, c="#1f77b4", label="idle")
            if xs_partial:
                ax.scatter(xs_partial, ys_partial, s=18, c="#f39c12", label="partial object")
            if xs_updated:
                ax.scatter(xs_updated, ys_updated, s=28, c="#2ecc71", label="fully updated")
            if xs_source:
                ax.scatter(xs_source, ys_source, s=85, c="#2ecc71", marker="*", edgecolors="black", linewidths=0.9, zorder=6, label="initial source")

            legend_handles.extend([
                Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#1f77b4", markeredgecolor="none", markersize=6, label="idle"),
                Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#f39c12", markeredgecolor="none", markersize=7, label="partial object"),
                Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#2ecc71", markeredgecolor="none", markersize=7, label="fully updated"),
                Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#2ecc71", markeredgecolor="black", markersize=11, label="initial source"),
            ])

            if xs_page:
                cmap = LinearSegmentedColormap.from_list("deluge_pages", ["#f39c12", "#8e44ad"], N=256)
                norm = Normalize(vmin=1, vmax=max(1, page_total))
                ax.scatter(xs_page, ys_page, s=44, c=[cmap(norm(v)) for v in page_vals], edgecolors="black", linewidths=0.6, zorder=5)
                for v, lab in [(1, "page relay: 1 page"), (max(1, page_total // 2), "page relay: mid"), (page_total, "page relay: max")]:
                    legend_handles.append(Patch(facecolor=cmap(norm(v)), edgecolor="black", label=lab))
        except Exception:
            if xs_idle:
                ax.scatter(xs_idle, ys_idle, s=12, label="idle")
            if xs_partial:
                ax.scatter(xs_partial, ys_partial, s=18, label="partial object")
            if xs_page:
                ax.scatter(xs_page, ys_page, s=44, label="page relay")
            if xs_updated:
                ax.scatter(xs_updated, ys_updated, s=28, label="fully updated")
            if xs_source:
                ax.scatter(xs_source, ys_source, s=85, marker="*", label="initial source")

        total = len(self.nodes)
        fully_updated = sum(1 for n in self.nodes if self._is_complete_holder(n, version) or n.updated)
        page_relays = sum(1 for n in self.nodes if (not self._is_complete_holder(n, version)) and len(n.pages_have.get(version, set())) > 0)
        partial = sum(1 for n in self.nodes if (not self._is_complete_holder(n, version)) and len(n.pages_have.get(version, set())) == 0 and len(n.have_chunks.get(version, set())) > 0)

        ax.set_title(f"Cycle {day_idx + 1}: full={fully_updated}/{total}, page_relays={page_relays}, partial={partial}")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)
        else:
            ax.legend(loc="upper right", fontsize=8)

        out = os.path.join(self.out_dir, f"cycle_{day_idx + 1:03d}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def run_day(self):
        args = self.args
        self.env = simpy.Environment()
        self.active_packets.clear(); self.tx_history.clear(); self.clear_radio_states()
        self.day_ctr = defaultdict(int); self.requests_by_sender_page.clear(); self.requesters_by_sender_page.clear()
        self.sender_data_process_scheduled = set()
        start_stats = self._snapshot_stats()
        self.day_start_adv_used_s = {n.nid: float(n.adv_used_s) for n in self.nodes}
        self.day_start_data_used_s = {n.nid: float(n.data_used_s) for n in self.nodes}
        # Choose sender data channels. Reserve chan_adv as control if possible.
        control_chan = args.chan_adv
        if args.num_channels >= 2:
            data_channels = [c for c in range(args.num_channels) if c != control_chan]
            for n in self.nodes:
                n.data_chan = self.node_rng[n.nid].choice(data_channels)
        else:
            for n in self.nodes:
                n.data_chan = args.chan_data
        t0 = 0.0
        adv_end = t0 + args.deluge_adv_window_s
        # Local REQ/DATA interactions are scheduled from actual ADV TX times.
        # The day safety cap bounds CSMA/CAD deferrals and DATA trains so this
        # remains a discrete-cycle model, not continuous Deluge.
        self.day_safety_end_t = adv_end + args.deluge_req_window_s + args.deluge_round_gap_s + max(args.data_budget_s, 1.0) + 1.0
        self._schedule_adv_phase(t0, adv_end)
        self.env.run(until=self.day_safety_end_t)
        self._append_day_stats(start_stats)
        if getattr(args, "animate", False):
            self.save_day_snapshot(self.day_idx)
        if getattr(args, "debug", False):
            updated = sum(1 for n in self.nodes if n.updated)
            print(f"[DEBUG] Cycle {self.day_idx+1}: updated={updated}/{len(self.nodes)} tx_adv={self.day_ctr['tx_adv']} tx_req={self.day_ctr['tx_req']} tx_data={self.day_ctr['tx_data']} rx_adv={self.day_ctr['rx_adv']} rx_req={self.day_ctr['rx_req']} rx_data={self.day_ctr['rx_data']} duty_page_skips={self.day_ctr['duty_data_page_skips']} duty_packets_deferred={self.day_ctr['duty_data_packets_deferred']} csma_drops={self.day_ctr['csma_drops']}")

    def run(self, testbed_path: str):
        self.args.deluge_page_count = self.page_count()
        self.apply_auto_control_windows()
        for d in range(int(self.args.max_days)):
            self.day_idx = d
            self.run_day()
            if self.args.until_complete and all(n.updated for n in self.nodes):
                break
        self.write_metadata(testbed_path)
        self.write_node_summary()
        self.write_node_state()


def main(argv=None):
    global RADIO_PROFILE
    args = parse_args(argv)
    if getattr(args, "animate", False) and plt is None:
        print("[WARN] --animate requested but matplotlib not available; disabling.")
        args.animate = False
    if args.seed is None:
        args.seed = 1
    rng = random.Random(args.seed)
    RADIO_PROFILE = get_radio_profile(args)
    if getattr(args, "model_size_bytes", 0) and args.model_size_bytes > 0:
        total_chunks = math.ceil(args.model_size_bytes / args.data_pl_bytes)
        args.complete_need = total_chunks
        args.data_chunks_per_round = total_chunks
    nodes = load_testbed(args.testbed)
    sim = Sim(args, nodes, rng)
    sim.run(args.testbed)


if __name__ == "__main__":
    main()
