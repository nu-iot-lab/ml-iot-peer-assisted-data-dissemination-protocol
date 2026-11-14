# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Set, Tuple

from models_deluge import Node, Packet


class ChannelMixin:
    # ---------- tuning helpers ----------
    def _plan_channel(self, n: Node, start: float, end: float, chan: int, mode: str = "UNKNOWN"):
        n.tune_plan.append((float(start), float(end), int(chan), str(mode).upper()))

    def _channel_at_time(self, n: Node, t: float) -> Optional[int]:
        chosen = None
        latest_start = -1.0
        # Static/planned listening windows.
        for win in n.tune_plan:
            # Backward-compatible support for old 3-tuples: (start, end, chan)
            s, e, c = float(win[0]), float(win[1]), int(win[2])
            if s <= t < e and s >= latest_start:
                chosen = c
                latest_start = s
        # Dynamic owner-local DATA/REPAIR/MISS windows. These may be shortened at runtime.
        for win in getattr(n, "dynamic_tune_plan", []):
            try:
                s, e, c = float(win[0]), float(win[1]), int(win[2])
            except Exception:
                continue
            if s <= t < e and s >= latest_start:
                chosen = c
                latest_start = s
        return chosen

    def _start_dynamic_listen(self, n: Node, start: float, end: float, chan: int, mode: str = "UNKNOWN") -> int:
        """Start a mutable RX/listening interval and return its index."""
        if end <= start:
            end = start
        n.dynamic_tune_plan.append([float(start), float(end), int(chan), str(mode).upper()])
        return len(n.dynamic_tune_plan) - 1

    def _stop_dynamic_listen(self, n: Node, idx: Optional[int], stop_t: Optional[float] = None) -> None:
        """Shorten a dynamic RX/listening interval."""
        if idx is None:
            return
        if idx < 0 or idx >= len(getattr(n, "dynamic_tune_plan", [])):
            return
        win = n.dynamic_tune_plan[idx]
        t = self.env.now if stop_t is None and self.env is not None else float(stop_t if stop_t is not None else win[1])
        # Clamp to [start, current_end].
        win[1] = max(float(win[0]), min(float(win[1]), t))

    def clear_radio_states(self):
        for n in self.nodes:
            n.tx_busy_until = 0.0
            n.rx_busy_until = 0.0
            n.day_tx_intervals.clear()
            n.tune_plan.clear()
            n.dynamic_tune_plan.clear()
            n.data_listen_idx = None
            n.repair_listen_idx = None
            n.owner_miss_listen_idx = None
            n.data_timeout_token = 0
            n.repair_timeout_token = 0
            n.data_miss_scheduled = False
            n.repair_done_today = False
            n.repair_listen_chan = None
            n.day_rx_chunks.clear()
            n.seen_logical_packets.clear()
            n.last_day_partial = False
            n.owner_for_day = None
            n.listen_chan_data = None
            n.data_chan = None
            n.end_rx_counts.clear()
            n.last_chunk_idx.clear()
            n.version_changed_today = False

    def distance(self, a: Node, b: Node) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _rx_power_for_packet_at_node(self, p: Packet, n: Node) -> float:
        """Return cached received power for a physical packet at a receiver.

        Path loss, lognormal shadowing, and small-scale fading are sampled once
        for each physical packet / receiver pair. The same value is then reused
        for sensitivity and capture/collision decisions.
        """
        cache = getattr(p, "_rx_power_by_nid", None)
        if cache is None:
            p._rx_power_by_nid = {}
            cache = p._rx_power_by_nid
        if n.nid in cache:
            return cache[n.nid]
        src_node = self.by_id[p.src]
        d = self.distance(src_node, n)
        rx_dbm = self.radio.rx_power_dbm(self.args, self.rng, d, p.power_dbm)
        cache[n.nid] = rx_dbm
        return rx_dbm

    # ---------- simplified CSMA/CA-like channel access helpers ----------
    def _csma_stage_set(self) -> Set[str]:
        return {str(x).upper() for x in getattr(self.args, "csma_stages", [])}

    def _csma_enabled_for_packet(self, p: Packet) -> bool:
        if not bool(getattr(self.args, "csma_ca", False)):
            return False
        return str(p.ptype).upper() in self._csma_stage_set()

    def _csma_channel_busy_for_node(self, src: Node, chan: int, t: float) -> bool:
        """Carrier-sense decision at node src on channel chan at time t.

        The channel is busy if any currently active same-channel packet is
        detectable at src above --csma-carrier-sense-thresh-dbm. Hidden nodes
        below the threshold are not sensed and may still collide later, which is
        the intended simplified CSMA/CA behavior.
        """
        thresh = float(getattr(self.args, "csma_carrier_sense_thresh_dbm", -85.0))
        for q in list(self.active_packets):
            if q.src == src.nid:
                continue
            if int(q.chan) != int(chan):
                continue
            q_end = q.start_s + q.airtime_s
            if not (q.start_s <= t < q_end):
                continue
            rx_dbm = self._rx_power_for_packet_at_node(q, src)
            if rx_dbm >= thresh:
                return True
        return False

    def _scaled_busy_defer_s(self, p: Packet, factor: float, jitter: float, max_factor: float = 2.0) -> float:
        """Return busy-channel defer time with consecutive-busy scaling.

        Consecutive busy detections double the initial airtime factor until
        max_factor is reached. For generic CSMA, defaults are 0.5x, 1x,
        2x, then 2x... packet airtime. A max deferral count of 0 means
        keep deferring until the channel becomes free, subject only to the
        simulator day safety deadline.
        """
        initial = max(0.0, float(factor or 0.0))
        max_factor = max(0.0, float(max_factor or 0.0))
        consecutive = max(0, int(getattr(p, "csma_consecutive_busy", 0) or 0))
        uncapped = initial * (2.0 ** consecutive)
        scaled_factor = min(max_factor, uncapped) if max_factor > 0.0 else uncapped
        base = scaled_factor * float(p.airtime_s)
        jitter = max(0.0, float(jitter or 0.0))
        if jitter > 0.0:
            lo = max(0.0, 1.0 - jitter)
            hi = 1.0 + jitter
            base *= self.node_rng[p.src].uniform(lo, hi)
        return max(0.0, base)

    def _csma_defer_s(self, p: Packet) -> float:
        return self._scaled_busy_defer_s(
            p,
            getattr(self.args, "csma_busy_wait_airtime_factor", 0.5),
            getattr(self.args, "csma_busy_wait_jitter", 0.0),
            getattr(self.args, "csma_busy_wait_airtime_max_factor", 2.0),
        )

    def _cad_defer_s(self, p: Packet) -> float:
        return self._scaled_busy_defer_s(
            p,
            getattr(self.args, "cad_busy_wait_airtime_factor", 1.0),
            getattr(self.args, "cad_busy_wait_jitter", 0.0),
            getattr(self.args, "cad_busy_wait_airtime_max_factor", 4.0),
        )

    def _cad_stage_set(self) -> Set[str]:
        return {str(x).upper() for x in getattr(self.args, "cad_csma_stages", [])}

    def _cad_enabled_for_packet(self, p: Packet) -> bool:
        if not bool(getattr(self.args, "cad_csma", False)):
            return False
        return str(p.ptype).upper() in self._cad_stage_set()

    def _cad_duration_s(self, p: Packet) -> float:
        return float(self.radio.cad_duration_seconds_for_args(self.args, p.sf, p.bw_hz))

    def _cad_channel_busy_for_node(self, src: Node, p: Packet, t: float) -> bool:
        """LoRa CAD-style busy check at node src for packet p.

        CAD is modeled as SF-selective: only overlapping same-channel, same-SF
        LoRa transmissions are considered. Detection uses the selected radio
        sensitivity plus --cad-detect-margin-db rather than the RSSI/CCA threshold.
        """
        margin = float(getattr(self.args, "cad_detect_margin_db", 0.0) or 0.0)
        thresh = self.radio.sensitivity_dbm_for_args(self.args, p.sf) + margin
        for q in list(self.active_packets):
            if q.src == src.nid:
                continue
            if int(q.chan) != int(p.chan):
                continue
            if int(q.sf) != int(p.sf):
                continue
            q_end = q.start_s + q.airtime_s
            if not (q.start_s <= t < q_end):
                continue
            rx_dbm = self._rx_power_for_packet_at_node(q, src)
            if rx_dbm >= thresh:
                return True
        return False

    def _csma_deadline_for_packet(self, p: Packet) -> Optional[float]:
        """Return the latest allowed packet end time for CSMA/CAD deferrals.

        Deluge v5 is local/event-driven, so there is no global data_start_t or
        data_end_t. Packets may carry a per-packet phase_deadline_s:
          - ADV: end before the ADV window closes
          - REQ: end before the sender-local REQ window closes
          - DATA: end before the cycle safety deadline / duty budget deadline
        Fall back only to day_safety_end_t.
        """
        phase_deadline = getattr(p, "phase_deadline_s", None)
        if phase_deadline is not None:
            return float(phase_deadline)
        return getattr(self, "day_safety_end_t", None)

    def can_rx(self, n: Node, t: float, pchan: int) -> bool:
        tuned = self._channel_at_time(n, t)
        return (t >= n.rx_busy_until) and (t >= n.tx_busy_until) and (tuned == pchan)

    def mark_tx(self, n: Node, t_start: float, duration: float):
        n.tx_busy_until = max(n.tx_busy_until, t_start + duration)
        n.rx_busy_until = max(n.rx_busy_until, t_start + duration)
        n.day_tx_intervals.append((t_start, t_start + duration))

    def mark_rx(self, n: Node, t_start: float, duration: float):
        """Mark receiver as busy and accumulate RX *busy* time.

        Busy time counts active reception attempts (including collided ones). This is
        different from RX-on/listening time, which we compute from tune windows.
        We accumulate only the *incremental* busy time beyond the current rx_busy_until
        to avoid double-counting overlapping RX intervals.
        """
        end = t_start + duration
        prev_end = n.rx_busy_until
        if end > prev_end:
            inc = end - max(prev_end, t_start)
            if inc > 0.0:
                n.rx_busy_s += inc
        n.rx_busy_until = max(prev_end, end)

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged: List[Tuple[float, float]] = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    @staticmethod
    def _interval_total(intervals: List[Tuple[float, float]]) -> float:
        return sum(max(0.0, e - s) for s, e in intervals)

    @staticmethod
    def _subtract_intervals(
        base: List[Tuple[float, float]],
        sub: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Return base intervals minus sub intervals for merged, non-overlapping interval lists."""
        if not base:
            return []
        if not sub:
            return base
        out: List[Tuple[float, float]] = []
        j = 0
        for s, e in base:
            cur = s
            while j < len(sub) and sub[j][1] <= s:
                j += 1
            k = j
            while k < len(sub) and sub[k][0] < e:
                ss, ee = sub[k]
                if ss > cur:
                    out.append((cur, min(ss, e)))
                cur = max(cur, ee)
                if cur >= e:
                    break
                k += 1
            if cur < e:
                out.append((cur, e))
        return out

    def _normalize_rx_mode(self, mode: object) -> str:
        m = str(mode or "UNKNOWN").strip().upper()
        if m in {"ADV", "REQ", "UPD", "DATA", "MISS", "REPAIR"}:
            return m
        return "UNKNOWN"

    def _compute_rx_on_by_mode_for_day(self, n: Node) -> Dict[str, float]:
        """Compute RX-on listening time split by protocol mode, minus TX overlap.

        If listening windows overlap, the same precedence as _channel_at_time is used:
        the active window with the latest start time owns the interval. This prevents
        double-counting while preserving the simulator's tuning behavior.
        """
        windows: List[Tuple[float, float, str]] = []
        for win in getattr(n, "tune_plan", []):
            try:
                s, e = float(win[0]), float(win[1])
                mode = self._normalize_rx_mode(win[3] if len(win) >= 4 else "UNKNOWN")
            except Exception:
                continue
            if e > s:
                windows.append((s, e, mode))

        for win in getattr(n, "dynamic_tune_plan", []):
            try:
                s, e = float(win[0]), float(win[1])
                mode = self._normalize_rx_mode(win[3] if len(win) >= 4 else "UNKNOWN")
            except Exception:
                continue
            if e > s:
                windows.append((s, e, mode))

        out: Dict[str, float] = {"ADV": 0.0, "REQ": 0.0, "UPD": 0.0, "DATA": 0.0, "MISS": 0.0, "REPAIR": 0.0, "UNKNOWN": 0.0}
        if not windows:
            return out

        tx_windows = self._merge_intervals([(s, e) for (s, e) in n.day_tx_intervals if e > s])
        boundaries = set()
        for s, e, _m in windows:
            boundaries.add(s); boundaries.add(e)
            for ts, te in tx_windows:
                if s < ts < e:
                    boundaries.add(ts)
                if s < te < e:
                    boundaries.add(te)
        points = sorted(boundaries)
        if len(points) < 2:
            return out

        def in_tx(mid: float) -> bool:
            return any(ts <= mid < te for ts, te in tx_windows)

        for a, b in zip(points[:-1], points[1:]):
            if b <= a:
                continue
            mid = (a + b) / 2.0
            if in_tx(mid):
                continue
            active = [(s, mode) for s, e, mode in windows if s <= mid < e]
            if not active:
                continue
            _latest_start, mode = max(active, key=lambda x: x[0])
            out[self._normalize_rx_mode(mode)] += b - a
        return out

    def _compute_rx_on_for_day(self, n: Node) -> float:
        """Backward-compatible total RX-on listening time for the day."""
        by_mode = self._compute_rx_on_by_mode_for_day(n)
        return sum(by_mode.values())

    # ---------- physical repetition helpers ----------
    def _repeat_stage_set(self) -> Set[str]:
        return {str(x).upper() for x in getattr(self.args, "phy_repeat_stages", [])}

    def _repetitions_for_ptype(self, ptype: str) -> int:
        reps = int(getattr(self.args, "phy_repetitions", 1) or 1)
        if reps <= 1:
            return 1
        if str(ptype).upper() not in self._repeat_stage_set():
            return 1
        return reps

    def _repeat_gap_max_s(self) -> float:
        """Conservative maximum gap used for budget/window calculations."""
        gap = float(getattr(self.args, "phy_repeat_gap_ms", 0.0) or 0.0) / 1000.0
        jitter = float(getattr(self.args, "phy_repeat_jitter_ms", 0.0) or 0.0) / 1000.0
        return max(0.0, gap + jitter)

    def logical_airtime_s(self, ptype: str, single_airtime_s: float) -> float:
        """Duration occupied by one logical packet including physical repetitions.

        The physical copies are sent sequentially. If repetitions are disabled for
        the packet type, this returns the original single-copy airtime.
        """
        reps = self._repetitions_for_ptype(ptype)
        if reps <= 1:
            return single_airtime_s
        return reps * single_airtime_s + (reps - 1) * self._repeat_gap_max_s()

    def _repeat_channel_count(self, reps: int) -> int:
        configured = int(getattr(self.args, "phy_repeat_channel_count", 0) or 0)
        if configured <= 0:
            return min(max(1, reps), max(1, int(getattr(self.args, "num_channels", 1) or 1)))
        return min(configured, max(1, int(getattr(self.args, "num_channels", 1) or 1)))

    def _repeat_channel(self, base_chan: int, rep_idx: int, rep_count: int) -> int:
        """Return channel for a physical repetition, with wraparound.

        Example with num_channels=37, base_chan=35, rep_count=3:
        rep 0 -> 35, rep 1 -> 36, rep 2 -> 0.
        """
        nchan = max(1, int(getattr(self.args, "num_channels", 1) or 1))
        if getattr(self.args, "phy_repeat_mode", "same_channel") != "channel_hop":
            return int(base_chan) % nchan
        hop_count = self._repeat_channel_count(rep_count)
        return (int(base_chan) + (int(rep_idx) % hop_count)) % nchan

    def _repeat_channel_sequence(self, base_chan: int, rep_count: int) -> Set[int]:
        return {self._repeat_channel(base_chan, i, rep_count) for i in range(max(1, rep_count))}

    def _can_rx_packet(self, n: Node, p: Packet) -> bool:
        if not (p.start_s >= n.rx_busy_until and p.start_s >= n.tx_busy_until):
            return False
        tuned = self._channel_at_time(n, p.start_s)
        if tuned == p.chan:
            return True

        # BLE Mesh-like channel hopping approximation: a receiver tuned to the
        # logical/base channel is allowed to follow the physical copies across
        # consecutive channels in the same advertising-like event. This keeps the
        # existing tune_plan abstraction while making hopped repetitions useful.
        if (
            getattr(self.args, "phy_repeat_mode", "same_channel") == "channel_hop"
            and bool(getattr(self.args, "phy_repeat_rx_scan", True))
            and int(getattr(p, "rep_count", 1) or 1) > 1
        ):
            base_chan = p.base_chan if p.base_chan is not None else p.chan
            return tuned == base_chan and p.chan in self._repeat_channel_sequence(base_chan, p.rep_count)
        return False

    def _logical_packet_key(self, p: Packet) -> Tuple:
        if p.logical_id is not None:
            return (self.day_idx, p.logical_id)
        # END1/END2/END3 are three different logical END packets even though
        # ptype remains "END" for backwards-compatible counters/logs. Include
        # phase/end_seq so DATA ENDs and REPAIR ENDs do not suppress each other,
        # and so END2/END3 are not incorrectly marked as duplicates of END1.
        return (
            self.day_idx,
            p.src,
            p.ptype,
            getattr(p, "phase", None),
            int(getattr(p, "end_seq", 0) or 0),
            p.version,
            p.chunk_idx,
            p.dst,
        )

    def tx_logical_packet(self, p: Packet, field_name: str):
        """Schedule a logical packet, expanding it into physical repetitions if enabled."""
        reps = self._repetitions_for_ptype(p.ptype)
        if reps <= 1:
            p.rep_idx = 0
            p.rep_count = 1
            p.base_chan = p.chan
            self.tx_packet(p, field_name)
            return

        logical_id = getattr(self, "next_logical_packet_id", 0)
        self.next_logical_packet_id = logical_id + 1
        base_start = float(p.start_s)
        base_chan = int(p.chan)
        gap_fixed_s = float(getattr(self.args, "phy_repeat_gap_ms", 0.0) or 0.0) / 1000.0
        jitter_max_s = float(getattr(self.args, "phy_repeat_jitter_ms", 0.0) or 0.0) / 1000.0

        cursor = base_start
        for rep_idx in range(reps):
            p_rep = copy.copy(p)
            # Shallow copy would otherwise share this cache across repeated copies.
            p_rep._rx_power_by_nid = {}
            p_rep._rx_candidates = None
            p_rep.logical_id = logical_id
            p_rep.rep_idx = rep_idx
            p_rep.rep_count = reps
            p_rep.base_chan = base_chan
            p_rep.start_s = cursor
            p_rep.chan = self._repeat_channel(base_chan, rep_idx, reps)
            self.tx_packet(p_rep, field_name)

            if rep_idx < reps - 1:
                jitter = self.node_rng[p.src].uniform(0.0, jitter_max_s) if jitter_max_s > 0.0 else 0.0
                cursor += p.airtime_s + gap_fixed_s + jitter

    def tx_packet(self, p: Packet, field_name: str):
        """Schedule a single physical transmission in SimPy.

        When --cad-csma is enabled for the packet type, LoRa CAD-style sensing is
        performed before TX. Only CAD scan duration is charged as RX-like sensing
        time; the later defer/backoff sleep is not charged as RX time.

        When --csma-ca is enabled for the packet type, the packet waits until its
        scheduled start, senses the channel, defers by an airtime-based interval
        while the channel is busy, then transmits at the final start time. Existing
        collision/capture logic is still used after transmission starts, so hidden
        node collisions remain possible.
        """
        env = self.env

        def _finish_tx_bookkeeping(src: Node):
            self.day_ctr[f"tx_{p.ptype.lower()}"] += 1
            src.tx_ok += 1
            if p.ptype == 'ADV':
                src.tx_adv += 1
            elif p.ptype == 'REQ':
                src.tx_req += 1
            elif p.ptype == 'UPD':
                src.tx_upd += 1
            elif p.ptype == 'DATA':
                src.tx_data += 1
            elif p.ptype == 'MISS':
                src.tx_miss += 1
            elif p.ptype == 'END':
                src.tx_end += 1
            elif p.ptype == 'REPAIR':
                src.tx_repair += 1
            if hasattr(src, field_name):
                setattr(src, field_name, getattr(src, field_name) + p.airtime_s)

        def tx_proc():
            src = self.by_id[p.src]
            cad_enabled = self._cad_enabled_for_packet(p)
            csma_enabled = self._csma_enabled_for_packet(p)
            original_start = float(p.start_s)

            if not cad_enabled and not csma_enabled:
                # Legacy behavior: reserve the future TX interval immediately.
                self.mark_tx(src, p.start_s, p.airtime_s)
                yield env.timeout(max(0.0, p.start_s - env.now))
            elif cad_enabled:
                # LoRa CAD-CSMA behavior. CAD itself is RX-like sensing time; the
                # later busy-channel defer is modeled as backoff/sleep and is not
                # added to rx_on_s/csma_sense_s.
                yield env.timeout(max(0.0, p.start_s - env.now))
                max_deferrals = int(getattr(self.args, "cad_max_deferrals", 0) or 0)
                deadline = self._csma_deadline_for_packet(p)

                while True:
                    local_busy_until = max(float(getattr(src, "tx_busy_until", 0.0)), float(getattr(src, "rx_busy_until", 0.0)))
                    if env.now < local_busy_until:
                        yield env.timeout(max(0.0, local_busy_until - env.now))

                    cad_s = max(0.0, self._cad_duration_s(p))
                    if deadline is not None and env.now + cad_s + p.airtime_s > float(deadline):
                        p.csma_dropped = True
                        self.day_ctr["csma_drops"] += 1
                        return

                    # Sense at the start of the CAD interval. Reserve the local
                    # radio for CAD so same-node transmissions do not overlap it.
                    busy = self._cad_channel_busy_for_node(src, p, env.now)
                    if cad_s > 0.0:
                        src.cad_scan_s += cad_s
                        src.csma_sense_s += cad_s  # backward-compatible aggregate
                        src.rx_busy_until = max(src.rx_busy_until, env.now + cad_s)
                        yield env.timeout(cad_s)

                    if not busy:
                        p.csma_consecutive_busy = 0
                        break

                    if max_deferrals > 0 and p.csma_deferrals >= max_deferrals:
                        p.csma_dropped = True
                        self.day_ctr["csma_drops"] += 1
                        return

                    p.csma_deferrals += 1
                    self.day_ctr["csma_deferrals"] += 1
                    wait_s = self._cad_defer_s(p)
                    p.csma_consecutive_busy += 1
                    if wait_s <= 0.0:
                        wait_s = 1e-9
                    yield env.timeout(wait_s)

                p.start_s = float(env.now)
                p.csma_delay_s = max(0.0, p.start_s - original_start)
                self.mark_tx(src, p.start_s, p.airtime_s)
            else:
                # Generic CSMA/CCA behavior. Busy-channel defer is treated as
                # RX/CCA listening time for energy accounting. By default the
                # packet keeps deferring until the channel becomes free; waits
                # scale as 1x, 2x, then 4x packet airtime and stay capped there.
                yield env.timeout(max(0.0, p.start_s - env.now))
                max_deferrals = int(getattr(self.args, "csma_max_deferrals", 0) or 0)
                deadline = self._csma_deadline_for_packet(p)

                while True:
                    # Respect half-duplex radio state of the transmitter.
                    local_busy_until = max(float(getattr(src, "tx_busy_until", 0.0)), float(getattr(src, "rx_busy_until", 0.0)))
                    if env.now < local_busy_until:
                        yield env.timeout(max(0.0, local_busy_until - env.now))

                    # Day safety cap. Dynamic DATA/REPAIR receivers time out, but
                    # the day itself must not grow forever. If a deferred packet no
                    # longer fits in the day, drop this physical attempt.
                    if deadline is not None and env.now + p.airtime_s > float(deadline):
                        p.csma_dropped = True
                        self.day_ctr["csma_drops"] += 1
                        return

                    if not self._csma_channel_busy_for_node(src, p.chan, env.now):
                        p.csma_consecutive_busy = 0
                        break

                    if max_deferrals > 0 and p.csma_deferrals >= max_deferrals:
                        p.csma_dropped = True
                        self.day_ctr["csma_drops"] += 1
                        return

                    p.csma_deferrals += 1
                    self.day_ctr["csma_deferrals"] += 1
                    wait_s = self._csma_defer_s(p)
                    p.csma_consecutive_busy += 1
                    if wait_s <= 0.0:
                        # Avoid zero-time infinite loops if the user sets factor=0.
                        wait_s = 1e-9
                    # Treat the busy-channel defer interval as CCA/RX listening
                    # time for energy accounting. It is added to rx_on_s at the
                    # end of the day, and also kept separately in csma_sense_s
                    # for debugging/analysis.
                    src.csma_cca_s += wait_s
                    src.csma_sense_s += wait_s  # backward-compatible aggregate
                    yield env.timeout(wait_s)

                p.start_s = float(env.now)
                p.csma_delay_s = max(0.0, p.start_s - original_start)
                self.mark_tx(src, p.start_s, p.airtime_s)

            # TX starts now
            self.active_packets.append(p)
            self.tx_history.append(p)

            # Deluge v5 protocol callbacks are tied to actual post-CSMA/CAD TX
            # start times. This is especially important for ADV, because the
            # sender-local REQ collection window must open after the actual ADV
            # transmission, not the originally scheduled one.
            cb = getattr(self, "_on_packet_tx_started", None)
            if cb is not None:
                cb(p)

            # Decide RX candidates at TX start and mark them busy for the whole airtime
            self._start_rx_attempts(p)

            # TX in progress
            yield env.timeout(p.airtime_s)

            # TX ends now
            if p in self.active_packets:
                self.active_packets.remove(p)

            _finish_tx_bookkeeping(src)

            # Finalize reception success/failure at TX end
            self._finish_rx_for_packet(p)

        env.process(tx_proc())

    def _start_rx_attempts(self, p: Packet) -> None:
        """At packet start, select RX candidates and mark them RX-busy."""
        args = self.args
        src_node = self.by_id[p.src]
        sens_dbm = self.radio.sensitivity_dbm_for_args(args, p.sf)

        rx_candidates: List[Tuple[int, float]] = []
        for n in self.nodes:
            if (p.dst is not None) and (p.dst != n.nid):
                continue
            if n.nid == p.src:
                continue

            if not self._can_rx_packet(n, p):
                continue

            rx_dbm = self._rx_power_for_packet_at_node(p, n)
            if rx_dbm < sens_dbm:
                continue

            # Occupy the receiver radio for the full packet duration
            self.mark_rx(n, p.start_s, p.airtime_s)
            rx_candidates.append((n.nid, rx_dbm))

        p._rx_candidates = rx_candidates

    def _finish_rx_for_packet(self, p: Packet) -> None:
        """At packet end, resolve collisions/capture for the RX candidates."""
        args = self.args
        p_end = p.start_s + p.airtime_s

        # All interferers that overlap in time (not only the ones still active at the end)
        overlapping: List[Packet] = []
        for q in self.tx_history:
            if q is p or q.chan != p.chan:
                continue
            q_end = q.start_s + q.airtime_s
            if not (q_end <= p.start_s or p_end <= q.start_s):
                overlapping.append(q)
        overlapping.sort(key=lambda pkt: (pkt.start_s, pkt.src, pkt.ptype))

        src_node = self.by_id[p.src]
        rx_candidates = p._rx_candidates or []

        for nid, rx_dbm in rx_candidates:
            n = self.by_id[nid]

            ok = True
            for q in overlapping:
                rx2_dbm = self._rx_power_for_packet_at_node(q, n)
                if rx2_dbm + args.collision_th_db > rx_dbm:
                    ok = False
                    break

            if not ok:
                n.coll += 1
                continue

            n.rx_ok += 1
            self.day_ctr[f"rx_{p.ptype.lower()}"] += 1

            logical_key = self._logical_packet_key(p)
            is_duplicate = logical_key in n.seen_logical_packets
            self.write_packet_log_row(p, n.nid, is_duplicate=is_duplicate)
            if is_duplicate:
                continue
            n.seen_logical_packets.add(logical_key)

            # Deluge protocol-specific handling is implemented by the Sim
            # class. The channel layer only resolves radio/collision outcomes
            # and dispatches successful, non-duplicate receptions.
            handler = getattr(self, "_handle_packet_rx", None)
            if handler is not None:
                handler(n, p)

