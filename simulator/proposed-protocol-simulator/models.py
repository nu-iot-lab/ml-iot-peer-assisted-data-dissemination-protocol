# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

def _splitmix64(seed: int) -> int:
    """Deterministic 64-bit mixing function to derive independent per-node seeds."""
    x = seed & 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF

@dataclass
class Node:
    nid: int
    x: float
    y: float
    is_gateway: bool = False
    updated: bool = False
    # Version tracking
    version: int = 0
    have_chunks: Dict[int, Set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    day_rx_chunks: Set[Tuple[int, int]] = field(default_factory=set)
    # Sender-side progress for duty-cycle-limited DATA dissemination.
    # version -> next position in sorted chunk list to send in a later cycle.
    data_next_chunk_by_version: Dict[int, int] = field(default_factory=dict)
    # Logical packet duplicate suppression for repeated physical transmissions.
    # Keys are usually (day_idx, logical_id), with a fallback tuple for legacy packets.
    seen_logical_packets: Set[Tuple] = field(default_factory=set)
    # Radio state
    tx_busy_until: float = 0.0
    rx_busy_until: float = 0.0
    # Radio energy accounting
    # rx_busy_s: time spent actively attempting to receive (packet overlaps), including collided receptions.
    # rx_on_s: total time radio is in RX mode (listening), including idle listening windows
    #          and CSMA/CCA/CAD sensing time. Kept for backward compatibility.
    # rx_on_*_s: RX/listening windows split by protocol stage for accurate energy reports.
    # csma_cca_s: generic CSMA/CCA busy-channel sensing/defer time.
    # cad_scan_s: LoRa CAD scan time. cad_busy defer/backoff sleep is not counted as RX.
    # csma_sense_s: backward-compatible aggregate = csma_cca_s + cad_scan_s.
    rx_busy_s: float = 0.0
    rx_on_s: float = 0.0
    rx_on_adv_s: float = 0.0
    rx_on_upd_s: float = 0.0
    rx_on_data_s: float = 0.0
    rx_on_miss_s: float = 0.0
    rx_on_repair_s: float = 0.0
    csma_cca_s: float = 0.0
    cad_scan_s: float = 0.0
    csma_sense_s: float = 0.0
    # TX intervals for subtracting RX-on overlap (half-duplex)
    day_tx_intervals: List[Tuple[float, float]] = field(default_factory=list)
    tune_plan: List[Tuple[float, float, int, str]] = field(default_factory=list)
    # Dynamic RX windows used by asynchronous DATA/REPAIR owner-local stages.
    # Mutable [start, end, chan, mode] entries let runtime timeout/END events shorten listening.
    dynamic_tune_plan: List[List[Any]] = field(default_factory=list)
    data_listen_idx: Optional[int] = None
    repair_listen_idx: Optional[int] = None
    owner_miss_listen_idx: Optional[int] = None
    data_timeout_token: int = 0
    repair_timeout_token: int = 0
    data_miss_scheduled: bool = False
    repair_done_today: bool = False
    # Stats
    rx_ok: int = 0
    tx_ok: int = 0
    # TX counters by type (cumulative)
    tx_adv: int = 0
    tx_upd: int = 0
    tx_data: int = 0
    tx_miss: int = 0
    tx_end: int = 0
    tx_repair: int = 0
    # Per-day deltas (aligned by day index)
    day_tx_adv: List[int] = field(default_factory=list)
    day_tx_upd: List[int] = field(default_factory=list)
    day_tx_data: List[int] = field(default_factory=list)
    day_tx_miss: List[int] = field(default_factory=list)
    day_tx_end: List[int] = field(default_factory=list)
    day_tx_repair: List[int] = field(default_factory=list)
    # Per-day deltas
    day_rx_on_ms: List[float] = field(default_factory=list)
    day_rx_on_adv_ms: List[float] = field(default_factory=list)
    day_rx_on_upd_ms: List[float] = field(default_factory=list)
    day_rx_on_data_ms: List[float] = field(default_factory=list)
    day_rx_on_miss_ms: List[float] = field(default_factory=list)
    day_rx_on_repair_ms: List[float] = field(default_factory=list)
    day_rx_busy_ms: List[float] = field(default_factory=list)
    day_csma_cca_ms: List[float] = field(default_factory=list)
    day_cad_scan_ms: List[float] = field(default_factory=list)
    day_csma_sense_ms: List[float] = field(default_factory=list)
    coll: int = 0
    adv_used_s: float = 0.0
    upd_used_s: float = 0.0
    data_used_s: float = 0.0
    miss_used_s: float = 0.0
    repair_used_s: float = 0.0
    last_day_partial: bool = False
    # Per-day logical role / channels
    owner_for_day: Optional[int] = None
    listen_chan_data: Optional[int] = None
    data_chan: Optional[int] = None
    repair_listen_chan: Optional[int] = None
    # END tracking (per day, per version)
    end_rx_counts: Dict[int, int] = field(default_factory=dict)  # version -> END count
    last_chunk_idx: Dict[int, int] = field(default_factory=dict)  # version -> last chunk index
    # ADV probability state
    adv_cycles: int = 0  # number of completed cycles spent at current committed version
    version_changed_today: bool = False  # used to avoid incrementing adv_cycles on the same day as an update


@dataclass
class Packet:
    ptype: str
    src: int
    dst: Optional[int]
    sf: int
    bw_hz: int
    cr_denom: int
    start_s: float
    airtime_s: float
    power_dbm: float
    chan: int = 0
    version: int = 0
    chunk_idx: Optional[int] = None
    data_chan: Optional[int] = None
    # Simulated 16-bit on-air Session ID. DATA and REPAIR carry it in the header.
    session_id: int = 0
    # Logical phase metadata for END packets and dynamic owner-local stages.
    # ptype remains END for backwards-compatible logs/counters; end_seq tells END1/END2/END3.
    phase: Optional[str] = None
    end_seq: int = 0
    # Repetition metadata for BLE Mesh-like / advertising-like physical copies.
    # All repetitions of the same logical packet share logical_id and differ by rep_idx.
    logical_id: Optional[int] = None
    rep_idx: int = 0
    rep_count: int = 1
    base_chan: Optional[int] = None
    # CSMA/CAD-like channel access metadata. These are filled at physical TX time.
    csma_deferrals: int = 0
    csma_delay_s: float = 0.0
    csma_dropped: bool = False
    # Number of consecutive busy detections for the current physical packet.
    # Used to scale defer time exponentially, e.g. 1x, 2x, then 4x airtime by default.
    csma_consecutive_busy: int = 0
    _requested_indices: Optional[List[int]] = None
    _miss_payload_bytes: Optional[int] = None
    _rx_candidates: Optional[List[Tuple[int, float]]] = None  # (nid, rx_dbm)
    # Per physical packet / per receiver received-power cache.
    # This keeps sensitivity and capture decisions consistent when fading is enabled.
    _rx_power_by_nid: Dict[int, float] = field(default_factory=dict)


