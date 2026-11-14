# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import os
import shlex
import sys
from typing import List, Optional

from radio_profiles import add_radio_profile_args, normalize_radio_args

def _load_config_file_args(path: str) -> list[str]:
    """Load CLI args from a config file.
    Supports:
      - .csv with rows: key,value  (key may be 'sf-adv' or '--sf-adv')
      - .txt/.args: whitespace-separated CLI tokens (like a shell), comments allowed with '#'
    """
    p = path.strip()
    ext = os.path.splitext(p)[1].lower()
    if ext == ".csv":
        out: list[str] = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) < 2:
                    continue
                key = row[0].strip()
                val = row[1].strip()
                if not key or key.startswith("#"):
                    continue
                if not key.startswith("--"):
                    key = "--" + key.lstrip("-")
                # allow boolean flags in csv with empty/true/false
                if val == "" or val.lower() in ("true", "yes", "1"):
                    out.append(key)
                elif val.lower() in ("false", "no", "0"):
                    # skip flag
                    continue
                else:
                    out.extend([key, val])
        return out

    # default: treat as args text file
    tokens: list[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # allow inline comments
            if "#" in s:
                s = s.split("#", 1)[0].strip()
            if not s:
                continue
            tokens.extend(shlex.split(s))
    return tokens


def _preprocess_argv(argv: list[str]) -> list[str]:
    """Expand --config file contents into argv (before argparse parses)."""
    if "--config" not in argv:
        return argv
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("--config requires a path")
            cfg_path = argv[i + 1]
            out.extend(_load_config_file_args(cfg_path))
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description="Deluge-style page-based data dissemination simulator.",
        fromfile_prefix_chars="@",
    )
    p.add_argument("--testbed", required=True)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--config", default=None,
                   help="Optional config file (.txt/.args or .csv). You can also use @file syntax to load args from a text file.")
    p.add_argument("--sf-adv", type=int, default=9)
    p.add_argument("--sf-data", type=int, default=9)
    p.add_argument("--bw-hz", type=int, default=125000)
    p.add_argument("--cr-denom", type=int, default=5, choices=[5, 6, 7, 8])
    add_radio_profile_args(p)
    p.add_argument("--tx-dbm", type=float, default=None)
    p.add_argument(
        "--pl-ref",
        default="100m",
        choices=["100m", "40m"],
        help=(
            "Reference-distance selector. Defaults to 100m because the current "
            "literature-based radio profiles and comparisons use PL(d0) at d0=100 m. "
            "Prefer --pl-ref-m for explicit overrides."
        ),
    )
    p.add_argument("--pl-40m-lpld0", type=float, default=None)
    p.add_argument("--pl-40m-gamma", type=float, default=None)
    p.add_argument("--pl-sigma", type=float, default=None)
    p.add_argument(
        "--fading",
        choices=["none", "rayleigh"],
        default="none",
        help="Small-scale fading model applied per physical packet/receiver link. Default: none.",
    )
    p.add_argument(
        "--rayleigh-min-db",
        type=float,
        default=-40.0,
        help="Lower clamp for Rayleigh fading power gain in dB. Default: -40 dB.",
    )
    p.add_argument(
        "--rayleigh-max-db",
        type=float,
        default=20.0,
        help="Upper clamp for Rayleigh fading power gain in dB. Default: 20 dB.",
    )
    # Accept both spellings for collision threshold
    p.add_argument("--collision-thresh-db", dest="collision_th_db", type=float)
    p.add_argument(
        "--collision_th_db", dest="collision_th_db", type=float, default=None
    )
    p.add_argument("--adv-pl-bytes", type=int, default=8)
    p.add_argument("--upd-pl-bytes", type=int, default=5)
    p.add_argument("--data-pl-bytes", type=int, default=None,
                   help="Useful model bytes inside DATA packet. If omitted, derived from radio max payload minus DATA header.")
    p.add_argument(
        "--model-size-bytes",
        type=int,
        default=0,
        help="Total model size in bytes; if >0, complete_need and data_chunks_per_round are derived from it.",
    )
    p.add_argument("--miss-pl-bytes", type=int, default=8)
    p.add_argument("--miss-len-mode", choices=["auto", "fixed"], default="auto",
                   help="MISS payload length mode: auto uses encoding-based length; fixed uses --miss-pl-bytes for airtime/energy.")
    p.add_argument(
        "--miss-encoding",
        choices=["bitmap", "list"],
        default="list",
    )
    p.add_argument("--miss-hdr-bytes", type=int, default=None,
                   help="Backward-compatible alias for --miss-header-bytes.")
    p.add_argument("--data-header-bytes", type=int, default=None,
                   help="DATA protocol header bytes. Default: 8 if radio max payload <256 B, else 9.")
    p.add_argument("--miss-header-bytes", type=int, default=None,
                   help="MISS protocol header bytes. Default: 6.")
    p.add_argument("--miss-max-packet-bytes", type=int, default=None,
                   help="Maximum total MISS packet size. Default: 247 B.")
    p.add_argument("--adv-budget-s", type=float, default=7.2)
    p.add_argument("--data-budget-s", type=float, default=120.0)
    p.add_argument("--miss-budget-s", type=float, default=20.0)
    p.add_argument("--repair-budget-s", type=float, default=40.0)
    p.add_argument(
        "--auto-data-repair-windows",
        action="store_true",
        help=(
            "Compute DATA and REPAIR phase durations from the selected radio airtime, "
            "complete_need, END packets, proc_gap_s, and the maximum DATA backoff. "
            "When enabled, --data-budget-s and --repair-budget-s are overwritten."
        ),
    )
    p.add_argument(
        "--data-repair-window-margin",
        type=float,
        default=1.0,
        help="Multiplier applied to the computed DATA/REPAIR phase duration when --auto-data-repair-windows is enabled.",
    )
    p.add_argument("--area-m2", type=float, default=None,
                   help="Deployment area in square meters. If omitted, the simulator estimates it from the testbed bounding box.")
    p.add_argument("--comm-radius-m", type=float, default=None,
                   help="Override communication radius R in meters for probability/window calculations.")
    p.add_argument("--auto-control-windows", action="store_true",
                   help="Compute ADV/UPD/MISS windows and backoffs from the collision-probability model.")
    p.add_argument("--control-window-node-factor", type=float, default=1.0,
                   help="Multiplier for the node count used in automatic control-window formulas. For CSMA hidden-terminal adjustment use 0.5.")
    p.add_argument("--control-window-node-offset", type=float, default=0.0,
                   help="Additive offset for the effective node count used in automatic control-window formulas. For N/2 - 1 use -1 with factor 0.5.")
    p.add_argument("--cad-control-window-node-factor", type=float, default=0.5,
                   help=(
                       "Multiplier for the node count used in automatic control-window formulas when --cad-csma is enabled. "
                       "Default: 0.5, giving approximately N/2 - 1 together with --cad-control-window-node-offset -1."
                   ))
    p.add_argument("--cad-control-window-node-offset", type=float, default=-1.0,
                   help=(
                       "Additive offset for the effective node count used in automatic control-window formulas when --cad-csma is enabled. "
                       "Default: -1, giving approximately N/2 - 1 together with --cad-control-window-node-factor 0.5."
                   ))
    p.add_argument("--reply-collision-rho", type=float, default=0.05,
                   help="Maximum acceptable reply collision probability rho for automatic control windows.")
    p.add_argument("--min-control-window-s", type=float, default=0.0,
                   help="Optional lower clamp for computed control windows.")
    p.add_argument("--max-control-window-s", type=float, default=0.0,
                   help="Optional upper clamp for computed control windows; 0 disables upper clamp.")
    p.add_argument("--miss-window-scale-factor", type=float, default=1.0,
                   help="Multiplier applied after dividing the raw MISS selection interval by the data-channel count.")

    p.add_argument("--phase-gap-ms", type=float, default=200.0)
    p.add_argument("--upd-listen-ms", type=float, default=1000.0,
                   help="Extra time after ADV phase to keep owners listening on UPD channel before DATA starts (ms).")
    p.add_argument("--miss-listen-ms", type=float, default=0.0,
                   help="Extra time to keep owners listening for MISS before REPAIR starts (ms). If larger than miss-budget, extends the MISS window.")
    p.add_argument("--proc-gap-ms", type=float, default=2.0)
    p.add_argument("--adv-backoff-ms", type=float, nargs=2, default=[0.0, 500.0])
    p.add_argument("--upd-backoff-ms", type=float, nargs=2, default=[0.0, 20.0])
    p.add_argument("--miss-backoff-ms", type=float, nargs=2, default=[0.0, 800.0])
    p.add_argument("--data-backoff-ms", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument(
        "--repair-backoff-ms", type=float, nargs=2, default=[0.0, 800.0]
    )
    p.add_argument("--tx-to-rx-s", type=float, default=0.003)
    p.add_argument("--rx-to-tx-s", type=float, default=0.004)
    p.add_argument("--num-channels", type=int, default=None,
                   help="Total usable radio channels c. If omitted, selected radio profile default is used.")
    p.add_argument("--chan-adv", type=int, default=0)
    p.add_argument("--chan-upd", type=int, default=0)
    p.add_argument("--chan-data", type=int, default=0)
    # Generic physical repetition layer. Used for BLE Mesh-like / advertising-like behavior,
    # but kept generic so experiments can enable it for any profile.
    p.add_argument("--phy-repetitions", type=int, default=1,
                   help="Number of physical transmissions per logical packet. Default: 1.")
    p.add_argument("--phy-repeat-gap-ms", type=float, default=0.0,
                   help="Fixed gap between repeated physical transmissions, in ms. Gap is added after each copy airtime.")
    p.add_argument("--phy-repeat-jitter-ms", type=float, default=0.0,
                   help="Random extra gap [0, jitter] between repeated physical transmissions, in ms.")
    p.add_argument("--phy-repeat-stages", nargs="+",
                   default=["ADV", "REQ", "DATA"],
                   choices=["ADV", "REQ", "DATA", "UPD", "MISS", "REPAIR", "END"],
                   help="Packet types that use physical repetitions when --phy-repetitions > 1.")
    p.add_argument("--phy-repeat-mode", choices=["same_channel", "channel_hop"], default="same_channel",
                   help="Whether repeated copies stay on the same channel or hop to consecutive channels with wraparound.")
    p.add_argument("--phy-repeat-channel-count", type=int, default=0,
                   help="Number of channels used for channel_hop repetitions. 0 means min(phy_repetitions, num_channels).")
    p.add_argument("--phy-repeat-rx-scan", action=argparse.BooleanOptionalAction, default=True,
                   help="Allow a receiver tuned to the base channel to follow channel-hopped repetitions. Default: true.")

    # Simplified CSMA/CA-like channel access. Intended for IEEE 802.11ah / HaLow
    # experiments, but kept generic so it can be enabled for any profile.
    p.add_argument("--csma-ca", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable simplified CSMA/CA-like clear-channel assessment before TX. Default: disabled.")
    p.add_argument("--csma-stages", nargs="+",
                   default=["ADV", "REQ", "DATA"],
                   choices=["ADV", "REQ", "DATA", "UPD", "MISS", "REPAIR", "END"],
                   help="Packet types that use CSMA when --csma-ca is enabled.")
    p.add_argument("--csma-carrier-sense-thresh-dbm", type=float, default=-85.0,
                   help="CCA threshold in dBm. Same-channel packets above this power at the sender make the channel busy.")
    p.add_argument("--csma-busy-wait-airtime-factor", type=float, default=0.5,
                   help=(
                       "Initial busy-channel defer time as a multiple of packet airtime. "
                       "Consecutive busy detections double this up to 2.0x airtime. Default: 0.5."
                   ))
    p.add_argument("--csma-busy-wait-jitter", type=float, default=0.0,
                   help="Relative jitter for CSMA defer time. 0.25 means uniform multiplier in [0.75, 1.25].")
    p.add_argument("--csma-max-deferrals", type=int, default=3,
                   help=(
                       "Maximum CSMA busy-channel deferrals per physical packet. "
                       "Default: 3, which with --csma-busy-wait-airtime-factor=0.5 "
                       "gives max backoff 0.5+1.0+2.0=3.5 packet airtimes, "
                       "so T_CSMA,max = T_pkt + T_bo,max = 4.5*T_pkt. "
                       "Use 0 for unlimited deferrals until the day safety cap."
                   ))

    # LoRa CAD-based CSMA. This is separate from generic RSSI/CCA-style CSMA:
    # it uses LoRa CAD duration and SF-selective LoRa packet detection.
    p.add_argument("--cad-csma", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable LoRa CAD-based CSMA before TX. Default: disabled.")
    p.add_argument("--cad-csma-stages", nargs="+",
                   default=["ADV", "REQ", "DATA"],
                   choices=["ADV", "REQ", "DATA", "UPD", "MISS", "REPAIR", "END"],
                   help="Packet types that use CAD-CSMA when --cad-csma is enabled.")
    p.add_argument("--cad-symbol-num", type=int, default=2, choices=[1, 2, 4, 8, 16],
                   help="Number of LoRa symbols used for one CAD scan. SX1280 supports 1, 2, 4, 8, or 16. Default: 2.")
    p.add_argument("--cad-detect-margin-db", type=float, default=0.0,
                   help="Margin added to LoRa sensitivity for CAD busy detection. 0 means sensitivity threshold.")
    p.add_argument("--cad-busy-wait-airtime-factor", type=float, default=0.5,
                   help="Initial CAD busy defer time as a multiple of packet airtime. Consecutive busy detections double this up to 2.0x airtime. Default: 0.5.")
    p.add_argument("--cad-busy-wait-jitter", type=float, default=0.0,
                   help="Relative jitter for CAD busy defer time. 0.25 means uniform multiplier in [0.75, 1.25].")
    p.add_argument("--cad-max-deferrals", type=int, default=0,
                   help="Maximum CAD-CSMA busy-channel deferrals per physical packet. 0 means unlimited until the day safety cap.")

    # Simplified per-cycle duty-cycle TX budget. Intended for Sub-GHz FSK and
    # IEEE 802.11ah/HaLow experiments. A value of 0 disables the limit.
    p.add_argument("--duty-cycle-limit-s", type=float, default=0.0,
                   help="Maximum TX airtime per node per cycle/day in seconds. 0 disables duty-cycle limiting. Example: 36 for 1%% of one hour.")
    p.add_argument("--duty-cycle-data-fraction", type=float, default=0.5,
                   help="Fraction of the duty-cycle limit reserved for DATA-side transmissions. Remaining budget is used for REPAIR. Default: 0.5.")
    p.add_argument("--duty-cycle-include-adv-in-data", action=argparse.BooleanOptionalAction, default=True,
                   help="Subtract owner's ADV TX airtime from the DATA-side duty-cycle budget. Default: true.")
    p.add_argument("--data-chunks-per-round", type=int, default=50)
    p.add_argument("--complete-need", type=int, default=40)
    p.add_argument(
        "--version-start",
        type=int,
        default=1,
        help="Initial version advertised by starting gateways.",
    )
    p.add_argument(
        "--dynamic-data-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use owner-local dynamic DATA/MISS/REPAIR behavior with inactivity timeouts. Default: enabled.",
    )
    p.add_argument(
        "--data-inactivity-airtimes",
        type=float,
        default=10.0,
        help="DATA receiver inactivity timeout measured in logical DATA airtimes. Default: 10.",
    )
    p.add_argument(
        "--repair-inactivity-airtimes",
        type=float,
        default=10.0,
        help="REPAIR receiver inactivity timeout measured in logical REPAIR airtimes. Default: 10.",
    )
    p.add_argument(
        "--owner-miss-extra-airtimes",
        type=float,
        default=2.0,
        help="Extra owner MISS listening time after the expected latest MISS, measured in MISS airtimes. Default: 2.",
    )
    p.add_argument(
        "--repair-chunks-per-round",
        type=int,
        default=0,
        help="Max REPAIR chunks per owner per round. 0 means unlimited.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--until-complete", action="store_true")
    p.add_argument("--max-days", type=int, default=30)
    p.add_argument("--no-packet-log", action="store_true")
    p.add_argument("--animate", action="store_true")

    # Deluge protocol parameters
    p.add_argument("--deluge-page-packets", type=int, default=48,
                   help="Number of DATA packets/chunks per Deluge page.")
    p.add_argument("--deluge-adv-window-ms", type=float, default=None,
                   help="ADV window in ms. If omitted, --adv-budget-s is used.")
    p.add_argument("--deluge-req-window-ms", type=float, default=1000.0,
                   help="REQ collection window in ms.")
    p.add_argument("--deluge-req-backoff-ms", type=float, nargs=2, default=[0.0, 500.0],
                   help="Random REQ backoff range in ms.")
    p.add_argument("--deluge-data-gap-ms", type=float, default=2.0,
                   help="Gap between DATA packets in a Deluge page train, in ms.")
    p.add_argument("--deluge-round-gap-ms", type=float, default=200.0,
                   help="Gap between ADV/REQ/DATA sections, in ms.")
    p.add_argument("--deluge-sender-selection", choices=["recent"], default="recent")
    p.add_argument("--deluge-page-selection", choices=["lowest"], default="lowest")
    p.add_argument("--deluge-packet-order", choices=["sequential", "random"], default="sequential")
    p.add_argument("--deluge-immediate-adv-on-page-complete", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-cycles", type=int, default=None, help="Alias for --max-days in Deluge output.")
    p.add_argument("--anim-fps", type=float, default=0.5)
    argv = sys.argv[1:] if argv is None else argv
    argv = _preprocess_argv(argv)
    args = p.parse_args(argv)
    # Backward-compatible alias: --miss-hdr-bytes maps to --miss-header-bytes.
    if getattr(args, "miss_header_bytes", None) is None and getattr(args, "miss_hdr_bytes", None) is not None:
        args.miss_header_bytes = args.miss_hdr_bytes
    args = normalize_radio_args(args)

    # Derived fields
    args.phase_gap_s = args.phase_gap_ms / 1000.0
    if getattr(args, "max_cycles", None) is not None:
        args.max_days = args.max_cycles
    if getattr(args, "deluge_adv_window_ms", None) is None:
        args.deluge_adv_window_ms = float(args.adv_budget_s) * 1000.0
    args.deluge_adv_window_s = args.deluge_adv_window_ms / 1000.0
    args.deluge_req_window_s = args.deluge_req_window_ms / 1000.0
    args.deluge_data_gap_s = args.deluge_data_gap_ms / 1000.0
    args.deluge_round_gap_s = args.deluge_round_gap_ms / 1000.0
    args.upd_listen_s = args.upd_listen_ms / 1000.0
    args.miss_listen_s = args.miss_listen_ms / 1000.0
    args.proc_gap_s = args.proc_gap_ms / 1000.0

    # Validate channel args
    if args.num_channels < 1:
        raise SystemExit("--num-channels must be >= 1")
    for name in ["chan_adv", "chan_upd", "chan_data"]:
        val = getattr(args, name)
        if not (0 <= val < args.num_channels):
            raise SystemExit(
                f"--{name} must be in [0,{args.num_channels-1}] when --num-channels={args.num_channels}"
            )

    if args.data_inactivity_airtimes <= 0:
        raise SystemExit("--data-inactivity-airtimes must be > 0")
    if args.repair_inactivity_airtimes <= 0:
        raise SystemExit("--repair-inactivity-airtimes must be > 0")
    if args.owner_miss_extra_airtimes < 0:
        raise SystemExit("--owner-miss-extra-airtimes must be >= 0")
    if args.repair_chunks_per_round < 0:
        raise SystemExit("--repair-chunks-per-round must be >= 0")

    if args.deluge_page_packets < 1:
        raise SystemExit("--deluge-page-packets must be >= 1")
    if args.phy_repetitions < 1:
        raise SystemExit("--phy-repetitions must be >= 1")
    if args.phy_repeat_gap_ms < 0.0:
        raise SystemExit("--phy-repeat-gap-ms must be >= 0")
    if args.phy_repeat_jitter_ms < 0.0:
        raise SystemExit("--phy-repeat-jitter-ms must be >= 0")
    if args.phy_repeat_channel_count < 0:
        raise SystemExit("--phy-repeat-channel-count must be >= 0")
    # Normalize repeat stage names to uppercase for robust config parsing.
    args.phy_repeat_stages = [str(x).upper() for x in (args.phy_repeat_stages or [])]

    if args.csma_busy_wait_airtime_factor < 0.0:
        raise SystemExit("--csma-busy-wait-airtime-factor must be >= 0")
    if args.csma_busy_wait_jitter < 0.0:
        raise SystemExit("--csma-busy-wait-jitter must be >= 0")
    if args.csma_max_deferrals < 0:
        raise SystemExit("--csma-max-deferrals must be >= 0")
    args.csma_stages = [str(x).upper() for x in (args.csma_stages or [])]

    if args.cad_busy_wait_airtime_factor < 0.0:
        raise SystemExit("--cad-busy-wait-airtime-factor must be >= 0")
    if args.cad_busy_wait_jitter < 0.0:
        raise SystemExit("--cad-busy-wait-jitter must be >= 0")
    if args.cad_max_deferrals < 0:
        raise SystemExit("--cad-max-deferrals must be >= 0")
    args.cad_csma_stages = [str(x).upper() for x in (args.cad_csma_stages or [])]

    if args.duty_cycle_limit_s < 0.0:
        raise SystemExit("--duty-cycle-limit-s must be >= 0")
    if not (0.0 <= args.duty_cycle_data_fraction <= 1.0):
        raise SystemExit("--duty-cycle-data-fraction must be in [0, 1]")
    if args.control_window_node_factor < 0.0:
        raise SystemExit("--control-window-node-factor must be >= 0")
    if args.cad_control_window_node_factor < 0.0:
        raise SystemExit("--cad-control-window-node-factor must be >= 0")

    # Safety default if None due to merged flags
    if getattr(args, "collision_th_db", None) is None:
        args.collision_th_db = 1.0

    if args.rayleigh_min_db > args.rayleigh_max_db:
        raise SystemExit("--rayleigh-min-db must be <= --rayleigh-max-db")

    return args


