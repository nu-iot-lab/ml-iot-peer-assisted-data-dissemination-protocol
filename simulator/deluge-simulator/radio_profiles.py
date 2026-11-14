#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Radio profile layer for the data distribution simulator.

The simulator protocol stays the same (ADV -> UPD -> DATA -> MISS -> REPAIR),
while radio profiles provide PHY-specific defaults and calculations:
  - airtime calculation
  - sensitivity lookup
  - maximum payload size
  - default TX power
  - default path-loss / attenuation parameters

The profile gives technology defaults. CLI args/config files select the profile
and can override parameters for experiments.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional


def _coalesce(*values):
    """Return the first value that is not None."""
    for v in values:
        if v is not None:
            return v
    return None


@dataclass(frozen=True)
class RadioProfile:
    """Base radio profile.

    Subclasses should override `airtime_seconds()` and `sensitivity_dbm()`.
    Profiles define default link-budget/path-loss assumptions, while args can
    override them.
    """

    name: str
    max_payload_bytes: int
    default_tx_dbm: float
    default_pl_ref_m: float
    default_pl_ref_loss_db: float
    default_pl_gamma: float
    default_pl_sigma: float
    default_collision_th_db: float
    default_channel_count: int

    def airtime_seconds(self, sf: int, bw_hz: int, cr_denom: int, pl_bytes: int) -> float:
        raise NotImplementedError

    def sensitivity_dbm(self, sf: int) -> float:
        raise NotImplementedError

    def airtime_seconds_for_args(
        self,
        args: argparse.Namespace,
        sf: int,
        bw_hz: int,
        cr_denom: int,
        pl_bytes: int,
    ) -> float:
        return self.airtime_seconds(sf, bw_hz, cr_denom, pl_bytes)

    def sensitivity_dbm_for_args(self, args: argparse.Namespace, sf: int) -> float:
        return self.sensitivity_dbm(sf)

    def cad_duration_seconds_for_args(self, args: argparse.Namespace, sf: int, bw_hz: int) -> float:
        """LoRa CAD duration in seconds using the SX1280-style formula.

        T_CAD = (cadSymbolNum + (2*SF + 3)/32) * T_sym,
        where T_sym = 2**SF / BW.

        This is meaningful for LoRa packet profiles. Non-LoRa profiles should not
        enable --cad-csma unless they intentionally reuse this approximation.
        """
        cad_symbols = int(getattr(args, "cad_symbol_num", 2) or 2)
        if cad_symbols <= 0:
            raise ValueError("cad_symbol_num must be > 0")
        bw = float(bw_hz)
        if bw <= 0.0:
            raise ValueError("LoRa CAD bandwidth must be > 0")
        tsym = (2.0 ** int(sf)) / bw
        return (float(cad_symbols) + ((2.0 * float(sf) + 3.0) / 32.0)) * tsym

    def effective_pl_ref_m(self, args: argparse.Namespace) -> float:
        # Generic arg preferred; legacy --pl-ref 40m supported via parse normalization.
        return float(_coalesce(getattr(args, "pl_ref_m", None), self.default_pl_ref_m))

    def effective_pl_ref_loss_db(self, args: argparse.Namespace) -> float:
        return float(
            _coalesce(
                getattr(args, "pl_ref_loss_db", None),
                getattr(args, "pl_40m_lpld0", None),
                self.default_pl_ref_loss_db,
            )
        )

    def effective_pl_gamma(self, args: argparse.Namespace) -> float:
        return float(
            _coalesce(
                getattr(args, "pl_gamma", None),
                getattr(args, "pl_40m_gamma", None),
                self.default_pl_gamma,
            )
        )

    def effective_pl_sigma(self, args: argparse.Namespace) -> float:
        return float(_coalesce(getattr(args, "pl_sigma", None), self.default_pl_sigma))

    def path_loss_db(self, args: argparse.Namespace, d_m: float) -> float:
        """Log-distance path-loss model.

        PL(d) = PL(d0) + 10 * gamma * log10(d / d0)

        Profile defaults define d0, PL(d0), gamma, and sigma. Args can override
        them with --pl-ref-m, --pl-ref-loss-db, --pl-gamma, --pl-sigma.
        """
        d = max(float(d_m), 1.0)
        d0 = self.effective_pl_ref_m(args)
        if d0 <= 0:
            raise ValueError("Path-loss reference distance must be > 0")
        l0 = self.effective_pl_ref_loss_db(args)
        gamma = self.effective_pl_gamma(args)
        return l0 + 10.0 * gamma * math.log10(d / d0)

    def small_scale_fading_db(self, args: argparse.Namespace, rng: random.Random) -> float:
        """Small-scale fading power gain in dB.

        Supported by --fading:
          - none:     0 dB gain/loss
          - rayleigh: Rayleigh envelope, i.e. exponential power gain with mean 1

        The channel layer calls rx_power_dbm() once per physical packet/receiver
        link and caches the result on the Packet object. Therefore the same
        Rayleigh/shadowing sample is reused consistently for sensitivity and
        capture/collision decisions for that packet at that receiver.

        --rayleigh-min-db and --rayleigh-max-db clamp rare extreme samples so
        experiments remain numerically stable and reproducible.
        """
        fading = str(getattr(args, "fading", "none") or "none").lower()
        if fading == "none":
            return 0.0
        if fading != "rayleigh":
            return 0.0

        # Rayleigh envelope squared -> exponential power gain in linear units.
        # Mean is 1, so the average power gain is 0 dB before clamping.
        gain_linear = max(rng.expovariate(1.0), 1e-12)
        gain_db = 10.0 * math.log10(gain_linear)
        min_db = float(getattr(args, "rayleigh_min_db", -40.0))
        max_db = float(getattr(args, "rayleigh_max_db", 20.0))
        return max(min_db, min(max_db, gain_db))

    def rx_power_dbm(
        self,
        args: argparse.Namespace,
        rng: random.Random,
        d_m: float,
        ptx_dbm: float,
    ) -> float:
        """Received power in dBm with log-distance PL, shadowing, and optional fading."""
        mean_pl = self.path_loss_db(args, d_m)
        sigma = self.effective_pl_sigma(args)
        shadow = rng.gauss(0.0, sigma) if sigma > 0.0 else 0.0
        fading_db = self.small_scale_fading_db(args, rng)
        return ptx_dbm - (mean_pl + shadow) + fading_db

    @property
    def default_data_header_bytes(self) -> int:
        """DATA protocol header including session ID: 10 B or 11 B."""
        return 10 if int(self.max_payload_bytes) < 256 else 11

    @property
    def default_miss_header_bytes(self) -> int:
        return 6

    @property
    def default_miss_max_packet_bytes(self) -> int:
        return 247

    def communication_radius_m(self, args: argparse.Namespace, sf: int) -> float:
        """Estimate max communication radius from link budget and log-distance path loss."""
        d0 = self.effective_pl_ref_m(args)
        l0 = self.effective_pl_ref_loss_db(args)
        gamma = self.effective_pl_gamma(args)
        tx_dbm = float(_coalesce(getattr(args, "tx_dbm", None), self.default_tx_dbm))
        sens_dbm = self.sensitivity_dbm_for_args(args, sf)
        if d0 <= 0.0 or gamma <= 0.0:
            return 0.0
        return d0 * (10.0 ** ((tx_dbm - sens_dbm - l0) / (10.0 * gamma)))

    def validate_payload_bytes(self, payload_bytes: int, label: str) -> None:
        if payload_bytes > self.max_payload_bytes:
            raise SystemExit(
                f"{label}={payload_bytes} B exceeds max payload for "
                f"{self.name}: {self.max_payload_bytes} B"
            )


@dataclass(frozen=True)
class LoRaSubGHzProfile(RadioProfile):
    """LoRa sub-GHz / SX127x-style raw packet profile."""

    def __init__(self):
        super().__init__(
            name="lora_subghz",
            max_payload_bytes=255,
            default_tx_dbm=14.0,
            # Converted legacy sub-GHz LoRa reference from PL(40m)=110 dB, gamma=2.08
            # to PL(100m)=118.28 dB so all current profile defaults use d0=100 m.
            default_pl_ref_m=100.0,
            default_pl_ref_loss_db=118.28,
            default_pl_gamma=2.08,
            default_pl_sigma=0.0,
            default_collision_th_db=1.0,
            default_channel_count=9,
        )

    def airtime_seconds(self, sf: int, bw_hz: int, cr_denom: int, pl_bytes: int) -> float:
        """LoRa sub-GHz/SX127x airtime approximation in seconds."""
        cr = cr_denom - 4
        preamble_len = 8
        ih = 0
        crc = 1
        de = 1 if sf >= 11 else 0
        tsym = (2.0**sf) / bw_hz
        tpreamble = (preamble_len + 4.25) * tsym
        payload_symb_nb = 8 + max(
            math.ceil(
                (8 * pl_bytes - 4 * sf + 28 + 16 * crc - 20 * ih)
                / (4.0 * (sf - 2 * de))
            ) * (cr + 4),
            0,
        )
        return tpreamble + payload_symb_nb * tsym

    def sensitivity_dbm(self, sf: int) -> float:
        table: Dict[int, float] = {
            7: -123.0,
            8: -126.0,
            9: -129.0,
            10: -132.0,
            11: -134.5,
            12: -137.0,
        }
        return table.get(sf, -140.0)


@dataclass(frozen=True)
class LoRa24Profile(RadioProfile):
    """LoRa 2.4 GHz / SX1280-style raw packet profile."""

    def __init__(self):
        super().__init__(
            name="lora24",
            max_payload_bytes=255,
            default_tx_dbm=12.5,
            # Literature/profile table value for LoRa 2.4 SF7:
            # d0=100 m, PL(d0)=97 dB, Srx(SF7)=-112 dBm, gamma=3.0, R≈825 m.
            # The sensitivity table below still allows SF9/SF10/etc. experiments.
            default_pl_ref_m=100.0,
            default_pl_ref_loss_db=97.0,
            default_pl_gamma=3.0,
            default_pl_sigma=3.57,
            default_collision_th_db=1.0,
            default_channel_count=83,
        )

    def airtime_seconds(self, sf: int, bw_hz: int, cr_denom: int, pl_bytes: int) -> float:
        """SX1280 LoRa airtime, variable-length header, CRC enabled."""
        cr = cr_denom - 4
        preamble = 12
        crc_bits = 16
        header_symbols = 20

        tsym = (2.0 ** sf) / float(bw_hz)

        if sf < 7:
            n_payload = math.ceil(
                max(
                    8 * pl_bytes + crc_bits - 4 * sf + header_symbols,
                    0
                ) / (4.0 * sf)
            ) * (cr + 4)

            n_symbols = preamble + 6.25 + 8 + n_payload

        elif sf <= 10:
            n_payload = math.ceil(
                max(
                    8 * pl_bytes + crc_bits - 4 * sf + 8 + header_symbols,
                    0
                ) / (4.0 * sf)
            ) * (cr + 4)

            n_symbols = preamble + 4.25 + 8 + n_payload

        else:
            n_payload = math.ceil(
                max(
                    8 * pl_bytes + crc_bits - 4 * sf + 8 + header_symbols,
                    0
                ) / (4.0 * (sf - 2))
            ) * (cr + 4)

            n_symbols = preamble + 4.25 + 8 + n_payload

        return n_symbols * tsym

    def sensitivity_dbm(self, sf: int) -> float:
        table: Dict[int, float] = {
            5: -105.0,
            6: -108.0,
            7: -112.0,
            8: -115.0,
            9: -117.0,
            10: -120.0,
            11: -123.0,
            12: -126.0,
        }
        return table.get(sf, -140.0)


@dataclass(frozen=True)
class RawBitrateProfile(RadioProfile):
    """Generic raw PHY-like profile using bitrate-based airtime.

    This intentionally strips MAC/network details and behaves like a simple
    broadcast datagram radio in the simulator.
    """

    default_bitrate_bps: float
    default_phy_overhead_bytes: int
    default_sensitivity_dbm: float

    def effective_bitrate_bps(self, args: argparse.Namespace) -> float:
        return float(_coalesce(getattr(args, "raw_bitrate_bps", None), self.default_bitrate_bps))

    def effective_phy_overhead_bytes(self, args: argparse.Namespace) -> int:
        return int(_coalesce(getattr(args, "raw_phy_overhead_bytes", None), self.default_phy_overhead_bytes))

    def airtime_seconds(self, sf: int, bw_hz: int, cr_denom: int, pl_bytes: int) -> float:
        # This method is kept for compatibility with the simulator call signature.
        # Args-specific overrides are applied through `airtime_seconds_with_args()` below.
        return ((pl_bytes + self.default_phy_overhead_bytes) * 8.0) / self.default_bitrate_bps

    def airtime_seconds_with_args(self, args: argparse.Namespace, pl_bytes: int) -> float:
        bitrate = self.effective_bitrate_bps(args)
        if bitrate <= 0:
            raise ValueError("Raw bitrate must be > 0")
        overhead = self.effective_phy_overhead_bytes(args)
        return ((pl_bytes + overhead) * 8.0) / bitrate

    def sensitivity_dbm(self, sf: int) -> float:
        return self.default_sensitivity_dbm

    def airtime_seconds_for_args(
        self,
        args: argparse.Namespace,
        sf: int,
        bw_hz: int,
        cr_denom: int,
        pl_bytes: int,
    ) -> float:
        return self.airtime_seconds_with_args(args, pl_bytes)

    def sensitivity_dbm_with_args(self, args: argparse.Namespace, sf: int) -> float:
        return float(_coalesce(getattr(args, "raw_sensitivity_dbm", None), self.default_sensitivity_dbm))

    def sensitivity_dbm_for_args(self, args: argparse.Namespace, sf: int) -> float:
        return self.sensitivity_dbm_with_args(args, sf)


class BLE5CodedS8RawProfile(RawBitrateProfile):
    """BLE 5 Coded PHY S=8 raw profile.

    Literature/table defaults:
      - payload: 247 B
      - effective bitrate used by the experiment table: 18 kbps
      - d0=100 m, Pt=8 dBm, PL(d0)=89.35 dB, Srx=-103 dBm, gamma=3.0, R≈527 m
    """

    def __init__(self):
        super().__init__(
            name="ble5_coded_s8_raw",
            max_payload_bytes=247,
            default_tx_dbm=8.0,
            default_pl_ref_m=100.0,
            default_pl_ref_loss_db=89.35,
            default_pl_gamma=3.0,
            default_pl_sigma=3.57,
            default_collision_th_db=1.0,
            default_channel_count=37,
            default_bitrate_bps=18_000.0,
            default_phy_overhead_bytes=20,
            default_sensitivity_dbm=-103.0,
        )


# Backward-compatible class name. Older configs may still say ble5_dle_raw,
# but the profile now represents BLE 5 Coded S=8.
class BLE5DLERawProfile(BLE5CodedS8RawProfile):
    pass


class SubGHzFSKRawProfile(RawBitrateProfile):
    """Sub-GHz 2-FSK profile with explicit PHY framing.

    The payload is transmitted at the configured raw bitrate.  PHY airtime adds
    an explicit preamble, SFD, and PHR instead of a generic byte-overhead fudge
    factor.  Defaults use a compact SUN-FSK-style framing: 4-B preamble, 2-B
    SFD, and 2-B PHR.  The experiment still caps the usable radio packet at
    255 B, matching the manuscript configuration.
    """

    def __init__(self):
        super().__init__(
            name="subghz_fsk_raw",
            max_payload_bytes=255,
            default_tx_dbm=14.0,
            default_pl_ref_m=100.0,
            default_pl_ref_loss_db=101.51,
            default_pl_gamma=3.0,
            default_pl_sigma=3.57,
            default_collision_th_db=1.0,
            default_channel_count=16,
            default_bitrate_bps=50_000.0,
            default_phy_overhead_bytes=0,
            default_sensitivity_dbm=-109.0,
        )

    @staticmethod
    def _airtime(bitrate_bps: float, pl_bytes: int, preamble_bytes: int, sfd_bytes: int, phr_bytes: int) -> float:
        if bitrate_bps <= 0.0:
            raise ValueError("FSK bitrate must be > 0")
        phy_bytes = int(preamble_bytes) + int(sfd_bytes) + int(phr_bytes)
        if min(preamble_bytes, sfd_bytes, phr_bytes) < 0:
            raise ValueError("FSK PHY field lengths must be >= 0")
        return ((int(pl_bytes) + phy_bytes) * 8.0) / float(bitrate_bps)

    def airtime_seconds(self, sf: int, bw_hz: int, cr_denom: int, pl_bytes: int) -> float:
        return self._airtime(self.default_bitrate_bps, pl_bytes, 4, 2, 2)

    def airtime_seconds_for_args(
        self,
        args: argparse.Namespace,
        sf: int,
        bw_hz: int,
        cr_denom: int,
        pl_bytes: int,
    ) -> float:
        return self._airtime(
            self.effective_bitrate_bps(args),
            pl_bytes,
            int(getattr(args, "fsk_preamble_bytes", 4)),
            int(getattr(args, "fsk_sfd_bytes", 2)),
            int(getattr(args, "fsk_phr_bytes", 2)),
        )


class IEEE802154gRawProfile(RawBitrateProfile):
    """IEEE 802.15.4g / Wi-SUN raw profile.

    The packet-size/rate table gives 2047 B and 300 kbps. The second photo did
    not include a separate 802.15.4g range row, so these defaults use the
    earlier 1 m reference model converted to d0=100 m:
      PL(1m)=31.7 dB, gamma=3.0 -> PL(100m)=91.7 dB.
    """

    def __init__(self):
        super().__init__(
            name="ieee802154g_raw",
            max_payload_bytes=2047,
            default_tx_dbm=14.0,
            default_pl_ref_m=100.0,
            default_pl_ref_loss_db=91.7,
            default_pl_gamma=3.0,
            default_pl_sigma=3.57,
            default_collision_th_db=1.0,
            default_channel_count=129,
            default_bitrate_bps=300_000.0,
            default_phy_overhead_bytes=20,
            default_sensitivity_dbm=-103.0,
        )


class HaLowRawProfile(RawBitrateProfile):
    """IEEE 802.11ah 1-MHz MCS10 profile with OFDM symbol timing.

    Airtime is modeled as the 1-MHz PHY preamble/header plus an integer number
    of OFDM symbols.  For MCS10 with long GI, the defaults are 560 us for the
    preamble+PHY header, 40 us per OFDM symbol, 6 data bits per symbol, and
    22 SERVICE+tail bits.  This corresponds to the 150-kbit/s nominal payload
    rate used by the manuscript while avoiding a generic byte-overhead model.
    """

    def __init__(self):
        super().__init__(
            default_sensitivity_dbm=-105.0,
            name="halow_raw",
            max_payload_bytes=469,  # max MSDU: 511-byte PSDU minus 42-byte MAC/Mesh/FCS
            default_tx_dbm=18.0,
            default_pl_ref_m=100.0,
            default_pl_ref_loss_db=92.5,
            default_pl_gamma=3.76,
            default_pl_sigma=3.57,
            default_collision_th_db=2.0,
            default_channel_count=5,
            default_bitrate_bps=150_000.0,
            default_phy_overhead_bytes=42,
        )

    @staticmethod
    def _airtime(pl_bytes: int, preamble_header_us: float, symbol_us: float, ndbps: int, service_tail_bits: int) -> float:
        if preamble_header_us < 0.0 or symbol_us <= 0.0 or ndbps <= 0 or service_tail_bits < 0:
            raise ValueError("Invalid IEEE 802.11ah PHY timing parameters")
        n_sym = math.ceil((int(service_tail_bits) + 8 * int(pl_bytes)) / float(ndbps))
        return (float(preamble_header_us) + n_sym * float(symbol_us)) * 1e-6

    def airtime_seconds(self, sf: int, bw_hz: int, cr_denom: int, pl_bytes: int) -> float:
        return self._airtime(pl_bytes + 42, 560.0, 40.0, 6, 22)

    def airtime_seconds_for_args(
        self,
        args: argparse.Namespace,
        sf: int,
        bw_hz: int,
        cr_denom: int,
        pl_bytes: int,
    ) -> float:
        mac_bytes = int(getattr(args, "halow_mac_overhead_bytes", 42))
        if mac_bytes < 0:
            raise ValueError("HaLow MAC/Mesh/FCS overhead must not be negative")
        if pl_bytes + mac_bytes > 511:
            raise ValueError(f"HaLow PSDU exceeds 511 B: MSDU={pl_bytes} + MAC/Mesh/FCS={mac_bytes}")
        return self._airtime(
            pl_bytes + mac_bytes,
            float(getattr(args, "halow_preamble_header_us", 560.0)),
            float(getattr(args, "halow_symbol_us", 40.0)),
            int(getattr(args, "halow_ndbps", 6)),
            int(getattr(args, "halow_service_tail_bits", 22)),
        )


_BLE5_CODED_S8_PROFILE = BLE5CodedS8RawProfile()

RADIO_PROFILES = {
    "lora_subghz": LoRaSubGHzProfile(),
    "subghz": LoRaSubGHzProfile(),  # backward-compatible alias
    "lora24": LoRa24Profile(),
    "ble5_coded_s8_raw": _BLE5_CODED_S8_PROFILE,
    "ble5_dle_raw": _BLE5_CODED_S8_PROFILE,  # backward-compatible alias
    "subghz_fsk_raw": SubGHzFSKRawProfile(),
    "ieee802154g_raw": IEEE802154gRawProfile(),
    "halow_raw": HaLowRawProfile(),  # name kept; values represent IEEE 802.11ah
}


def add_radio_profile_args(parser: argparse.ArgumentParser) -> None:
    """Register radio-profile-related CLI flags."""
    profile_choices = [
        "lora_subghz",
        "lora24",
        "ble5_coded_s8_raw",
        "ble5_dle_raw",  # backward-compatible alias for BLE 5 Coded S=8
        "subghz_fsk_raw",
        "ieee802154g_raw",
        "halow_raw",  # name kept; values represent IEEE 802.11ah
    ]
    parser.add_argument(
        "--radio-profile",
        choices=profile_choices,
        default=None,
        help="Radio profile. Prefer this new flag over --lora-radio.",
    )
    parser.add_argument(
        "--lora-radio",
        dest="lora_radio",
        choices=["subghz", "lora24"],
        default=None,
        help="Backward-compatible alias: subghz -> lora_subghz, lora24 -> lora24.",
    )

    # Generic path-loss / attenuation overrides. If omitted, profile defaults are used.
    parser.add_argument("--pl-ref-m", type=float, default=None,
                        help="Path-loss reference distance in meters. Defaults to selected radio profile.")
    parser.add_argument("--pl-ref-loss-db", type=float, default=None,
                        help="Path loss at --pl-ref-m. Defaults to selected radio profile.")
    parser.add_argument("--pl-gamma", type=float, default=None,
                        help="Log-distance path-loss exponent. Defaults to selected radio profile.")

    # Generic raw-bitrate overrides. Used only by *_raw bitrate profiles.
    parser.add_argument("--raw-bitrate-bps", type=float, default=None,
                        help="Override bitrate for raw bitrate profiles.")
    parser.add_argument("--raw-max-payload-bytes", type=int, default=None,
                        help="Override max payload bytes for raw bitrate profiles.")
    parser.add_argument("--raw-phy-overhead-bytes", type=int, default=None,
                        help="Override PHY overhead bytes for raw bitrate profiles.")
    parser.add_argument("--raw-sensitivity-dbm", type=float, default=None,
                        help="Override sensitivity for raw bitrate profiles.")

    # Explicit Sub-GHz FSK PHY framing. These fields are transmitted before the
    # configured FSK packet/PSDU bytes and therefore contribute to airtime.
    parser.add_argument("--fsk-preamble-bytes", type=int, default=4,
                        help="Sub-GHz FSK preamble length in bytes. Default: 4.")
    parser.add_argument("--fsk-sfd-bytes", type=int, default=2,
                        help="Sub-GHz FSK SFD length in bytes. Default: 2.")
    parser.add_argument("--fsk-phr-bytes", type=int, default=2,
                        help="Sub-GHz FSK PHY header length in bytes. Default: 2.")

    # IEEE 802.11ah 1-MHz MCS10 timing parameters. 6 data bits every 40 us
    # corresponds to 150 kbit/s before preamble/header and symbol rounding.
    parser.add_argument("--halow-preamble-header-us", type=float, default=560.0,
                        help="IEEE 802.11ah 1-MHz preamble+PHY-header time in us. Default: 560.")
    parser.add_argument("--halow-symbol-us", type=float, default=40.0,
                        help="IEEE 802.11ah OFDM symbol duration in us (long GI). Default: 40.")
    parser.add_argument("--halow-ndbps", type=int, default=6,
                        help="IEEE 802.11ah MCS10 data bits per OFDM symbol. Default: 6.")
    parser.add_argument("--halow-service-tail-bits", type=int, default=22,
                        help="IEEE 802.11ah SERVICE+tail bits before symbol rounding. Default: 22.")
    parser.add_argument("--halow-mac-overhead-bytes", type=int, default=42,
                        help="IEEE 802.11ah MAC header (32), Mesh Control (6), FCS (4): default 42 B, added to every frame airtime.")


def normalize_radio_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve selected radio profile and fill effective defaults.

    Existing run files using `--lora-radio lora24` still work.
    Existing sub-GHz run files that omit both flags default to lora_subghz.
    """
    if args.radio_profile is None:
        if args.lora_radio == "lora24":
            args.radio_profile = "lora24"
        else:
            args.radio_profile = "lora_subghz"

    profile = RADIO_PROFILES.get(args.radio_profile)
    if profile is None:
        raise SystemExit(f"Unknown radio profile: {args.radio_profile}")

    # Keep args.lora_radio for old metadata/scripts.
    args.lora_radio = "lora24" if args.radio_profile == "lora24" else "subghz"

    # Reference-distance compatibility. Current literature-based comparisons use
    # d0=100 m by default; old configs can still explicitly request 40m.
    if getattr(args, "pl_ref_m", None) is None:
        if getattr(args, "pl_ref", None) == "40m":
            args.pl_ref_m = 40.0
        elif getattr(args, "pl_ref", None) == "100m":
            args.pl_ref_m = 100.0
        else:
            args.pl_ref_m = profile.default_pl_ref_m

    # Fill generic/effective path-loss fields from overrides or profile defaults.
    if getattr(args, "pl_ref_loss_db", None) is None:
        args.pl_ref_loss_db = _coalesce(getattr(args, "pl_40m_lpld0", None), profile.default_pl_ref_loss_db)
    if getattr(args, "pl_gamma", None) is None:
        args.pl_gamma = _coalesce(getattr(args, "pl_40m_gamma", None), profile.default_pl_gamma)
    if getattr(args, "pl_sigma", None) is None:
        args.pl_sigma = profile.default_pl_sigma

    # Fill TX/collision/channel defaults from profile if user omitted them.
    if getattr(args, "tx_dbm", None) is None:
        args.tx_dbm = profile.default_tx_dbm
    if getattr(args, "collision_th_db", None) is None:
        args.collision_th_db = profile.default_collision_th_db
    if getattr(args, "num_channels", None) is None:
        args.num_channels = profile.default_channel_count

    # Mirror generic effective values into legacy fields so older code/metadata keeps working.
    if abs(float(args.pl_ref_m) - 100.0) < 1e-9:
        args.pl_ref = "100m"
    elif abs(float(args.pl_ref_m) - 40.0) < 1e-9:
        args.pl_ref = "40m"
    else:
        args.pl_ref = f"{args.pl_ref_m:g}m"
    args.pl_40m_lpld0 = args.pl_ref_loss_db
    args.pl_40m_gamma = args.pl_gamma

    # Raw profile max payload override. This only changes validation, not class-level defaults.
    effective_max_payload = int(_coalesce(getattr(args, "raw_max_payload_bytes", None), profile.max_payload_bytes))
    args.radio_max_payload_bytes = effective_max_payload
    if args.radio_profile == "halow_raw" and effective_max_payload + int(getattr(args, "halow_mac_overhead_bytes", 42)) > 511:
        raise SystemExit("IEEE 802.11ah 1-MHz MCS10 PSDU limit is 511 B including MAC/Mesh/FCS overhead")

    # Protocol header includes 2-byte session ID in every DATA/REPAIR packet.
    # Compact: 1 T/E/V + 2 session + 2 chunk ID + 2 total chunks + 1 length + 2 CRC = 10 B.
    # Large (>255-B) profile uses 2-byte length, hence 11 B.
    args.data_session_id_bytes = 2
    min_data_header_bytes = 10 if effective_max_payload < 256 else 11
    if getattr(args, "data_header_bytes", None) is None:
        args.data_header_bytes = min_data_header_bytes
    if int(args.data_header_bytes) < min_data_header_bytes:
        raise SystemExit(
            f"--data-header-bytes={args.data_header_bytes} omits the mandatory 2-byte "
            f"DATA/REPAIR Session ID; minimum is {min_data_header_bytes} B"
        )
    if getattr(args, "miss_header_bytes", None) is None:
        args.miss_header_bytes = profile.default_miss_header_bytes
    if getattr(args, "miss_max_packet_bytes", None) is None:
        args.miss_max_packet_bytes = profile.default_miss_max_packet_bytes

    # Keep old miss_hdr_bytes field synchronized for existing code paths.
    args.miss_hdr_bytes = args.miss_header_bytes

    if getattr(args, "data_pl_bytes", None) is None:
        args.data_pl_bytes = effective_max_payload - int(args.data_header_bytes)
    if args.data_pl_bytes <= 0:
        raise SystemExit("Effective --data-pl-bytes must be > 0. Check radio max payload and DATA header size.")
    args.data_packet_bytes = int(args.data_pl_bytes) + int(args.data_header_bytes)

    args.miss_bitmap_capacity_bytes = int(args.miss_max_packet_bytes) - int(args.miss_header_bytes)
    if args.miss_bitmap_capacity_bytes < 0:
        raise SystemExit("MISS header is larger than max MISS packet size")
    args.miss_bitmap_capacity_chunks = int(args.miss_bitmap_capacity_bytes) * 8

    # Validate configured packet sizes against profile max payload.
    payload_fields = [
        ("adv_pl_bytes", "--adv-pl-bytes"),
        ("upd_pl_bytes", "--upd-pl-bytes"),
    ]
    for field, label in payload_fields:
        value = int(getattr(args, field))
        if value > effective_max_payload:
            raise SystemExit(
                f"{label}={value} B exceeds max payload for {args.radio_profile}: "
                f"{effective_max_payload} B"
            )

    if int(args.data_packet_bytes) > effective_max_payload:
        raise SystemExit(
            f"DATA packet size data_pl_bytes + data_header_bytes = {args.data_packet_bytes} B "
            f"exceeds max payload for {args.radio_profile}: {effective_max_payload} B"
        )

    if getattr(args, "miss_len_mode", "auto") == "fixed":
        value = int(getattr(args, "miss_pl_bytes"))
        if value > int(args.miss_max_packet_bytes):
            raise SystemExit(
                f"--miss-pl-bytes={value} B exceeds --miss-max-packet-bytes={args.miss_max_packet_bytes} B"
            )
        if value > effective_max_payload:
            raise SystemExit(
                f"--miss-pl-bytes={value} B exceeds max payload for {args.radio_profile}: "
                f"{effective_max_payload} B"
            )


    return args


def get_radio_profile(args: argparse.Namespace) -> RadioProfile:
    profile_name = getattr(args, "radio_profile", None) or "lora_subghz"
    try:
        profile = RADIO_PROFILES[profile_name]
    except KeyError as exc:
        raise SystemExit(f"Unknown radio profile: {profile_name}") from exc
    return profile
