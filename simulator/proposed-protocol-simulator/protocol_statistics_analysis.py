#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical comparison of Proposed vs. Deluge dissemination simulations.

Inputs
------
1) Proposed update_cdf_all_samples.csv
2) Deluge update_cdf_all_samples.csv
3) Proposed run_energy_summary.csv (preferred) or node_energy_summary.csv
4) Deluge run_energy_summary.csv (preferred) or node_energy_summary.csv

Main statistics
---------------
- 90% and 100% completion probability with 95% Wilson confidence intervals.
- Kaplan-Meier median cycles to 90% and 100% completion; runs that do not
  reach the threshold by the horizon are right-censored.
- Bootstrap 95% confidence intervals for the Kaplan-Meier median.
- Mean and median per-ED energy with bootstrap 95% confidence intervals.
- Exact paired McNemar tests for Proposed vs. Deluge completion probability.
- Paired Wilcoxon signed-rank tests for Proposed vs. Deluge per-run energy.
- Holm correction for the family of pairwise tests across all evaluated
  technology/payload combinations.

Outputs
-------
- statistics_protocol_summary.csv
- statistics_protocol_comparisons.csv
- statistics_results.json
- statistics_report.txt
- statistics_table.tex

The independent experimental unit is the terrain/run, not an individual ED.
Energy comparisons therefore use each run's mean per-ED energy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception as exc:  # pragma: no cover
    wilcoxon = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


TECH_ORDER = [
    "lora24_sf5",
    "lora24_sf5_csma",
    "lora24_sf7",
    "lora24_sf7_csma",
    "subghz_fsk_raw",
    "subghz_fsk_csma",
    "halow_raw",
]

TECH_LABELS = {
    "lora24_sf5": "LoRa2.4 SF5",
    "lora24_sf5_csma": "LoRa2.4 SF5 CAD",
    "lora24_sf7": "LoRa2.4 SF7",
    "lora24_sf7_csma": "LoRa2.4 SF7 CAD",
    "subghz_fsk_raw": "Sub-GHz FSK",
    "subghz_fsk_csma": "Sub-GHz FSK CSMA",
    "halow_raw": "IEEE 802.11ah",
}


def canonical_technology(value: object) -> str:
    s = str(value).strip()
    while s.startswith("deluge_"):
        s = s[len("deluge_"):]
    aliases = {
        "ieee80211ah": "halow_raw",
        "ieee_80211ah": "halow_raw",
        "80211ah": "halow_raw",
        "halow": "halow_raw",
    }
    return aliases.get(s, s)


def normalize_testbed(value: object) -> str:
    s = str(value).strip().replace("\\", "/")
    name = s.rsplit("/", 1)[-1]
    if name.lower().endswith(".csv"):
        name = name[:-4]
    return name


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def km_median(times: Sequence[float], events: Sequence[int]) -> float:
    """Kaplan-Meier median: first event time at which S(t) <= 0.5."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    valid = np.isfinite(t) & np.isin(e, [0, 1])
    t = t[valid]
    e = e[valid]
    if len(t) == 0:
        return math.nan

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    e = e[order]
    survival = 1.0

    for current in np.unique(t):
        at_risk = int(np.sum(t >= current))
        d = int(np.sum((t == current) & (e == 1)))
        if at_risk <= 0:
            continue
        if d > 0:
            survival *= 1.0 - d / at_risk
            if survival <= 0.5:
                return float(current)

    return math.inf


def km_bootstrap_median_ci(
    times: Sequence[float],
    events: Sequence[int],
    rng: np.random.Generator,
    n_boot: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    valid = np.isfinite(t) & np.isin(e, [0, 1])
    t = t[valid]
    e = e[valid]
    n = len(t)
    if n == 0:
        return math.nan, math.nan, 0.0

    medians: List[float] = []
    finite_count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = km_median(t[idx], e[idx])
        if math.isfinite(m):
            finite_count += 1
            medians.append(m)

    frac = finite_count / n_boot if n_boot else 0.0
    if not medians:
        return math.nan, math.nan, frac

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(np.asarray(medians, dtype=float), [alpha, 1.0 - alpha])
    return float(lo), float(hi), frac


def bootstrap_ci(
    values: Sequence[float],
    statistic: str,
    rng: np.random.Generator,
    n_boot: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan, math.nan
    fn = np.mean if statistic == "mean" else np.median
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = x[rng.integers(0, len(x), size=len(x))]
        stats[i] = fn(sample)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value = exact binomial test on discordant pairs."""
    n = int(b + c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * lower)


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return out.tolist()

    ordered = valid_idx[np.argsort(p[valid_idx])]
    m = len(ordered)
    running = 0.0
    for rank, idx in enumerate(ordered):
        adj = (m - rank) * p[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)
    return out.tolist()


def p_fmt(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def ci_pct(p: float, lo: float, hi: float) -> str:
    if not all(math.isfinite(v) for v in [p, lo, hi]):
        return "--"
    return f"{100*p:.1f} [{100*lo:.1f},{100*hi:.1f}]"


def ci_num(value: float, lo: float, hi: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return "NR"
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]"


def prepare_cdf(path: Path, protocol: str, node_count: int, data_sizes: Sequence[int], horizon: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "data_size_bytes", "technology", "testbed", "node_count",
        "threshold", "cycles_to_threshold", "status"
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing CDF columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["protocol"] = protocol
    df["technology_canonical"] = df["technology"].map(canonical_technology)
    df["testbed_key"] = df["testbed"].map(normalize_testbed)
    df["node_count"] = pd.to_numeric(df["node_count"], errors="coerce")
    df["data_size_bytes"] = pd.to_numeric(df["data_size_bytes"], errors="coerce")
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["cycles_to_threshold"] = pd.to_numeric(df["cycles_to_threshold"], errors="coerce")

    df = df[
        (df["node_count"] == node_count)
        & (df["data_size_bytes"].isin(list(data_sizes)))
        & (df["technology_canonical"].isin(TECH_ORDER))
        & (df["threshold"].apply(lambda x: abs(x - 0.9) < 1e-9 or abs(x - 1.0) < 1e-9))
    ].copy()

    df["event"] = (df["status"].astype(str) == "reached").astype(int)
    bad_reached = (df["event"] == 1) & (~np.isfinite(df["cycles_to_threshold"]))
    if bad_reached.any():
        raise SystemExit(f"{path}: {int(bad_reached.sum())} reached rows have invalid cycles_to_threshold.")
    df["time"] = np.where(df["event"] == 1, df["cycles_to_threshold"], float(horizon))

    # Optional rep key; if no rep column, use blank.
    if "rep" not in df.columns:
        df["rep"] = ""
    df["rep_key"] = df["rep"].fillna("").astype(str)
    df["pair_key"] = df["testbed_key"] + "||" + df["rep_key"]
    return df


def prepare_energy(path: Path, protocol: str, node_count: int, data_sizes: Sequence[int]) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    for col in ["data_size_bytes", "technology", "node_count"]:
        if col not in df.columns:
            raise SystemExit(f"{path}: missing required energy column {col!r}")

    if "testbed" not in df.columns:
        raise SystemExit(f"{path}: energy file needs a 'testbed' column for paired terrain analysis.")

    df["technology_canonical"] = df["technology"].map(canonical_technology)
    df["testbed_key"] = df["testbed"].map(normalize_testbed)
    df["node_count"] = pd.to_numeric(df["node_count"], errors="coerce")
    df["data_size_bytes"] = pd.to_numeric(df["data_size_bytes"], errors="coerce")
    if "rep" not in df.columns:
        df["rep"] = ""
    df["rep_key"] = df["rep"].fillna("").astype(str)
    df["pair_key"] = df["testbed_key"] + "||" + df["rep_key"]

    df = df[
        (df["node_count"] == node_count)
        & (df["data_size_bytes"].isin(list(data_sizes)))
        & (df["technology_canonical"].isin(TECH_ORDER))
    ].copy()

    # Preferred: run-level summaries.
    candidates = [
        "avg_per_node_total_energy_with_mcu_mj",   # Proposed run_energy_summary.csv
        "mean_total_energy_with_mcu_mj",           # Deluge run_energy_summary.csv
        "total_mj",                                # Proposed node_energy_summary.csv
        "total_energy_with_mcu_mj",                # Deluge node_energy_summary.csv
        "total_radio_plus_mcu_energy_mj",
    ]
    energy_col = next((c for c in candidates if c in df.columns), None)
    if energy_col is None:
        raise SystemExit(
            f"{path}: could not find a supported total-energy column. "
            f"Tried: {', '.join(candidates)}"
        )

    df["energy_mj"] = pd.to_numeric(df[energy_col], errors="coerce")
    df = df[np.isfinite(df["energy_mj"])].copy()

    # If per-node input was supplied, collapse to one mean per run/terrain.
    per_node_cols = {"total_mj", "total_energy_with_mcu_mj", "total_radio_plus_mcu_energy_mj"}
    if energy_col in per_node_cols:
        group_cols = [
            "protocol", "data_size_bytes", "technology_canonical",
            "node_count", "testbed_key", "rep_key", "pair_key"
        ]
        df["protocol"] = protocol
        out = (
            df.groupby(group_cols, as_index=False, dropna=False)["energy_mj"]
            .mean()
        )
    else:
        df["protocol"] = protocol
        out = df[
            [
                "protocol", "data_size_bytes", "technology_canonical",
                "node_count", "testbed_key", "rep_key", "pair_key", "energy_mj"
            ]
        ].copy()

    # Convert mJ -> J for the paper table.
    out["energy_j"] = out["energy_mj"] / 1000.0
    return out


def summarize_protocol(
    cdf: pd.DataFrame,
    energy: pd.DataFrame,
    protocol: str,
    data_size: int,
    tech: str,
    horizon: float,
    rng: np.random.Generator,
    n_boot: int,
    confidence: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "protocol": protocol,
        "data_size_bytes": int(data_size),
        "technology": tech,
        "tech_label": TECH_LABELS.get(tech, tech),
    }

    for threshold, tag in [(0.9, "90"), (1.0, "100")]:
        sub = cdf[
            (cdf["protocol"] == protocol)
            & (cdf["data_size_bytes"] == data_size)
            & (cdf["technology_canonical"] == tech)
            & ((cdf["threshold"] - threshold).abs() < 1e-9)
        ].copy()

        valid = sub[sub["status"].isin(["reached", "not_reached"])].copy()
        n = len(valid)
        successes = int((valid["status"] == "reached").sum())
        failures = int((valid["status"] == "not_reached").sum())
        p = successes / n if n else math.nan
        lo, hi = wilson_ci(successes, n, confidence)

        median = km_median(valid["time"].to_numpy(), valid["event"].to_numpy()) if n else math.nan
        med_lo, med_hi, estimable_frac = km_bootstrap_median_ci(
            valid["time"].to_numpy(),
            valid["event"].to_numpy(),
            rng,
            n_boot,
            confidence,
        ) if n else (math.nan, math.nan, 0.0)

        row.update({
            f"n_{tag}": n,
            f"reached_{tag}": successes,
            f"failed_{tag}": failures,
            f"completion_prob_{tag}": p,
            f"completion_ci_low_{tag}": lo,
            f"completion_ci_high_{tag}": hi,
            f"failure_prob_{tag}": 1.0 - p if math.isfinite(p) else math.nan,
            f"km_median_cycles_{tag}": median,
            f"km_median_ci_low_{tag}": med_lo,
            f"km_median_ci_high_{tag}": med_hi,
            f"km_boot_estimable_fraction_{tag}": estimable_frac,
        })

    esub = energy[
        (energy["protocol"] == protocol)
        & (energy["data_size_bytes"] == data_size)
        & (energy["technology_canonical"] == tech)
    ]["energy_j"].to_numpy(dtype=float)
    esub = esub[np.isfinite(esub)]

    if len(esub):
        mean_e = float(np.mean(esub))
        median_e = float(np.median(esub))
        q1, q3 = np.quantile(esub, [0.25, 0.75])
        mean_lo, mean_hi = bootstrap_ci(esub, "mean", rng, n_boot, confidence)
        med_lo, med_hi = bootstrap_ci(esub, "median", rng, n_boot, confidence)
    else:
        mean_e = median_e = q1 = q3 = math.nan
        mean_lo = mean_hi = med_lo = med_hi = math.nan

    row.update({
        "energy_runs": int(len(esub)),
        "energy_mean_j_per_ed": mean_e,
        "energy_mean_ci_low_j": mean_lo,
        "energy_mean_ci_high_j": mean_hi,
        "energy_median_j_per_ed": median_e,
        "energy_median_ci_low_j": med_lo,
        "energy_median_ci_high_j": med_hi,
        "energy_q1_j": float(q1) if math.isfinite(q1) else math.nan,
        "energy_q3_j": float(q3) if math.isfinite(q3) else math.nan,
    })
    return row


def paired_comparison(
    proposed_cdf: pd.DataFrame,
    deluge_cdf: pd.DataFrame,
    proposed_energy: pd.DataFrame,
    deluge_energy: pd.DataFrame,
    data_size: int,
    tech: str,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "data_size_bytes": int(data_size),
        "technology": tech,
        "tech_label": TECH_LABELS.get(tech, tech),
    }

    for threshold, tag in [(0.9, "90"), (1.0, "100")]:
        p = proposed_cdf[
            (proposed_cdf["data_size_bytes"] == data_size)
            & (proposed_cdf["technology_canonical"] == tech)
            & ((proposed_cdf["threshold"] - threshold).abs() < 1e-9)
            & (proposed_cdf["status"].isin(["reached", "not_reached"]))
        ][["pair_key", "event"]].drop_duplicates("pair_key")

        d = deluge_cdf[
            (deluge_cdf["data_size_bytes"] == data_size)
            & (deluge_cdf["technology_canonical"] == tech)
            & ((deluge_cdf["threshold"] - threshold).abs() < 1e-9)
            & (deluge_cdf["status"].isin(["reached", "not_reached"]))
        ][["pair_key", "event"]].drop_duplicates("pair_key")

        pairs = p.merge(d, on="pair_key", suffixes=("_proposed", "_deluge"))
        b = int(((pairs["event_proposed"] == 1) & (pairs["event_deluge"] == 0)).sum())
        c = int(((pairs["event_proposed"] == 0) & (pairs["event_deluge"] == 1)).sum())
        out.update({
            f"paired_runs_{tag}": int(len(pairs)),
            f"mcnemar_b_proposed_only_{tag}": b,
            f"mcnemar_c_deluge_only_{tag}": c,
            f"mcnemar_p_raw_{tag}": exact_mcnemar_p(b, c),
        })

    pe = proposed_energy[
        (proposed_energy["data_size_bytes"] == data_size)
        & (proposed_energy["technology_canonical"] == tech)
    ][["pair_key", "energy_j"]].drop_duplicates("pair_key")

    de = deluge_energy[
        (deluge_energy["data_size_bytes"] == data_size)
        & (deluge_energy["technology_canonical"] == tech)
    ][["pair_key", "energy_j"]].drop_duplicates("pair_key")

    pairs_e = pe.merge(de, on="pair_key", suffixes=("_proposed", "_deluge"))
    out["paired_energy_runs"] = int(len(pairs_e))

    if len(pairs_e):
        diffs = pairs_e["energy_j_proposed"].to_numpy() - pairs_e["energy_j_deluge"].to_numpy()
        out["energy_mean_diff_j_proposed_minus_deluge"] = float(np.mean(diffs))
        out["energy_median_diff_j_proposed_minus_deluge"] = float(np.median(diffs))

        if np.allclose(diffs, 0.0):
            pval = 1.0
            statistic = 0.0
        else:
            if wilcoxon is None:
                raise SystemExit(
                    "SciPy is required for the paired Wilcoxon test. "
                    "Install it with: python3 -m pip install scipy\n"
                    f"Import error: {SCIPY_IMPORT_ERROR}"
                )
            res = wilcoxon(
                pairs_e["energy_j_proposed"].to_numpy(),
                pairs_e["energy_j_deluge"].to_numpy(),
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(res.statistic)
            pval = float(res.pvalue)
        out["wilcoxon_energy_statistic"] = statistic
        out["wilcoxon_energy_p_raw"] = pval
    else:
        out["energy_mean_diff_j_proposed_minus_deluge"] = math.nan
        out["energy_median_diff_j_proposed_minus_deluge"] = math.nan
        out["wilcoxon_energy_statistic"] = math.nan
        out["wilcoxon_energy_p_raw"] = math.nan

    return out


def add_holm_adjustments(comp_df: pd.DataFrame) -> pd.DataFrame:
    out = comp_df.copy()
    for raw_col, adj_col in [
        ("mcnemar_p_raw_90", "mcnemar_p_holm_90"),
        ("mcnemar_p_raw_100", "mcnemar_p_holm_100"),
        ("wilcoxon_energy_p_raw", "wilcoxon_energy_p_holm"),
    ]:
        out[adj_col] = holm_adjust(pd.to_numeric(out[raw_col], errors="coerce").to_numpy())
    return out


def make_text_report(summary_df: pd.DataFrame, comp_df: pd.DataFrame, node_count: int, horizon: int) -> str:
    lines: List[str] = []
    lines.append("PROPOSED VS DELUGE STATISTICAL SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Node count: {node_count}")
    lines.append(f"Observation horizon: {horizon} cycles")
    lines.append("95% Wilson CIs: completion probability")
    lines.append("Kaplan-Meier + bootstrap CIs: completion time")
    lines.append("Bootstrap CIs: per-run mean energy/ED")
    lines.append("McNemar exact test: paired completion outcome")
    lines.append("Wilcoxon signed-rank test: paired run-level energy")
    lines.append("Reported pairwise p-values use Holm correction across all tested")
    lines.append("technology/payload comparisons within each test family.")
    lines.append("")

    for ds in sorted(summary_df["data_size_bytes"].unique()):
        lines.append(f"Payload: {int(ds):,} B")
        lines.append("-" * 78)
        for tech in TECH_ORDER:
            s = summary_df[
                (summary_df["data_size_bytes"] == ds)
                & (summary_df["technology"] == tech)
            ]
            c = comp_df[
                (comp_df["data_size_bytes"] == ds)
                & (comp_df["technology"] == tech)
            ]
            if s.empty:
                continue
            lines.append(TECH_LABELS.get(tech, tech))
            for protocol in ["Proposed", "Deluge"]:
                r = s[s["protocol"] == protocol]
                if r.empty:
                    continue
                r = r.iloc[0]
                c90 = ci_pct(
                    r["completion_prob_90"],
                    r["completion_ci_low_90"],
                    r["completion_ci_high_90"],
                )
                c100 = ci_pct(
                    r["completion_prob_100"],
                    r["completion_ci_low_100"],
                    r["completion_ci_high_100"],
                )
                km90 = ci_num(
                    r["km_median_cycles_90"],
                    r["km_median_ci_low_90"],
                    r["km_median_ci_high_90"],
                    0,
                )
                km100 = ci_num(
                    r["km_median_cycles_100"],
                    r["km_median_ci_low_100"],
                    r["km_median_ci_high_100"],
                    0,
                )
                en = ci_num(
                    r["energy_mean_j_per_ed"],
                    r["energy_mean_ci_low_j"],
                    r["energy_mean_ci_high_j"],
                    3,
                )
                lines.append(
                    f"  {protocol:8s} | C90 {c90:24s} | C100 {c100:24s} "
                    f"| KM90 {km90:16s} | KM100 {km100:16s} | E {en} J/ED"
                )
            if not c.empty:
                c = c.iloc[0]
                lines.append(
                    "  paired tests | "
                    f"McNemar90 Holm p={p_fmt(c['mcnemar_p_holm_90'])}; "
                    f"McNemar100 Holm p={p_fmt(c['mcnemar_p_holm_100'])}; "
                    f"Energy Wilcoxon Holm p={p_fmt(c['wilcoxon_energy_p_holm'])}; "
                    f"nE={int(c['paired_energy_runs'])}"
                )
            lines.append("")
    return "\n".join(lines)


def make_latex_table(summary_df: pd.DataFrame, comp_df: pd.DataFrame, node_count: int, horizon: int) -> str:
    """Generate the compact paper table with one row per protocol."""
    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Statistical summary of the proposed protocol and Deluge for "
        + f"{node_count}"
        + r" EDs. Completion percentages are shown with 95\% Wilson confidence intervals; "
          r"completion times are Kaplan--Meier medians with 95\% bootstrap confidence intervals; "
          r"energy is the mean per-ED energy across terrain runs with 95\% bootstrap confidence intervals. "
          r"Runs not reaching the corresponding threshold within "
        + f"{horizon}"
        + r" cycles are treated as right-censored.}"
    )
    lines.append(r"\label{tab:protocol-statistics}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}llcccccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Radio} & "
        r"\textbf{Protocol} & "
        r"\makecell[c]{\textbf{90\% reached}\\\textbf{(\%) [95\% CI]}} & "
        r"\makecell[c]{\textbf{100\% reached}\\\textbf{(\%) [95\% CI]}} & "
        r"\makecell[c]{\textbf{Median cycles}\\\textbf{to 90\% [95\% CI]}} & "
        r"\makecell[c]{\textbf{Median cycles}\\\textbf{to 100\% [95\% CI]}} & "
        r"\makecell[c]{\textbf{Energy/ED}\\\textbf{(J) [95\% CI]}} & "
        r"\makecell[c]{\textbf{Proposed vs. Deluge}\\\textbf{Holm-adjusted $p$}} \\"
    )
    lines.append(r"\midrule")

    data_sizes = sorted(summary_df["data_size_bytes"].dropna().astype(int).unique())

    for ds_i, ds in enumerate(data_sizes):
        lines.append(r"\multicolumn{8}{c}{\textit{" + f"{ds:,}" + r" B update}} \\")
        lines.append(r"\midrule")

        for tech in TECH_ORDER:
            subset = summary_df[
                (summary_df["data_size_bytes"] == ds)
                & (summary_df["technology"] == tech)
            ]
            if subset.empty:
                continue

            comp = comp_df[
                (comp_df["data_size_bytes"] == ds)
                & (comp_df["technology"] == tech)
            ]

            if comp.empty:
                comparison_text = "--"
            else:
                c = comp.iloc[0]
                comparison_text = (
                    r"$p_{90}=" + p_fmt(float(c["mcnemar_p_holm_90"])) +
                    r"$; $p_{100}=" + p_fmt(float(c["mcnemar_p_holm_100"])) +
                    r"$; $p_E=" + p_fmt(float(c["wilcoxon_energy_p_holm"])) + r"$"
                )

            first = True
            for protocol in ["Proposed", "Deluge"]:
                row = subset[subset["protocol"] == protocol]
                if row.empty:
                    continue
                r = row.iloc[0]

                reached90 = ci_pct(
                    r["completion_prob_90"],
                    r["completion_ci_low_90"],
                    r["completion_ci_high_90"],
                )
                reached100 = ci_pct(
                    r["completion_prob_100"],
                    r["completion_ci_low_100"],
                    r["completion_ci_high_100"],
                )
                km90 = ci_num(
                    r["km_median_cycles_90"],
                    r["km_median_ci_low_90"],
                    r["km_median_ci_high_90"],
                    0,
                )
                km100 = ci_num(
                    r["km_median_cycles_100"],
                    r["km_median_ci_low_100"],
                    r["km_median_ci_high_100"],
                    0,
                )
                energy = ci_num(
                    r["energy_mean_j_per_ed"],
                    r["energy_mean_ci_low_j"],
                    r["energy_mean_ci_high_j"],
                    3,
                )

                radio_cell = latex_escape(TECH_LABELS.get(tech, tech)) if first else ""
                compare_cell = comparison_text if first else ""

                lines.append(
                    f"{radio_cell} & {protocol} & "
                    f"{reached90} & {reached100} & "
                    f"{km90} & {km100} & {energy} & "
                    f"{compare_cell} \\\\"
                )
                first = False

            lines.append(r"\addlinespace[1pt]")

        if ds_i != len(data_sizes) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\begin{minipage}{0.99\textwidth}")
    lines.append(r"\footnotesize")
    lines.append(
        r"\textit{Note:} Failure percentage is $100\%$ minus the corresponding reached percentage. "
        r"``NR'' indicates that the Kaplan--Meier median was not reached within the observation horizon. "
        r"$p_{90}$ and $p_{100}$ are exact paired McNemar tests for completion at the respective threshold, "
        r"and $p_E$ is the paired Wilcoxon signed-rank test for run-level mean per-ED energy."
    )
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def make_compact_table_rows(summary_df: pd.DataFrame, comp_df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable table-shaped CSV matching the LaTeX table."""
    rows = []
    for ds in sorted(summary_df["data_size_bytes"].dropna().astype(int).unique()):
        for tech in TECH_ORDER:
            subset = summary_df[
                (summary_df["data_size_bytes"] == ds)
                & (summary_df["technology"] == tech)
            ]
            if subset.empty:
                continue

            comp = comp_df[
                (comp_df["data_size_bytes"] == ds)
                & (comp_df["technology"] == tech)
            ]
            if comp.empty:
                comparison = "--"
            else:
                c = comp.iloc[0]
                comparison = (
                    f"p90={p_fmt(float(c['mcnemar_p_holm_90']))}; "
                    f"p100={p_fmt(float(c['mcnemar_p_holm_100']))}; "
                    f"pE={p_fmt(float(c['wilcoxon_energy_p_holm']))}"
                )

            for protocol in ["Proposed", "Deluge"]:
                x = subset[subset["protocol"] == protocol]
                if x.empty:
                    continue
                r = x.iloc[0]
                rows.append({
                    "Payload (B)": int(ds),
                    "Radio": TECH_LABELS.get(tech, tech),
                    "Protocol": protocol,
                    "90% reached (%) [95% CI]": ci_pct(
                        r["completion_prob_90"],
                        r["completion_ci_low_90"],
                        r["completion_ci_high_90"],
                    ),
                    "100% reached (%) [95% CI]": ci_pct(
                        r["completion_prob_100"],
                        r["completion_ci_low_100"],
                        r["completion_ci_high_100"],
                    ),
                    "Failure at 90% (%)": (
                        100.0 * r["failure_prob_90"]
                        if math.isfinite(r["failure_prob_90"]) else math.nan
                    ),
                    "Failure at 100% (%)": (
                        100.0 * r["failure_prob_100"]
                        if math.isfinite(r["failure_prob_100"]) else math.nan
                    ),
                    "Median cycles to 90% (95% CI)": ci_num(
                        r["km_median_cycles_90"],
                        r["km_median_ci_low_90"],
                        r["km_median_ci_high_90"],
                        0,
                    ),
                    "Median cycles to 100% (95% CI)": ci_num(
                        r["km_median_cycles_100"],
                        r["km_median_ci_low_100"],
                        r["km_median_ci_high_100"],
                        0,
                    ),
                    "Energy/ED J (95% CI)": ci_num(
                        r["energy_mean_j_per_ed"],
                        r["energy_mean_ci_low_j"],
                        r["energy_mean_ci_high_j"],
                        3,
                    ),
                    "Proposed vs. Deluge": comparison if protocol == "Proposed" else "",
                })
    return pd.DataFrame(rows)


def json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate Proposed-vs-Deluge statistical summary, tests, CSV/JSON/text outputs, and a LaTeX table stub."
    )
    ap.add_argument("--proposed-cdf", required=True, help="Proposed update_cdf_all_samples.csv")
    ap.add_argument("--deluge-cdf", required=True, help="Deluge update_cdf_all_samples.csv")
    ap.add_argument("--proposed-energy", required=True, help="Proposed run_energy_summary.csv (preferred) or node_energy_summary.csv")
    ap.add_argument("--deluge-energy", required=True, help="Deluge run_energy_summary.csv (preferred) or node_energy_summary.csv")
    ap.add_argument("--output-dir", default="graphs/protocol_statistics", help="Output directory")
    ap.add_argument("--node-count", type=int, default=300)
    ap.add_argument("--data-sizes", type=int, nargs="+", default=[15850, 177894])
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--expected-runs", type=int, default=50)
    ap.add_argument("--bootstrap-reps", type=int, default=10000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    if wilcoxon is None:
        raise SystemExit(
            "SciPy is required for the Wilcoxon paired test. "
            "Install it with: python3 -m pip install scipy\n"
            f"Import error: {SCIPY_IMPORT_ERROR}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcdf = prepare_cdf(Path(args.proposed_cdf), "Proposed", args.node_count, args.data_sizes, args.horizon)
    dcdf = prepare_cdf(Path(args.deluge_cdf), "Deluge", args.node_count, args.data_sizes, args.horizon)
    peng = prepare_energy(Path(args.proposed_energy), "Proposed", args.node_count, args.data_sizes)
    deng = prepare_energy(Path(args.deluge_energy), "Deluge", args.node_count, args.data_sizes)

    rng = np.random.default_rng(args.seed)

    summary_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []

    for ds in args.data_sizes:
        for tech in TECH_ORDER:
            for protocol, cdf, eng in [
                ("Proposed", pcdf, peng),
                ("Deluge", dcdf, deng),
            ]:
                summary_rows.append(
                    summarize_protocol(
                        cdf, eng, protocol, ds, tech,
                        args.horizon, rng, args.bootstrap_reps, args.confidence
                    )
                )
            comparison_rows.append(
                paired_comparison(pcdf, dcdf, peng, deng, ds, tech)
            )

    summary_df = pd.DataFrame(summary_rows)
    comp_df = add_holm_adjustments(pd.DataFrame(comparison_rows))

    # Completeness audit.
    audit_messages: List[str] = []
    for ds in args.data_sizes:
        for tech in TECH_ORDER:
            for protocol in ["Proposed", "Deluge"]:
                s = summary_df[
                    (summary_df["data_size_bytes"] == ds)
                    & (summary_df["technology"] == tech)
                    & (summary_df["protocol"] == protocol)
                ]
                if s.empty:
                    audit_messages.append(f"MISSING: {protocol}, {ds} B, {tech}")
                    continue
                r = s.iloc[0]
                for tag in ["90", "100"]:
                    n = int(r[f"n_{tag}"])
                    if n != args.expected_runs:
                        audit_messages.append(
                            f"COUNT WARNING: {protocol}, {ds} B, {tech}, threshold {tag}%: "
                            f"{n} valid runs (expected {args.expected_runs})"
                        )
                ne = int(r["energy_runs"])
                if ne != args.expected_runs:
                    audit_messages.append(
                        f"ENERGY COUNT WARNING: {protocol}, {ds} B, {tech}: "
                        f"{ne} run-level energy records (expected {args.expected_runs})"
                    )

    summary_csv = out_dir / "statistics_protocol_summary.csv"
    comparisons_csv = out_dir / "statistics_protocol_comparisons.csv"
    compact_csv = out_dir / "statistics_table_rows.csv"
    json_path = out_dir / "statistics_results.json"
    report_path = out_dir / "statistics_report.txt"
    latex_path = out_dir / "statistics_table.tex"

    summary_df.to_csv(summary_csv, index=False)
    comp_df.to_csv(comparisons_csv, index=False)
    make_compact_table_rows(summary_df, comp_df).to_csv(compact_csv, index=False)

    report = make_text_report(summary_df, comp_df, args.node_count, args.horizon)
    if audit_messages:
        report += "\n\nAUDIT WARNINGS\n" + "-" * 78 + "\n" + "\n".join(audit_messages)
    else:
        report += "\n\nAUDIT\n" + "-" * 78 + "\nAll requested protocol/technology/payload groups contain the expected number of valid runs."

    report_path.write_text(report + "\n", encoding="utf-8")
    latex_path.write_text(
        make_latex_table(summary_df, comp_df, args.node_count, args.horizon),
        encoding="utf-8"
    )

    payload = {
        "settings": {
            "node_count": args.node_count,
            "data_sizes": list(map(int, args.data_sizes)),
            "horizon_cycles": args.horizon,
            "expected_runs": args.expected_runs,
            "confidence": args.confidence,
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
            "holm_scope": "all technology/payload comparisons within each test family",
        },
        "protocol_summary": [
            {k: json_safe(v) for k, v in row.items()}
            for row in summary_df.to_dict(orient="records")
        ],
        "comparisons": [
            {k: json_safe(v) for k, v in row.items()}
            for row in comp_df.to_dict(orient="records")
        ],
        "audit_warnings": audit_messages,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(report)
    print("\nWrote:")
    for p in [summary_csv, comparisons_csv, compact_csv, json_path, report_path, latex_path]:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
