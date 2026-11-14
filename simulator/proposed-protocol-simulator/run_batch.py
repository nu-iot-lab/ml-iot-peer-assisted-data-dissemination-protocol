#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch runner for the wireless network simulator.

Runs every selected radio technology config over every testbed CSV and every
configured model/data size, with repeat support and parallel workers.

Default output layout:
    output/run_<rep>/size_<bytes>B/<technology_name>/<testbed_stem>/

Example:
    python3 run_batch.py \
      --testbeds-dir testbeds \
      --config-dir config \
      --technologies lora24 ieee802154g_raw halow_raw \
      --reps 2 \
      --max-workers 4

Notes:
- Config files are passed with `--config`, not `@file`, so comments are OK if
  your simulator config loader supports them.
- The runner always appends --testbed, --out-dir, --seed, and
  --model-size-bytes after --config; therefore these values override whatever
  is written in the config file.
- Default DATA/model sizes are defined in DEFAULT_DATA_SIZES_BYTES below.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional


# Edit this list to change the model/data sizes used by default.
DEFAULT_DATA_SIZES_BYTES: List[int] = [
    #46773,
    177894,
    15850,
    #178363,
    #62176,
    #150505,
]


@dataclass(frozen=True)
class BatchJob:
    job_id: int
    rep: int
    data_size_bytes: int
    technology: str
    config_path: pathlib.Path
    testbed_path: pathlib.Path
    out_dir: pathlib.Path
    seed: int


def _splitmix64(x: int) -> int:
    """Deterministic 64-bit mixing function for repeatable per-job seeds."""
    x &= 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def make_job_seed(seed_base: Optional[int], job_id: int) -> int:
    """Create either deterministic or entropy-based seed for a job."""
    if seed_base is None:
        entropy = time.time_ns() ^ os.getpid() ^ random.SystemRandom().getrandbits(64)
        return _splitmix64(entropy ^ job_id)
    return _splitmix64(int(seed_base) ^ (job_id * 0xD1B54A32D192ED03))


def discover_testbeds(testbeds_dir: pathlib.Path, pattern: str, recursive: bool) -> List[pathlib.Path]:
    globber = testbeds_dir.rglob if recursive else testbeds_dir.glob
    return sorted(p for p in globber(pattern) if p.is_file())


def discover_configs(config_dir: pathlib.Path, suffixes: tuple[str, ...] = (".args", ".txt", ".csv")) -> dict[str, pathlib.Path]:
    configs: dict[str, pathlib.Path] = {}
    for p in sorted(config_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue
        configs[p.stem] = p
    return configs


def resolve_technologies(
    available: dict[str, pathlib.Path],
    requested: Optional[List[str]],
    exclude: Optional[List[str]],
) -> List[str]:
    if requested:
        missing = [name for name in requested if name not in available]
        if missing:
            raise SystemExit(
                "Unknown technology config(s): "
                + ", ".join(missing)
                + "\nAvailable: "
                + ", ".join(sorted(available))
            )
        selected = list(requested)
    else:
        selected = sorted(available)

    exclude_set = set(exclude or [])
    selected = [name for name in selected if name not in exclude_set]
    if not selected:
        raise SystemExit("No technologies selected.")
    return selected


def parse_size_token(token: str) -> int:
    """Parse sizes like 46773, 250k, 250KB, 1m, 1MB into bytes.

    Uses decimal units: k/KB = 1000, m/MB = 1_000_000.
    """
    s = token.strip().lower().replace("_", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(b|bytes|kb|k|mb|m)?", s)
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid data size: {token!r}")
    value = float(m.group(1))
    unit = m.group(2) or "b"
    if unit in ("b", "bytes"):
        mult = 1
    elif unit in ("k", "kb"):
        mult = 1000
    elif unit in ("m", "mb"):
        mult = 1_000_000
    else:
        raise argparse.ArgumentTypeError(f"Invalid data size unit in: {token!r}")
    out = int(round(value * mult))
    if out <= 0:
        raise argparse.ArgumentTypeError("Data size must be > 0")
    return out


def size_folder_name(size_bytes: int) -> str:
    return f"size_{size_bytes}B"


def build_jobs(
    reps: int,
    data_sizes: Iterable[int],
    technologies: Iterable[str],
    configs: dict[str, pathlib.Path],
    testbeds: Iterable[pathlib.Path],
    base_out: pathlib.Path,
    seed_base: Optional[int],
) -> List[BatchJob]:
    jobs: List[BatchJob] = []
    job_id = 0
    for rep in range(1, reps + 1):
        for size in data_sizes:
            for tech in technologies:
                for tb in testbeds:
                    out_dir = base_out / f"run_{rep}" / size_folder_name(size) / tech / tb.stem
                    seed = make_job_seed(seed_base, job_id)
                    jobs.append(
                        BatchJob(
                            job_id=job_id,
                            rep=rep,
                            data_size_bytes=size,
                            technology=tech,
                            config_path=configs[tech],
                            testbed_path=tb,
                            out_dir=out_dir,
                            seed=seed,
                        )
                    )
                    job_id += 1
    return jobs


def write_batch_plan(path: pathlib.Path, jobs: List[BatchJob], sim_path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "job_id",
            "rep",
            "data_size_bytes",
            "technology",
            "testbed",
            "config",
            "seed",
            "out_dir",
            "sim_path",
        ])
        for j in jobs:
            w.writerow([
                j.job_id,
                j.rep,
                j.data_size_bytes,
                j.technology,
                str(j.testbed_path),
                str(j.config_path),
                j.seed,
                str(j.out_dir),
                str(sim_path),
            ])


def run_job(job: BatchJob, sim_path: pathlib.Path, python_exe: str, extra_sim_args: List[str], cwd: Optional[pathlib.Path]) -> dict:
    job.out_dir.mkdir(parents=True, exist_ok=True)

    # Values after --config override values inside the config file because argparse keeps the last occurrence.
    cmd = [
        python_exe,
        str(sim_path),
        "--config",
        str(job.config_path),
        "--testbed",
        str(job.testbed_path),
        "--out-dir",
        str(job.out_dir),
        "--seed",
        str(job.seed),
        "--model-size-bytes",
        str(job.data_size_bytes),
        *extra_sim_args,
    ]

    manifest_path = job.out_dir / "batch_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["job_id", job.job_id])
        w.writerow(["rep", job.rep])
        w.writerow(["data_size_bytes", job.data_size_bytes])
        w.writerow(["technology", job.technology])
        w.writerow(["config_path", str(job.config_path)])
        w.writerow(["testbed_path", str(job.testbed_path)])
        w.writerow(["out_dir", str(job.out_dir)])
        w.writerow(["seed", job.seed])
        w.writerow(["command", " ".join(cmd)])

    stdout_path = job.out_dir / "stdout.txt"
    stderr_path = job.out_dir / "stderr.txt"
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, cwd=str(cwd) if cwd else None)
    elapsed = time.time() - start

    return {
        "job_id": job.job_id,
        "rep": job.rep,
        "data_size_bytes": job.data_size_bytes,
        "technology": job.technology,
        "testbed": job.testbed_path.name,
        "seed": job.seed,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "out_dir": str(job.out_dir),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run dist_lora_sim.py over selected data sizes, technology configs, and testbeds.")
    ap.add_argument("--sim-path", default="dist_lora_sim.py", help="Simulator script path.")
    ap.add_argument("--testbeds-dir", default="testbeds", help="Directory containing testbed CSV files.")
    ap.add_argument("--testbed-pattern", default="*.csv", help="Glob pattern for testbed files. Default: *.csv")
    ap.add_argument("--recursive-testbeds", action="store_true", help="Search testbeds directory recursively.")
    ap.add_argument("--config-dir", default="config", help="Directory containing technology config files.")
    ap.add_argument(
        "--technologies",
        nargs="+",
        default=None,
        help="Technology config names to run, without extension. Default: all configs in --config-dir.",
    )
    ap.add_argument(
        "--exclude-technologies",
        nargs="+",
        default=None,
        help="Technology config names to skip, useful when running all except one.",
    )
    ap.add_argument(
        "--data-sizes",
        nargs="+",
        type=parse_size_token,
        default=None,
        help=(
            "Optional data/model sizes in bytes. If omitted, uses DEFAULT_DATA_SIZES_BYTES inside this script. "
            "Examples: --data-sizes 46773 177894 250k 1MB"
        ),
    )
    ap.add_argument("--reps", type=int, default=1, help="Number of repetitions for every size/technology/testbed combination.")
    ap.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() or 1), help="Parallel worker count.")
    ap.add_argument("--base-out", default="output", help="Base output directory.")
    ap.add_argument("--seed-base", type=int, default=None, help="Set for reproducible per-job seeds.")
    ap.add_argument("--python", default=sys.executable, help="Python executable used to run the simulator.")
    ap.add_argument("--workdir", default=None, help="Optional working directory for simulator subprocesses.")
    ap.add_argument("--dry-run", action="store_true", help="Only print planned jobs and write batch_plan.csv; do not run.")
    ap.add_argument(
        "extra_sim_args",
        nargs=argparse.REMAINDER,
        help="Extra simulator args after --. These override config values too, e.g. -- --max-days 10",
    )
    args = ap.parse_args(argv)

    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")

    data_sizes = list(args.data_sizes if args.data_sizes is not None else DEFAULT_DATA_SIZES_BYTES)
    if not data_sizes:
        raise SystemExit("No data sizes selected.")
    if any(s <= 0 for s in data_sizes):
        raise SystemExit("All data sizes must be > 0")

    sim_path = pathlib.Path(args.sim_path)
    testbeds_dir = pathlib.Path(args.testbeds_dir)
    config_dir = pathlib.Path(args.config_dir)
    base_out = pathlib.Path(args.base_out)
    cwd = pathlib.Path(args.workdir) if args.workdir else None

    if not testbeds_dir.exists():
        raise SystemExit(f"Testbeds directory not found: {testbeds_dir}")
    if not config_dir.exists():
        raise SystemExit(f"Config directory not found: {config_dir}")

    testbeds = discover_testbeds(testbeds_dir, args.testbed_pattern, args.recursive_testbeds)
    if not testbeds:
        raise SystemExit(f"No testbeds found in {testbeds_dir} with pattern {args.testbed_pattern!r}")

    configs = discover_configs(config_dir)
    if not configs:
        raise SystemExit(f"No config files found in {config_dir}")

    technologies = resolve_technologies(configs, args.technologies, args.exclude_technologies)

    # argparse.REMAINDER keeps the leading "--" separator if present; remove it.
    extra_sim_args = list(args.extra_sim_args or [])
    if extra_sim_args and extra_sim_args[0] == "--":
        extra_sim_args = extra_sim_args[1:]

    jobs = build_jobs(args.reps, data_sizes, technologies, configs, testbeds, base_out, args.seed_base)
    base_out.mkdir(parents=True, exist_ok=True)
    write_batch_plan(base_out / "batch_plan.csv", jobs, sim_path)

    print(f"Discovered {len(testbeds)} testbed(s): {[p.name for p in testbeds]}")
    print(f"Selected {len(technologies)} technolog(y/ies): {technologies}")
    print(f"Selected {len(data_sizes)} data size(s): {data_sizes}")
    print(f"Planned {len(jobs)} job(s), reps={args.reps}, max_workers={args.max_workers}")
    print(f"Output layout: {base_out}/run_<rep>/size_<bytes>B/<technology>/<testbed_stem>/")
    print(f"Batch plan: {base_out / 'batch_plan.csv'}")

    if args.dry_run:
        for j in jobs:
            print(
                f"[DRY] job={j.job_id:04d} rep={j.rep} size={j.data_size_bytes}B "
                f"tech={j.technology} testbed={j.testbed_path.name} -> {j.out_dir}"
            )
        return 0

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(run_job, job, sim_path, args.python, extra_sim_args, cwd) for job in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            status = "OK" if r["returncode"] == 0 else f"ERR({r['returncode']})"
            print(
                f"[{status}] job={r['job_id']:04d} rep={r['rep']} size={r['data_size_bytes']}B "
                f"tech={r['technology']} testbed={r['testbed']} seed={r['seed']} "
                f"elapsed={r['elapsed_s']:.1f}s -> {r['out_dir']}",
                flush=True,
            )

    results_path = base_out / "batch_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "job_id",
                "rep",
                "data_size_bytes",
                "technology",
                "testbed",
                "seed",
                "returncode",
                "elapsed_s",
                "out_dir",
            ],
        )
        w.writeheader()
        for r in sorted(results, key=lambda x: x["job_id"]):
            w.writerow(r)

    failed = [r for r in results if r["returncode"] != 0]
    print(f"Wrote results: {results_path}")
    if failed:
        print(f"FAILED jobs: {len(failed)}/{len(results)}")
        return 1
    print(f"All jobs finished successfully: {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
