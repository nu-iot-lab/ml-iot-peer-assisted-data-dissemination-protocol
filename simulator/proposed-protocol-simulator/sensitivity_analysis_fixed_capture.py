#!/usr/bin/env python3
"""OFAT sensitivity analysis with FIXED technology-specific capture thresholds.

Capture thresholds are read from the seven radio .args files and verified:
LoRa2.4 (including CAD): 0 dB; sub-GHz FSK (including CSMA): 7 dB;
IEEE 802.11ah: 2 dB. Capture is NOT a sensitivity variable.

Four sensitivity families remain: channels, shadowing, carrier-sensing threshold,
and transmission-airtime limit. 64 conditions x 10 terrains = 640 runs.
Nominal duty policy: LoRa2.4 unlimited; sub-GHz FSK and IEEE 802.11ah
36 s per cycle. Duty-cycle sweeps apply only to FSK and HaLow.
The original simulator and its .args configurations are never modified.
Previous sensitivity_v4 outputs will NOT be reused.

From the simulator repository root:
  python3 sensitivity_analysis_fixed_capture.py --repo . --output-root output_sensitivity_fixed_capture --terrains 10 --dry-run
  python3 sensitivity_analysis_fixed_capture.py --repo . --output-root output_sensitivity_fixed_capture --terrains 10 --max-workers 16

Python 3.10+; standard library only.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import shutil
import concurrent.futures
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import threading
import traceback
from collections import defaultdict
from typing import Any, Iterable, Optional

TECHNOLOGIES = (
    'lora24_sf5', 'lora24_sf5_cad',
    'lora24_sf7', 'lora24_sf7_cad',
    'subghz_fsk_raw', 'subghz_fsk_csma', 'halow_raw',
)
# Labels used in plans, output folders, CSV summaries and LaTeX are CAD,
# although the original simulator's configuration filenames use *_csma.args.
# This aliases names only: it does not alter any radio/CAD physics.
CONFIG_FILE_STEMS = {
    'lora24_sf5_cad': 'lora24_sf5_csma',
    'lora24_sf7_cad': 'lora24_sf7_csma',
}
RADIO_LABELS = {
    'lora24_sf5': 'LoRa2.4 SF5',
    'lora24_sf5_cad': 'LoRa2.4 SF5 CAD',
    'lora24_sf7': 'LoRa2.4 SF7',
    'lora24_sf7_cad': 'LoRa2.4 SF7 CAD',
    'subghz_fsk_raw': 'Sub-GHz FSK',
    'subghz_fsk_csma': 'Sub-GHz FSK CSMA',
    'halow_raw': 'IEEE 802.11ah',
}


def config_file_stem(technology: str) -> str:
    return CONFIG_FILE_STEMS.get(technology, technology)

# 7 baselines + 21 channel + 21 shadowing + 6 CCA + 9 airtime = 64 conditions.
# A radio's capture threshold is constant in every baseline and sensitivity run.
EXPECTED_CAPTURE_DB = {
    'lora24_sf5': 0.0, 'lora24_sf5_cad': 0.0,
    'lora24_sf7': 0.0, 'lora24_sf7_cad': 0.0,
    'subghz_fsk_raw': 7.0, 'subghz_fsk_csma': 7.0,
    'halow_raw': 2.0,
}
GRID = {
    'channels': [2, 4, 8],                   # 7 radios => 21
    'shadowing': [0, 2, 6],                  # dB, 7 radios => 21
    'cs_threshold': [-95, -85, -75],         # dBm, 2 CSMA radios => 6
    'duty_cycle': [18, 36, 72],              # s/cycle, FSK x2 + HaLow => 9
}
SPECS = {
    'channels': ('--num-channels', (), TECHNOLOGIES, 'channels'),
    'shadowing': ('--pl-sigma', (), TECHNOLOGIES, 'dB'),
    'cs_threshold': ('--csma-carrier-sense-thresh-dbm', ('--csma-carrier-sense-threshold-dbm',), ('subghz_fsk_csma', 'halow_raw'), 'dBm'),
    'duty_cycle': ('--duty-cycle-limit-s', (), (
        'subghz_fsk_raw', 'subghz_fsk_csma', 'halow_raw'), 's'),
}
# Nominal per-cycle TX airtime policy for this experiment (simulation constraint,
# not a universal radio-regulatory requirement). A duty_cycle sweep overrides it.
NOMINAL_DUTY_CYCLE_S = {
    'lora24_sf5': 0.0, 'lora24_sf5_cad': 0.0,
    'lora24_sf7': 0.0, 'lora24_sf7_cad': 0.0,
    'subghz_fsk_raw': 36.0, 'subghz_fsk_csma': 36.0,
    'halow_raw': 36.0,
}
# Potential field spellings in metadata.csv. The simulator version may use any.
META_KEYS = {
    'channels': ('num_channels',),
    'shadowing': ('pl_sigma',),
    'cs_threshold': ('csma_carrier_sense_thresh_dbm',),
    'duty_cycle': ('duty_cycle_limit_s',),
}
PLAN_FIELDS = ('family', 'level', 'technology', 'technology_label', 'terrain', 'testbed', 'seed', 'flag', 'value', 'run_dir', 'command')
STATUS_FIELDS = ('family', 'level', 'technology', 'technology_label', 'terrain', 'testbed', 'seed', 'status', 'returncode', 'error', 'run_dir')
RUN_FIELDS = ('family', 'level', 'technology', 'technology_label', 'terrain', 'seed', 'status', 'n_nodes', 'r100', 'failed_nodes', 't90', 't100', 'energy_j_mean', 'energy_j_total', 'run_dir')
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Job:
    family: str
    level: str
    technology: str
    terrain: int
    testbed: Path
    seed: int
    flag: str
    value: str
    run_dir: Path


def say(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def keyvals(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        return {}
    if 'key' in rows[0] and 'value' in rows[0]:
        return {str(r['key']).strip(): str(r['value']) for r in rows if r.get('key')}
    return dict(rows[0])


def number(x: Any, default: float = math.nan) -> float:
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return default


def numeric_stem(path: Path) -> tuple:
    numbers = re.findall(r'\d+', path.stem)
    return tuple(int(v) for v in numbers), path.name


def level_name(family: str, value: Any) -> str:
    unit = SPECS[family][3]
    return str(value).replace('-', 'minus_').replace('.', 'p') + '_' + unit


def flag_registry(repo: Path) -> set[str]:
    """Read actual source registration flags, without relying on --help output."""
    registered = set()
    for name in ('config.py', 'dist_lora_sim.py'):
        path = repo / name
        if path.exists():
            text = path.read_text(encoding='utf-8', errors='replace')
            # Parse argparse add_argument declarations with literal flags.
            for m in re.finditer(r'add_argument\s*\((.*?)\)', text, re.DOTALL):
                registered.update(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", m.group(1)))
    return registered


def channel_consumes_capture_setting(repo: Path) -> bool:
    """Require actual receiver/collision code to read the capture setting.

    This guards against accepting a CLI override that has no physical effect.
    """
    import ast
    path = repo / "channel.py"
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "collision_th_db":
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "collision_th_db"):
            return True
    return False



def validate_nominal_duty_profiles(repo: Path) -> None:
    """Require explicit nominal DC settings before reusing old sensitivity runs."""
    for technology, expected in NOMINAL_DUTY_CYCLE_S.items():
        path = repo / 'config' / f'{config_file_stem(technology)}.args'
        values = []
        for line in path.read_text(encoding='utf-8').splitlines():
            tokens = shlex.split(line.split('#', 1)[0])
            for i, token in enumerate(tokens):
                flag = '--duty-cycle-limit-s'
                if token == flag:
                    if i + 1 >= len(tokens):
                        raise RuntimeError(f'Missing value after {flag} in {path}')
                    values.append(float(tokens[i + 1]))
                elif token.startswith(flag + '='):
                    values.append(float(token.split('=', 1)[1]))
        if len(values) != 1 or not math.isclose(values[0], expected, abs_tol=1e-9):
            raise RuntimeError(
                f'{technology}: nominal duty limit must be {expected:g} s/cycle '
                f'in {path}, found {values}. Correct the .args file first.')


def select_flags(repo: Path) -> dict[str, str]:
    registry = flag_registry(repo)
    flags = {}
    for family, (preferred, aliases, _techs, _unit) in SPECS.items():
        if not registry:  # source might use a different parser/config loader
            flags[family] = preferred
            continue
        matches = [v for v in (preferred, *aliases) if v in registry]
        if not matches:
            raise RuntimeError(
                f"No argparse flag found for {family}: tested {preferred}, {', '.join(aliases)}. "
                f"The isolated parser patch did not expose this flag. "
                f"Original simulator unchanged; no experiment was started."
            )
        flags[family] = matches[0]
    return flags


def patch_config_text(config_source: str, path: Path) -> str:
    """No capture CLI patch is needed: the simulator already supports its .args setting."""
    return config_source


def validate_capture_profiles(repo: Path) -> None:
    """Fail closed rather than accidentally running with the old 1 dB defaults."""
    for technology, expected in EXPECTED_CAPTURE_DB.items():
        path = repo / 'config' / f'{config_file_stem(technology)}.args'
        if not path.is_file():
            raise RuntimeError(f'Missing radio configuration: {path}')
        values = []
        for line in path.read_text(encoding='utf-8').splitlines():
            tokens = shlex.split(line.split('#', 1)[0])
            for i, token in enumerate(tokens):
                for flag in ('--collision-thresh-db', '--collision_th_db'):
                    if token == flag:
                        if i+1 >= len(tokens):
                            raise RuntimeError(f'Missing value after {flag} in {path}')
                        values.append(float(tokens[i+1]))
                    elif token.startswith(flag + '='):
                        values.append(float(token.split('=', 1)[1]))
        if len(values) != 1 or not math.isclose(values[0], expected, abs_tol=1e-9):
            raise RuntimeError(
                f'{technology}: expected exactly one capture setting of {expected:g} dB '
                f'in {path}, found {values}. Fix the .args file before running.')
    registration = flag_registry(repo)
    if not ({'--collision-thresh-db', '--collision_th_db'} & registration):
        raise RuntimeError('The simulator config.py does not register the capture flag in .args files.')
    if not channel_consumes_capture_setting(repo):
        raise RuntimeError('channel.py does not consume collision_th_db; cannot verify capture behavior.')


def isolated_simulator_repo(source_repo: Path, root: Path, output_root: Path,
                            simulator_name: str, *, temp_dir: Optional[Path] = None) -> Path:
    """Create an isolated source overlay on a Linux filesystem, never on a USB drive.

    Removable FAT/exFAT/NTFS drives commonly reject symlink_to() with EPERM.
    Keep the overlay in /tmp (or --temp-dir), with symlinks pointing TO the
    original repo's *read-only* assets. Python sources are normal copied files;
    only that copy of config.py is patched. Results remain in root on the
    requested output drive. No original source files are edited.

    A new overlay is used for every invocation, avoiding cross-run races and
    stale patched config modules from earlier sensitivity scripts.
    """
    if not channel_consumes_capture_setting(source_repo):
        raise RuntimeError(
            'channel.py does not read collision_th_db; exposing a flag alone '
            'would not modify the collision model. Original files untouched.'
        )

    # tempfile normally resolves to a native Linux filesystem (/tmp). If the
    # user overrides this, the destination must also support symbolic links.
    import tempfile
    base = temp_dir.expanduser().resolve() if temp_dir else Path(tempfile.gettempdir()).resolve()
    base.mkdir(parents=True, exist_ok=True)
    overlay = Path(tempfile.mkdtemp(prefix='sensitivity_v4_isolated_', dir=base))
    try:
        link_test = overlay / '_symlink_test'
        link_test.symlink_to(source_repo / 'config.py')
        link_test.unlink()
    except OSError as ex:
        raise RuntimeError(
            f'Isolated overlay directory {base} does not support Linux '
            f'symbolic links: {ex}. Specify --temp-dir pointing to a '
            'Linux filesystem, e.g. /tmp.'
        ) from ex

    # Avoid recursive or enormous source-tree copies. Only .py sources are
    # copied; directories and other assets are linked in the Linux overlay.
    excluded = {'.git', '.venv', 'venv', '__pycache__', 'node_modules',
                'output', 'graphs',
                output_root.parts[0] if not output_root.is_absolute() and output_root.parts
                else output_root.name}
    for entry in source_repo.iterdir():
        if entry.name in excluded or entry.name == overlay.name:
            continue
        target = overlay / entry.name
        if entry.is_dir():
            target.symlink_to(entry.resolve(), target_is_directory=True)
        elif entry.is_file() and entry.suffix == '.py':
            # copyfile() avoids copystat, chmod and timestamp preservation.
            shutil.copyfile(entry, target)
        elif entry.is_file():
            target.symlink_to(entry.resolve())

    config_file = overlay / 'config.py'
    if not config_file.is_file():
        raise RuntimeError('No config.py in simulator source repository')
    original = (source_repo / 'config.py').read_text(encoding='utf-8')
    isolated = patch_config_text(original, config_file)
    config_file.write_text(isolated, encoding='utf-8')
    simulator = overlay / simulator_name
    if not simulator.is_file():
        raise RuntimeError(f'Simulator script missing in overlay: {simulator_name}')

    # Persistent provenance (empty parser diff: capture settings are already supported);
    # these records remain after the /tmp overlay is cleaned by the OS.
    (root / '_capture_patch.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), isolated.splitlines(keepends=True),
        fromfile='original/config.py', tofile='isolated/config.py')),
        encoding='utf-8')
    (root / '_capture_provenance.json').write_text(json.dumps({
        'source_repo': str(source_repo),
        'isolated_runtime_repo': str(overlay),
        'source_config_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest(),
        'isolated_config_sha256': hashlib.sha256(isolated.encode('utf-8')).hexdigest(),
        'capture_threshold_policy': 'LoRa2.4=0 dB; FSK=7 dB; IEEE 802.11ah=2 dB (fixed in .args)',
        'original_sources_modified': False,
    }, indent=2) + '\n', encoding='utf-8')
    return overlay


def make_jobs(args: argparse.Namespace, repo: Path, root: Path, flags: dict[str, str]) -> list[Job]:
    testbeds = sorted((repo / 'testbeds').glob(args.testbed_glob), key=numeric_stem)
    if len(testbeds) < args.terrains:
        raise RuntimeError(f'Only {len(testbeds)} testbeds matched {args.testbed_glob}; requested {args.terrains}')
    testbeds = testbeds[:args.terrains]
    missing = [t for t in TECHNOLOGIES if not (repo / 'config' / f'{config_file_stem(t)}.args').is_file()]
    if missing:
        raise RuntimeError(f'Missing config files: {missing}')
    jobs = []
    for tech in TECHNOLOGIES:
        cases = [('baseline', 'nominal', '', '')]
        for family in GRID:
            if tech not in SPECS[family][2]:
                continue
            for value in GRID[family]:
                cases.append((family, level_name(family, value), flags[family], str(value)))
        for family, level, flag, value in cases:
            for i, testbed in enumerate(testbeds, 1):
                jobs.append(Job(family, level, tech, i, testbed.resolve(),
                                args.seed_base + i - 1, flag, value,
                                root / 'runs' / family / level / tech /
                                f'terrain_{i:02d}_{testbed.stem}'))
    return jobs


def command(job: Job, args: argparse.Namespace, repo: Path, *, preflight: bool = False) -> list[str]:
    out = job.run_dir if not preflight else job.run_dir.parent.parent.parent.parent.parent / 'preflight' / job.family / job.level / job.technology / f'terrain_{job.terrain:02d}'
    argv = [sys.executable, str((repo / args.simulator).resolve()),
            '--config', str((repo / 'config' / f'{config_file_stem(job.technology)}.args').resolve()),
            '--testbed', str(job.testbed), '--out-dir', str(out),
            '--model-size-bytes', str(args.model_size_bytes),
            '--max-days', str(args.preflight_days if preflight else args.max_days),
            '--seed', str(job.seed)]
    if job.flag:
        argv.extend((job.flag, job.value))
    if args.no_packet_log:
        argv.append('--no-packet-log')
    return argv


def run_dir_for(job: Job, args: argparse.Namespace, repo: Path, preflight: bool = False) -> Path:
    cmd = command(job, args, repo, preflight=preflight)
    return Path(cmd[cmd.index('--out-dir') + 1])


def metadata_matches(job: Job, run_dir: Path, args: argparse.Namespace, *, strict: bool, expected_days: int | None = None) -> tuple[bool, str]:
    meta = keyvals(run_dir / 'metadata.csv')
    if not meta:
        return False, 'missing or empty metadata.csv'
    for k, expected in [('model_size_bytes', args.model_size_bytes), ('max_days', args.max_days if expected_days is None else expected_days)]:
        if k in meta and (not math.isclose(number(meta[k]), expected, rel_tol=0, abs_tol=1e-8)):
            return False, f'metadata {k}={meta[k]} expected {expected}'
    # Verify technology-specific threshold whenever the simulator exported it.
    observed_capture = next((meta[k] for k in ('collision_th_db', 'collision_threshold_db', 'capture_th_db') if k in meta), None)
    if observed_capture is not None and not math.isclose(
            number(observed_capture), EXPECTED_CAPTURE_DB[job.technology], abs_tol=1e-9):
        return False, f'metadata capture={observed_capture} dB, expected {EXPECTED_CAPTURE_DB[job.technology]} dB'
    # Also check nominal duty for every non-duty scenario. This catches old
    # HaLow runs performed before its 36 s/cycle limit was enabled, which
    # otherwise could be silently reused after changing the .args file.
    expected_duty = float(job.value) if job.family == 'duty_cycle' else NOMINAL_DUTY_CYCLE_S[job.technology]
    actual_duty = meta.get('duty_cycle_limit_s')
    if actual_duty is None:
        if strict:
            return False, 'metadata duty_cycle_limit_s missing; cannot certify nominal duty policy'
    elif not math.isclose(number(actual_duty), expected_duty, rel_tol=0, abs_tol=1e-9):
        return False, f'metadata duty_cycle_limit_s={actual_duty}; expected {expected_duty:g}'
    if job.family == 'baseline':
        return True, ''
    aliases = META_KEYS[job.family]
    found = next((k for k in aliases if k in meta), None)
    if found is None:
        if strict:
            return False, f'Cannot verify override {job.flag} {job.value}; metadata lacks {aliases}'
        return True, 'override not independently verifiable in metadata'
    if not math.isclose(number(meta[found]), float(job.value), abs_tol=1e-7, rel_tol=1e-7):
        return False, f'OVERRIDE NOT APPLIED: metadata {found}={meta[found]}, requested {job.value}'
    return True, ''


def has_success(job: Job, args: argparse.Namespace, *, strict: bool = True) -> bool:
    if not (job.run_dir / 'metadata.csv').is_file() or not (job.run_dir / 'node_summary.csv').is_file():
        return False
    if not read_csv(job.run_dir / 'node_summary.csv'):
        return False
    # Only consider current fixed-capture runs certified by their metadata/manifest.
    manifest = job.run_dir / 'sensitivity_v4.json'
    if not manifest.exists():
        return False
    try:
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if (info.get('status') != 'success' or info.get('flag') != job.flag
                or info.get('value') != job.value
                or info.get('capture_threshold_db') != EXPECTED_CAPTURE_DB[job.technology]):
            return False
        ok, _ = metadata_matches(job, job.run_dir, args, strict=strict)
        return ok
    except (OSError, ValueError, TypeError):
        return False


def run_one(job: Job, args: argparse.Namespace, repo: Path, *, preflight: bool = False) -> dict:
    out = run_dir_for(job, args, repo, preflight=preflight)
    cmd = command(job, args, repo, preflight=preflight)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'command.txt').write_text(shlex.join(cmd) + '\n', encoding='utf-8')
    try:
        cp = subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors='replace', timeout=args.timeout_s or None,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        (out / 'stdout.txt').write_text(cp.stdout, encoding='utf-8')
        (out / 'stderr.txt').write_text(cp.stderr, encoding='utf-8')
        code = cp.returncode
        error = ''
        if code != 0:
            tail = cp.stderr.strip().splitlines()[-8:] or cp.stdout.strip().splitlines()[-8:]
            error = '\n'.join(tail)[-1600:] or f'exit={code} (see stderr.txt)'
        elif not read_csv(out / 'node_summary.csv'):
            error = 'Simulator exited 0 but node_summary.csv missing or empty'
        elif not (out / 'metadata.csv').is_file():
            error = 'Simulator exited 0 but metadata.csv missing'
        else:
            ok, error = metadata_matches(job, out, args, strict=args.strict_metadata,
                                         expected_days=args.preflight_days if preflight else args.max_days)
        status = 'success' if not error else 'failed'
    except (OSError, subprocess.TimeoutExpired) as exc:
        code, status, error = -1, 'failed', f'{type(exc).__name__}: {exc}'
        (out / 'stderr.txt').write_text(error + '\n', encoding='utf-8')
    manifest = {'status': status, 'returncode': code, 'flag': job.flag,
                'capture_threshold_db': EXPECTED_CAPTURE_DB[job.technology],
                'value': job.value, 'testbed': str(job.testbed), 'seed': job.seed,
                'max_days': args.preflight_days if preflight else args.max_days,
                'command': cmd, 'error': error}
    (out / 'sensitivity_v4.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return {**asdict(job), 'technology_label': RADIO_LABELS[job.technology], 'testbed': job.testbed.name, 'run_dir': str(out),
            'status': status, 'returncode': code, 'error': error}


def preflight(jobs: list[Job], args: argparse.Namespace, repo: Path) -> None:
    # Smoke each (technology, family) present in *pending* jobs.  This matters
    # for resumed extensions: the new CAD jobs have no generic CSMA threshold
    # cases, and the already-finished five technologies need not be rerun.
    representative = {}
    for j in jobs:
        representative.setdefault((j.technology, j.family), j)
    say(f'Preflight: {len(representative)} one-cycle smoke tests '
        '(one for each pending technology/parameter family).')
    for j in representative.values():
        result = run_one(j, args, repo, preflight=True)
        if result['status'] != 'success':
            raise RuntimeError(f"PREFLIGHT FAILED: {j.family}/{j.level}/{j.technology}\n"
                               f"{result['error']}\nSee {result['run_dir']}/stderr.txt")
    say('Preflight passed. Launching full fixed-horizon jobs.')


def mean(values: Iterable[Any]) -> float:
    valid = [float(x) for x in values if isinstance(x, (float, int)) and math.isfinite(float(x))]
    return statistics.mean(valid) if valid else math.nan


def stdev(values: Iterable[Any]) -> float:
    valid = [float(x) for x in values if isinstance(x, (float, int)) and math.isfinite(float(x))]
    return statistics.stdev(valid) if len(valid) > 1 else (0.0 if valid else math.nan)


def load_energy_csv(args: argparse.Namespace, root: Path) -> dict[tuple[str, str], list[float]]:
    """Read existing calculator results; do not invent currents/voltages.

    Expected: per-node energy report with run_dir + total_j or total_mj.
    Uses exact run_dir where available and a fallback suffix matcher.
    """
    sources = [Path(args.energy_csv).resolve()] if args.energy_csv else [
        root / 'energy_reports_combined' / 'node_energy_summary.csv',
        root / 'node_energy_summary.csv',
    ]
    by_run: dict[tuple[str, str], list[float]] = defaultdict(list)
    for source in sources:
        if not source.exists():
            continue
        for row in read_csv(source):
            path = row.get('run_dir') or row.get('source_run_dir') or row.get('output_dir') or ''
            if not path:
                continue
            energy_key = next((x for x in ('total_j', 'total_mj', 'total_energy_with_mcu_mj',
                                            'total_radio_plus_mcu_energy_mj', 'radio_mj') if row.get(x)), None)
            if not energy_key:
                continue
            amount = number(row[energy_key])
            if not math.isfinite(amount):
                continue
            joules = amount / 1000.0 if energy_key.endswith('_mj') else amount
            p = Path(path)
            by_run[(str(p.resolve()), p.as_posix())].append(joules)
    return by_run


def energy_for_run(energies: dict, path: Path) -> list[float]:
    for (absolute, original), values in energies.items():
        if absolute == str(path.resolve()) or original == path.as_posix():
            return values
    return []


def completion_days_from_summary(rows: list[dict], max_days: int) -> tuple[float, float]:
    """Only use explicit cumulative per-day updated state, never TX counters.

    Returns t90, t100 across cycles; blank if simulator does not export it.
    """
    if not rows:
        return math.nan, math.nan
    # Node state per cycle: d1_updated, d2_updated, etc. Not all builds emit it.
    daily = sorted(((int(m.group(1)), k) for k in rows[0]
                    if (m := re.fullmatch(r'd(\d+)_updated', k)) and int(m.group(1)) <= max_days))
    if not daily:
        return math.nan, math.nan
    t90 = t100 = math.nan
    for day, field in daily:
        valid = [number(r.get(field)) for r in rows]
        if not valid or any(not math.isfinite(x) for x in valid):
            continue
        frac = sum(x >= 1 for x in valid) / len(valid)
        if math.isnan(t90) and frac >= 0.90 - 1e-12:
            t90 = float(day)
        if math.isnan(t100) and frac >= 1.0 - 1e-12:
            t100 = float(day)
    return t90, t100





def completion_days_from_packet_log(run_dir: Path, node_rows: list[dict],
                                    max_days: int) -> tuple[float, float]:
    """Memory-efficient, order-independent reconstruction of T90 and T100.

    Original v4.4 read and sorted EVERY packet-log row. This version streams
    only the relevant columns and derives the first complete day for each node.
    A node is complete when it first sends ADV/DATA/REPAIR, or obtains at
    least complete_need distinct chunks for one version.

    Sorting completion days at the end is equivalent to replaying all valid
    packet events in day order for cycle-level T90/T100. Within-cycle
    timestamps cannot change the integer cycle in which a threshold is met.
    """
    metadata = keyvals(run_dir / 'metadata.csv')
    complete_need = int(number(metadata.get('complete_need'), 0))
    log_path = run_dir / 'packet_log.csv'
    if not log_path.is_file() or complete_need < 1 or not node_rows:
        return math.nan, math.nan

    valid = {int(number(r.get('nid'), -1)) for r in node_rows}
    valid.discard(-1)
    n_nodes = len(valid) or len(node_rows)
    first_completed_day: dict[int, int] = {}
    # (rx, version) -> chunk_idx -> earliest day of nonduplicate receipt
    chunks: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    saw_packet = False

    with log_path.open('r', newline='', encoding='utf-8-sig') as handle:
        for pkt in csv.DictReader(handle):
            saw_packet = True
            day = int(number(pkt.get('day'), 0))
            if day < 1 or day > max_days:
                continue
            kind = str(pkt.get('ptype', '')).strip().upper()
            if kind not in ('ADV', 'DATA', 'REPAIR'):
                continue

            src = int(number(pkt.get('src'), -1))
            if src in valid:
                first_completed_day[src] = min(
                    day, first_completed_day.get(src, day))

            if kind not in ('DATA', 'REPAIR'):
                continue
            rx = int(number(pkt.get('rx_node'), -1))
            if rx not in valid or int(number(pkt.get('is_duplicate'), 0)) != 0:
                continue
            idx = int(number(pkt.get('chunk_idx'), -1))
            version = int(number(pkt.get('version'), 0))
            if idx < 0 or version <= 0:
                continue
            group = chunks[(rx, version)]
            if idx not in group or day < group[idx]:
                group[idx] = day

    if not saw_packet:
        return math.nan, math.nan

    # First day when the node received its complete update for any version.
    for (rx, _version), by_chunk in chunks.items():
        if len(by_chunk) < complete_need:
            continue
        if rx in first_completed_day and min(by_chunk.values()) >= first_completed_day[rx]:
            continue
        from heapq import nsmallest
        day = nsmallest(complete_need, by_chunk.values())[-1]
        first_completed_day[rx] = min(day, first_completed_day.get(rx, day))

    days = sorted(first_completed_day.values())
    k90 = math.ceil(0.9 * n_nodes)
    t90 = float(days[k90 - 1]) if len(days) >= k90 else math.nan
    t100 = float(days[n_nodes - 1]) if len(days) >= n_nodes else math.nan
    return t90, t100


def load_prior_sensitivity_cache(root: Path) -> dict[str, dict]:
    """Reuse analyzed samples only from this fixed-capture output tree."""
    cache_path = root / 'sensitivity_run_samples.csv'
    result = {}
    checkpoint_path = root / 'sensitivity_run_samples_checkpoint.csv'
    for path in (cache_path, checkpoint_path):
        for record in read_csv(path):
            run_dir = record.get('run_dir', '')
            if run_dir and record.get('status') == 'success':
                result[str(Path(run_dir).resolve())] = record
    return result


def collect_samples(jobs: list[Job], args: argparse.Namespace, root: Path) -> list[dict]:
    energies = load_energy_csv(args, root)
    cached = load_prior_sensitivity_cache(root)
    samples = []
    n_jobs = len(jobs)
    reused = 0
    checkpoint_path = root / 'sensitivity_run_samples_checkpoint.csv'
    for i, job in enumerate(jobs, 1):
        rows = read_csv(job.run_dir / 'node_summary.csv')
        ok = has_success(job, args, strict=args.strict_metadata)
        sample = {'family': job.family, 'level': job.level, 'technology': job.technology, 'technology_label': RADIO_LABELS[job.technology],
                  'terrain': job.terrain, 'seed': job.seed, 'run_dir': str(job.run_dir),
                  'status': 'success' if ok else 'failed', 'n_nodes': len(rows),
                  'r100': math.nan, 'failed_nodes': math.nan,
                  't90': math.nan, 't100': math.nan,
                  'energy_j_mean': math.nan, 'energy_j_total': math.nan}
        if ok:
            if 'updated' in rows[0]:
                update = [number(r.get('updated'), 0) for r in rows]
                sample['r100'] = sum(x >= 1 for x in update) / len(update)
                sample['failed_nodes'] = sum(x < 1 for x in update)
            sample['t90'], sample['t100'] = completion_days_from_summary(rows, args.max_days)
            # Reuse completed current fixed-capture runs when the cached run identity and
            # final coverage are consistent with the validated current run.
            old = cached.get(str(job.run_dir.resolve()))
            if (old and int(number(old.get('n_nodes'), -1)) == len(rows)
                    and math.isfinite(number(old.get('r100')))
                    and math.isfinite(number(sample['r100']))
                    and abs(number(old['r100']) - number(sample['r100'])) < 1e-12):
                sample['t90'] = number(old.get('t90'))
                sample['t100'] = number(old.get('t100'))
                reused += 1
            elif not math.isfinite(sample['t90']) or not math.isfinite(sample['t100']):
                log_t90, log_t100 = completion_days_from_packet_log(job.run_dir, rows, args.max_days)
                if not math.isfinite(sample['t90']):
                    sample['t90'] = log_t90
                if not math.isfinite(sample['t100']):
                    sample['t100'] = log_t100
            vals = energy_for_run(energies, job.run_dir)
            if vals:
                sample['energy_j_mean'] = mean(vals)
                sample['energy_j_total'] = sum(vals)
        samples.append(sample)
        if i == 1 or i % 10 == 0 or i == n_jobs:
            # Save recovery point for this fixed-capture campaign.
            tmp_path = checkpoint_path.with_suffix('.tmp')
            write_csv(tmp_path, RUN_FIELDS, samples)
            tmp_path.replace(checkpoint_path)
            say(f'Summarizing [{i}/{n_jobs}] | prior results reused: {reused}')
    return samples


def latex_escape(x: str) -> str:
    return x.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def fmt(v: Any, decimals: int = 3) -> str:
    n = number(v)
    return f'{n:.{decimals}f}' if math.isfinite(n) else '--'


def create_summaries(jobs: list[Job], args: argparse.Namespace, root: Path) -> None:
    samples = collect_samples(jobs, args, root)
    write_csv(root / 'sensitivity_run_samples.csv', RUN_FIELDS, samples)
    grouped = defaultdict(list)
    for s in samples:
        grouped[(s['family'], s['level'], s['technology'])].append(s)
    summaries = []
    for (family, level, tech), series in sorted(grouped.items()):
        good = [s for s in series if s['status'] == 'success']
        summaries.append({
            'family': family, 'level': level, 'technology': tech, 'technology_label': RADIO_LABELS[tech],
            'n_terrains': len(series), 'n_success': len(good), 'n_failed': len(series) - len(good),
            'r100_mean': mean(s['r100'] for s in good), 'r100_std': stdev(s['r100'] for s in good),
            'failure_probability': mean(float(number(s['r100']) < 1 - 1e-12)
                                        for s in good if math.isfinite(number(s['r100']))),
            't90_mean': mean(s['t90'] for s in good), 't100_mean': mean(s['t100'] for s in good),
            'energy_j_mean': mean(s['energy_j_mean'] for s in good),
            'energy_j_total_mean': mean(s['energy_j_total'] for s in good),
        })
    cols = ('family', 'level', 'technology', 'technology_label', 'n_terrains', 'n_success', 'n_failed',
            'r100_mean', 'r100_std', 'failure_probability', 't90_mean', 't100_mean',
            'energy_j_mean', 'energy_j_total_mean')
    write_csv(root / 'sensitivity_summary.csv', cols, summaries)
    write_csv(root / 'sensitivity_energy_summary.csv',
              ('family', 'level', 'technology', 'technology_label', 'n_success', 'energy_j_mean', 'energy_j_total_mean'), summaries)
    baselines = {s['technology']: s for s in summaries if s['family'] == 'baseline'}
    ordering = []
    for s in summaries:
        base = baselines.get(s['technology'])
        a, b = number(s['r100_mean']), number(base['r100_mean']) if base else math.nan
        ordering.append({**s, 'delta_r100': a - b if math.isfinite(a) and math.isfinite(b) else math.nan,
                         'abs_delta_r100': abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.nan})
    ordering.sort(key=lambda s: -number(s['abs_delta_r100'], -1))
    write_csv(root / 'sensitivity_ordering.csv', (*cols, 'delta_r100', 'abs_delta_r100'), ordering)
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{One-factor-at-a-time sensitivity analysis over common terrains and seeds. Unavailable measurements are indicated by --.}',
             r'\label{tab:sensitivity}', r'\begin{tabular}{lllrrrr}', r'\hline',
             r'Parameter & Level & Radio & $n$ & $R_{100}$ & $P_{\mathrm{fail}}$ & Energy (J/node) \\',
             r'\hline']
    for s in summaries:
        lines.append(f"{latex_escape(s['family'])} & {latex_escape(s['level'])} & "
                     f"{latex_escape(s['technology_label'])} & {s['n_success']}/{s['n_terrains']} & "
                     f"{fmt(s['r100_mean'])} & {fmt(s['failure_probability'])} & "
                     f"{fmt(s['energy_j_mean'])} " + r'\\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table*}', '']
    (root / 'sensitivity_summary.tex').write_text('\n'.join(lines), encoding='utf-8')
    checkpoint_path = root / 'sensitivity_run_samples_checkpoint.csv'
    checkpoint_path.unlink(missing_ok=True)
    successes = sum(s['status'] == 'success' for s in samples)
    say(f'Summary: {successes}/{len(samples)} successful runs. Files under {root}')
    if not any(math.isfinite(number(s['energy_j_mean'])) for s in samples):
        say('Energy unavailable: pass --energy-csv after producing calculator node_energy_summary.csv.')
    if not any(math.isfinite(number(s['t100'])) for s in samples):
        say('T90/T100 unavailable: need explicit dN_updated columns or complete packet_log.csv with complete_need metadata.')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--simulator', default='dist_lora_sim.py')
    p.add_argument('--output-root', type=Path, default=Path('output_sensitivity_fixed_capture'))
    p.add_argument('--terrains', type=int, default=3)
    p.add_argument('--testbed-glob', default='testbed_2km_300_*.csv')
    p.add_argument('--max-workers', type=int, default=8)
    p.add_argument('--max-days', type=int, default=100)
    p.add_argument('--model-size-bytes', type=int, default=177894)
    p.add_argument('--seed-base', type=int, default=20261026)
    p.add_argument('--preflight-days', type=int, default=1)
    p.add_argument('--skip-preflight', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--force', action='store_true', help='Rerun even successful fixed-capture runs.')
    p.add_argument('--timeout-s', type=int, default=0, help='Per-simulation timeout (0=unlimited).')
    p.add_argument('--temp-dir', type=Path, default=None,
                   help='Native Linux temp location for isolated source overlay (default: /tmp).')
    p.add_argument('--no-packet-log', action='store_true', help='Not recommended when T90/T100 needed.')
    p.add_argument('--strict-metadata', action=argparse.BooleanOptionalAction, default=True,
                   help='Require metadata evidence that each sensitivity override was applied.')
    p.add_argument('--energy-csv', default=None, help='Existing per-node calculator CSV with run_dir and total_j/total_mj.')
    args = p.parse_args()
    if args.terrains < 1 or args.max_workers < 1 or args.max_days < 1:
        p.error('--terrains, --max-workers and --max-days must all be >= 1')
    repo = args.repo.resolve()
    simulator = (repo / args.simulator).resolve()
    if not simulator.is_file():
        p.error(f'Simulator not found: {simulator}')
    output_base = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    root = (output_base / 'sensitivity_fixed_capture').resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        validate_capture_profiles(repo)
        validate_nominal_duty_profiles(repo)
        runtime_repo = isolated_simulator_repo(repo, root, args.output_root,
                                               args.simulator, temp_dir=args.temp_dir)
        args.runtime_repo = runtime_repo
        flags = select_flags(runtime_repo)
        jobs = make_jobs(args, repo, root, flags)
    except (RuntimeError, OSError, SyntaxError) as ex:
        p.error(str(ex))
    say('Capture thresholds verified from .args: LoRa2.4=0 dB; FSK=7 dB; IEEE 802.11ah=2 dB.')
    say('Nominal duty limits verified: LoRa2.4=0; FSK=36; IEEE 802.11ah=36 s/cycle.')
    say(f'Original simulator unchanged: {repo}')
    say(f'Isolated simulator copy (Linux temporary storage): {runtime_repo}')
    plan = []
    for j in jobs:
        plan.append({**asdict(j), 'technology_label': RADIO_LABELS[j.technology], 'testbed': j.testbed.name, 'run_dir': str(j.run_dir),
                     'command': shlex.join(command(j, args, runtime_repo))})
    write_csv(root / 'sensitivity_plan.csv', PLAN_FIELDS, plan)
    say(f'Prepared {len(jobs)} runs / {args.terrains} terrains; output: {root}')
    say('Flags: ' + ', '.join(f'{family}={flag}' for family, flag in flags.items()))
    if args.dry_run:
        for row in plan[:12]:
            say(row['command'])
        say('Dry run only. Full plan is in sensitivity_plan.csv.')
        return 0
    if args.summarize_only:
        create_summaries(jobs, args, root)
        return 0
    runnable = [j for j in jobs if args.force or not has_success(j, args, strict=args.strict_metadata)]
    say(f'Skipping {len(jobs) - len(runnable)} validated prior successes; executing {len(runnable)}.')
    if not runnable:
        create_summaries(jobs, args, root)
        return 0
    if not args.skip_preflight:
        try:
            preflight(runnable, args, runtime_repo)
        except RuntimeError as exc:
            say(str(exc))
            say('Aborted before full batch. Fix source CLI/overrides, then rerun.')
            return 2
    status = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(run_one, j, args, runtime_repo): j for j in runnable}
        for i, f in enumerate(concurrent.futures.as_completed(future_map), 1):
            j = future_map[f]
            try:
                record = f.result()
            except Exception as ex:  # catch unexpected worker failure and preserve status
                record = {**asdict(j), 'technology_label': RADIO_LABELS[j.technology], 'status': 'failed', 'returncode': -1, 'error': str(ex)}
                say(traceback.format_exc())
            status.append(record)
            say(f'[{i}/{len(runnable)}] {record["status"]} '
                f'{j.family}/{j.level}/{j.technology}/t{j.terrain}'
                + (f' | {record["error"][:200]}' if record['status'] == 'failed' else ''))
            write_csv(root / 'sensitivity_status.csv', STATUS_FIELDS,
                      sorted(status, key=lambda r: (r['family'], r['level'], r['technology'], r['terrain'])))
    create_summaries(jobs, args, root)
    failures = sum(r['status'] != 'success' for r in status)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
