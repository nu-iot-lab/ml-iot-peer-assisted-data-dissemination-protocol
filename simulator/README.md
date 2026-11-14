# Network Dissemination Simulators

This directory contains two Python/SimPy implementations for evaluating the dissemination of model updates in constrained IoT networks:

1. **Proposed peer-assisted protocol** (`proposed-protocol-simulator/`): a dissemination process based on `ADV → UPD → DATA → END → MISS → REPAIR`.
2. **Deluge baseline** (`deluge-simulator/`): page-based dissemination using `ADV → REQ → DATA`. Devices can forward completed pages, but count as fully updated only after receiving the entire update.

The implementations are **independent**. Each simulator keeps its own `config/` and `testbeds/` directories, simulation scripts, batch runner, energy-profile definitions, and generated outputs. Run the commands below from the respective simulator directory.

## Directory layout

```text
simulator/
├── README.md
├── proposed-protocol-simulator/
│   ├── dist_lora_sim.py         # Main protocol simulator
│   ├── config.py                # Command-line/configuration parsing
│   ├── radio_profiles.py       # Radio models, airtimes, path loss
│   ├── channel.py              # Channel, collision, capture, CAD/CCA
│   ├── models.py               # Node and packet data structures
│   ├── io_utils.py             # CSV output helpers
│   ├── run_batch.py            # Batch experiment runner
│   ├── calc_energy_reports.py  # Energy post-processing
│   ├── energy_profiles.csv     # Radio current/voltage profiles
│   ├── config/                 # Seven radio configuration .args files
│   ├── testbeds/               # This simulator's terrain CSV files
│   └── output/                 # Generated when experiments run
└── deluge-simulator/
    ├── deluge_sim.py           # Deluge baseline simulator
    ├── config_deluge.py
    ├── radio_profiles.py
    ├── channel_deluge.py
    ├── models_deluge.py
    ├── io_utils_deluge.py
    ├── run_batch_deluge.py
    ├── calc_energy_reports_deluge.py
    ├── energy_profiles.csv
    ├── config/                 # Seven Deluge radio configuration .args files
    ├── testbeds/               # Deluge's own terrain CSV files
    └── output_deluge/          # Generated when experiments run
```

Output folders are created by the programs; they do not need to be committed to the repository.

## Requirements

- Python 3.10 or later
- [SimPy](https://simpy.readthedocs.io/) 4.x

Install the core dependency in your Python environment:

```bash
python3 -m pip install 'simpy>=4,<5'
```

Matplotlib is not required for normal simulation runs. The energy CSV utilities use the Python standard library and do not require plotting packages.

## Radio configurations

Both implementations include configurations for these seven evaluated technology/MAC combinations:

| Radio configuration | Proposed `.args` file | Deluge `.args` file |
| --- | --- | --- |
| LoRa 2.4 GHz, SF5 | `lora24_sf5.args` | `deluge_lora24_sf5.args` |
| LoRa 2.4 GHz, SF5 with CAD | `lora24_sf5_csma.args` | `deluge_lora24_sf5_csma.args` |
| LoRa 2.4 GHz, SF7 | `lora24_sf7.args` | `deluge_lora24_sf7.args` |
| LoRa 2.4 GHz, SF7 with CAD | `lora24_sf7_csma.args` | `deluge_lora24_sf7_csma.args` |
| Sub-GHz FSK, pure ALOHA | `subghz_fsk_raw.args` | `deluge_subghz_fsk_raw.args` |
| Sub-GHz FSK with CSMA | `subghz_fsk_csma.args` | `deluge_subghz_fsk_csma.args` |
| IEEE 802.11ah-style profile | `halow_raw.args` | `deluge_halow_raw.args` |

The channel model includes radio-specific packet airtime, path loss, shadowing, half-duplex operation, collisions/capture, and configured channel-access mechanisms. CAD and CSMA/CCA are simplified simulation models; in particular, the IEEE 802.11ah-style profile is **not** a complete implementation of the IEEE 802.11ah MAC.

## Testbeds

**Each simulator reads its own `testbeds/` directory**; there is no common testbed path. Testbed filenames typically look like `testbed_2km_300_1.csv`.

Each CSV specifies node positions and initial source devices:

```csv
id,x,y,gateway
0,100.0,200.0,1
1,130.0,220.0,0
```

For paired protocol comparisons, ensure that corresponding files in the two independent `testbeds/` directories contain equivalent node layouts and gateways. A matching filename alone is not sufficient if the files were regenerated separately.

## Run a single simulation

### Proposed protocol

```bash
cd simulator/proposed-protocol-simulator

python3 dist_lora_sim.py \
  --config config/lora24_sf7.args \
  --testbed testbeds/testbed_2km_300_1.csv \
  --model-size-bytes 15850 \
  --max-days 100 \
  --out-dir output/example_sf7 \
  --no-packet-log
```

### Deluge baseline

Start from the **repository root** (not from the proposed simulator directory):

```bash
cd simulator/deluge-simulator

python3 deluge_sim.py \
  --config config/deluge_lora24_sf7.args \
  --testbed testbeds/testbed_2km_300_1.csv \
  --model-size-bytes 15850 \
  --max-days 100 \
  --out-dir output_deluge/example_sf7 \
  --no-packet-log
```

The simulator interprets `--max-days` as the maximum number of dissemination cycles. Use `--until-complete` to stop early when all nodes have received the entire update; **omit it** when comparing protocols over a fixed observation horizon.

## Run batch experiments

Each simulator has a separate batch runner, operating only on its own configuration and testbed files. Both support parallel workers, node-count filtering, update-size selection, fixed random seeds, and a `--dry-run` mode.

### Proposed protocol

From `simulator/proposed-protocol-simulator/`:

```bash
python3 run_batch.py \
  --testbeds-dir testbeds \
  --config-dir config \
  --node-counts 100 300 500 \
  --data-sizes 15850 177894 \
  --days-by-size 15850:100 177894:200 \
  --reps 1 \
  --seed-base 2026 \
  --max-workers 8 \
  --base-out output
```

### Deluge baseline

From `simulator/deluge-simulator/`:

```bash
python3 run_batch_deluge.py \
  --testbeds-dir testbeds \
  --config-dir config \
  --node-counts 100 300 500 \
  --data-sizes 15850 177894 \
  --days-by-size 15850:100 177894:200 \
  --reps 1 \
  --seed-base 2026 \
  --max-workers 8 \
  --base-out output_deluge
```

Append `--dry-run` to either command to list the planned jobs without running the simulations. Update sizes and cycle limits here are explicit so the experiments do not depend on defaults hardcoded in a batch runner.

The batch output structure is organized by repetition, update size, radio configuration, and testbed. As configured above, the two simulators write to their **own** `output/` and `output_deluge/` directories.

## Output files

Each completed simulation normally writes:

| File | Description |
| --- | --- |
| `metadata.csv` | Simulation configuration and derived parameters |
| `node_summary.csv` | Per-node traffic, listening-time, and progress counters |
| `node_state.csv` | Final update state and received data |
| `packet_log.csv` | Reception events, unless `--no-packet-log` is used |

Batch runners also produce job/manifest information and capture program output. These CSVs can be analyzed independently; no plotting scripts are required to generate them.

## Energy reports

After running a batch, compute energy summaries separately for each protocol. Energy estimates use each implementation's own radio-current profiles and measured/modelled activity durations.

**Proposed** — from `simulator/proposed-protocol-simulator/`:

```bash
python3 calc_energy_reports.py \
  --input output --recursive \
  --energy-profiles energy_profiles.csv \
  --mcu-current-ma 8 --mcu-voltage 3.3 \
  --skip-per-run-reports
```

**Deluge** — from `simulator/deluge-simulator/`:

```bash
python3 calc_energy_reports_deluge.py \
  --input output_deluge --recursive \
  --energy-profiles energy_profiles.csv \
  --mcu-current-ma 8 --mcu-voltage-v 3.3 \
  --skip-per-run-reports
```

Both scripts produce combined CSV summaries, including run-level energy estimates, in the respective output tree. These are **post-processing** steps, not additional simulator dependencies.

## Scope and reproducibility notes

- The proposed protocol uses the `ADV`, `UPD`, `DATA`, `END`, `MISS`, and `REPAIR` stages; Deluge uses page-based `ADV`, `REQ`, and `DATA` exchanges.
- Both implementations support fixed-cycle experiments and early termination upon full completion; the selected mode matters when comparing completion and energy.
- The same radio settings and equivalent testbed layouts should be used for a fair comparison. The simulators nonetheless load the files from **their own directories**.
- The simulations are abstractions of the listed radio technologies, not complete hardware or standards-conformance implementations.
- No plotting scripts, historical simulation outputs, or paired-campaign launcher are required to run either implementation.

For the research context and the machine-learning part of the project, return to the [repository README](../README.md).
