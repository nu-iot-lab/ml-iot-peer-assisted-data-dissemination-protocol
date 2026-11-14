# Wireless Network Simulator — Deluge Variant

This repository variant implements a **Deluge-style page-based dissemination simulator**. It keeps the same radio profile layer, channel model, path-loss assumptions, testbed format, CSMA/CCA behavior, LoRa CAD-CSMA behavior, and analysis workflow as the original simulator, but replaces the custom protocol with Deluge-like dissemination.

The Deluge protocol flow used here is:

```text
ADV -> REQ -> DATA
```

A node may forward a completed page before it has the full object. However, a node is counted as `updated` only when it has reconstructed the **full object**.

---

## Core Deluge files

```text
deluge_sim.py                    Main Deluge simulator
config_deluge.py                 CLI/config parsing for Deluge
models_deluge.py                 Node and Packet dataclasses with page-level state
channel_deluge.py                Channel, RX/TX, collision, CSMA/CAD logic for ADV/REQ/DATA
io_utils_deluge.py               metadata.csv, node_summary.csv, node_state.csv, packet_log.csv writers
run_batch_deluge.py              Batch runner for Deluge experiments
```

The original files are kept as references:

```text
dist_lora_sim.py
config.py
models.py
channel.py
io_utils.py
```

The radio layer remains shared:

```text
radio_profiles.py
```

---

## Deluge model

The update object is divided into:

```text
object/version -> pages -> packets/chunks
```

Important rules:

```text
ADV advertises completed pages, not necessarily a full update.
REQ requests missing packets from one page.
DATA broadcasts requested page packets.
A page-complete node can forward that page.
A fully updated node is one with all chunks/pages of the object.
```

The 90% update script therefore counts only full-object completion, not page completion or forwarding participation.

---

## Running one Deluge simulation

Example:

```bash
python3 deluge_sim.py \
  --config config/deluge_lora24_sf7.args \
  --testbed testbeds/testbed_2km_300_1.csv \
  --out-dir output_deluge/manual_lora24_sf7
```

With debug counters:

```bash
python3 deluge_sim.py \
  --config config/deluge_lora24_sf7.args \
  --debug
```

Useful run-control flags:

```text
--max-days N          Maximum number of Deluge cycles
--max-cycles N        Alias for --max-days
--until-complete      Stop once all nodes are fully updated
--debug               Print per-cycle counters
--no-packet-log       Disable packet_log.csv
```

For fixed-cycle energy comparisons, do not use `--until-complete`; let all technologies run for the same number of cycles.

---

## Deluge-specific config flags

```text
--deluge-page-packets N
--deluge-adv-window-ms MS
--deluge-req-window-ms MS
--deluge-req-backoff-ms MIN MAX
--deluge-data-gap-ms MS
--deluge-round-gap-ms MS
--deluge-sender-selection recent
--deluge-page-selection lowest
--deluge-packet-order sequential
--deluge-immediate-adv-on-page-complete / --no-deluge-immediate-adv-on-page-complete
```

The first implementation uses:

```text
sender selection: most recent useful advertiser
page selection: lowest missing useful page
packet order: sequential by default
FEC: not implemented
sender suppression: not implemented
```

---

## Config files

Deluge-specific configs are provided as:

```text
config/deluge_lora24_sf5.args
config/deluge_lora24_sf5_csma.args
config/deluge_lora24_sf7.args
config/deluge_lora24_sf7_csma.args
config/deluge_subghz_fsk_raw.args
config/deluge_subghz_fsk_csma.args
config/deluge_halow_raw.args
```

The CAD/CSMA stage lists should use Deluge packet types:

```text
--csma-stages ADV REQ DATA
--cad-csma-stages ADV REQ DATA
--phy-repeat-stages ADV REQ DATA
```

---

## Channels and radio behavior

The simulator keeps the existing radio/channel abstractions:

```text
ADV/REQ use the control channel.
DATA uses data channels selected from channels excluding the control channel when possible.
Half-duplex TX/RX is enforced.
Collision/capture logic is retained.
Generic CSMA/CCA is retained.
LoRa CAD-CSMA is retained.
Physical repetitions are retained.
```

Deluge is treated as a protocol layer. CAD and CSMA are lower-layer channel-access options and can be enabled for Deluge packet types.

---

## Output files

Each run directory normally contains:

```text
metadata.csv
node_summary.csv
node_state.csv
packet_log.csv
stdout.txt / stderr.txt if run through run_batch_deluge.py
batch_manifest.csv if run through run_batch_deluge.py
```

Deluge-native node summary columns include:

```text
tx_adv
tx_req
tx_data
adv_used_s
req_used_s
data_used_s
rx_on_adv_ms
rx_on_req_ms
rx_on_data_ms
pages_complete_currentver
have_chunks_currentver
```

Compatibility columns such as `tx_upd`, `tx_miss`, `tx_repair`, `rx_on_upd_ms`, `rx_on_miss_ms`, and `rx_on_repair_ms` are written as zero so older workflows fail less often.

Packet logs keep global `chunk_idx` for DATA packets. This is required for full-object reconstruction analyses.

---

## Batch runs

Example:

```bash
python3 run_batch_deluge.py \
  --configs config/deluge_lora24_sf5.args config/deluge_lora24_sf5_csma.args \
  --testbeds 'testbeds/testbed_2km_300_*.csv' \
  --data-sizes 15850 229027 \
  --output-root output_deluge
```

The batch runner writes `batch_manifest.csv` in each run directory.

---

## Energy reports

Use the Deluge energy script:

```bash
python3 calc_energy_reports_deluge.py \
  --input output_deluge \
  --recursive \
  --energy-profiles energy_profiles.csv
```

It writes combined reports to:

```text
output_deluge/energy_reports_combined/node_energy_summary.csv
output_deluge/energy_reports_combined/run_energy_summary.csv
output_deluge/energy_reports_combined/node_count_energy_summary.csv
```

Deluge energy stages are:

```text
adv
req
data
```

---

## Control-packet reports

For Deluge, control packets are:

```text
ADV
REQ
```

Run:

```bash
python3 report_control_packets_deluge.py \
  --input output_deluge \
  --recursive \
  --data-size 15850 \
  --node-count 300 \
  --technologies lora24_sf5 lora24_sf5_csma lora24_sf7 lora24_sf7_csma subghz_fsk_raw subghz_fsk_csma halow_raw \
  --tech-labels "LoRa2.4 SF5" "LoRa2.4 SF5 CAD" "LoRa2.4 SF7" "LoRa2.4 SF7 CAD" "Sub-GHz FSK" "Sub-GHz FSK CSMA" "IEEE802.11ah" \
  --energy-profiles energy_profiles.csv \
  --output reports/control_packets_deluge_15850B_300nodes.csv \
  --violin-output graphs/control_packets_violin/control_packets_deluge_15850B_300nodes.pdf
```

---

## 90% fully updated devices

Use:

```bash
python3 plot_90pct_update_cycles_deluge.py \
  --input output_deluge \
  --recursive \
  --data-size 15850 \
  --node-count 300 \
  --technologies lora24_sf5 lora24_sf5_csma lora24_sf7 lora24_sf7_csma subghz_fsk_raw subghz_fsk_csma halow_raw \
  --tech-labels "LoRa2.4 SF5" "LoRa2.4 SF5 CAD" "LoRa2.4 SF7" "LoRa2.4 SF7 CAD" "Sub-GHz FSK" "Sub-GHz FSK CSMA" "IEEE802.11ah" \
  --threshold 0.90 \
  --expected-testbeds 50 \
  --output graphs/update_90pct_deluge/update_90pct_deluge_15850B_300nodes.pdf
```

Important: this script counts only nodes that received all chunks of the object. It does not count page holders, ADV senders, REQ senders, or DATA senders as fully updated.

---

## Energy violin plots

After generating energy reports:

```bash
python3 plot_violin_all_node_counts_deluge.py \
  --input output_deluge/energy_reports_combined/node_energy_summary.csv \
  --all-data-sizes \
  --techs lora24_sf5 lora24_sf5_csma lora24_sf7 lora24_sf7_csma subghz_fsk_raw subghz_fsk_csma halow_raw \
  --tech-labels "LoRa2.4 SF5" "LoRa2.4 SF5 CAD" "LoRa2.4 SF7" "LoRa2.4 SF7 CAD" "Sub-GHz FSK" "Sub-GHz FSK CSMA" "IEEE802.11ah" \
  --output graphs/violin_all_nodes_deluge/violin_all_nodes_deluge.pdf
```

The y-axis remains:

```text
Energy per device [J]
```
