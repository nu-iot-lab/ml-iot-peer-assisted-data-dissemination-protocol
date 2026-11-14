# Radio technology config files

Use these with the current simulator via `--config`, not `@file`, because `--config` supports comments.

Direct example:

```bash
python3 dist_lora_sim.py --config config/lora24.args
```

Batch example:

```bash
python3 run_batch.py \
  --testbeds-dir testbeds \
  --config-dir config \
  --technologies lora24 ieee802154g_raw halow_raw \
  --reps 2 \
  --max-workers 4
```

The batch runner overrides `--testbed`, `--out-dir`, and `--seed`, so the values in these config files are only direct-run defaults.

Common assumptions:
- 2 km × 2 km area: `--area-m2 4000000`
- 250 kB model: `--model-size-bytes 250000`
- bitmap MISS
- automatic ADV/UPD/MISS control windows enabled
- UPD on channel 1, ADV/control on channel 0
- number of radio channels is taken from each radio profile by default; do not set `--num-channels` here unless you intentionally want to override profile defaults.
