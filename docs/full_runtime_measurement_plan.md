# Full Runtime Measurement

Use this workflow only when runtime numbers are needed for all paper
configurations.

## Coverage

The runtime grid contains 75 rows:

| Family | Count |
| --- | ---: |
| Auto-Encoder | 9 |
| Classification (CE) | 5 |
| Classification (ArcFace) | 5 |
| Separation | 8 |
| Contrastive learning (SimCLR) | 5 |
| Contrastive learning (SimSiam) | 5 |
| Pre-trained | 8 |
| Shared backbone | 30 |

The matching checklist is `docs/full_runtime_measurements_template.csv`.

## Run

```bash
export ASD_DATASET_ROOT=/path/to/asd_dataset
export ASD_LOGMEL_ROOT=/path/to/asd_dataset_logmel
export RUNTIME_ROOT=logs/runtime/full_grid
export DEVICE=cuda:0

bash scripts/measure_full_runtime_grid.sh
DRY_RUN=0 bash scripts/measure_full_runtime_grid.sh
```

The first command prints the planned commands. The second command runs the
one-epoch runtime grid. Re-running with the same `RUNTIME_ROOT` skips commands
whose `.time` file already contains `exit_code=0`.

BEATs evaluation requires local checkpoints. The default paths are:

```bash
export BEATS_ITER3_CHECKPOINT="./beats/BEATs_iter3.pt"
export BEATS_ITER3_PLUS_CHECKPOINT="./beats/BEATs_iter3_plus_AS2M.pt"
export REQUIRE_FULL_PRETRAIN_GRID=1
```

## Summarize

After all jobs finish, create the completed table:

```bash
python scripts/summarize_runtime_grid.py \
  --runtime_root "${RUNTIME_ROOT}" \
  --output docs/full_runtime_measurements_completed.csv
```

The summarizer fails if any expected configuration is missing, failed, or still
running.
