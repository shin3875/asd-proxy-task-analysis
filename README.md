# Train and Test Code for Proxy-Task Analysis in ASD

This repository contains the training, evaluation, and aggregation code for the
manuscript *Quantitative Analysis of Proxy Tasks for Anomalous Sound
Detection*. The manuscript is currently under review. arXiv:
https://arxiv.org/abs/2601.08480

This repository is intended to make the training, evaluation, aggregation, and
runtime-measurement flow reproducible. It does not include raw datasets,
generated features, checkpoints, or experiment logs.

If you use this code, please cite the manuscript above.

## Data Policy

Raw DCASE-style audio data is not part of the public repository. Reproduction
runs expect the dataset to be supplied separately with this layout:

```text
asd_dataset/
  <device>/
    train/*.wav
    test/*.wav
    aug/*.wav
```

The expected devices are:

```text
bearing fan gearbox pump slider ToyCar ToyConveyor ToyTrain valve
```

Use the official DCASE/Zenodo sources to build `asd_dataset/`:

| Device class | Source |
| --- | --- |
| `pump`, `ToyConveyor` | DCASE 2020 Task 2 Development Dataset, https://zenodo.org/records/3678171 |
| `bearing`, `fan`, `gearbox`, `slider`, `ToyCar`, `ToyTrain`, `valve` | DCASE 2022 Task 2 Development Dataset, https://zenodo.org/records/6355122 |
| Additional normal training data for the 2022 classes, when reproducing the paper setup | DCASE 2022 Task 2 Additional Training Dataset, https://zenodo.org/records/6462969 |

The 2020 dataset provides the six machine types ToyCar, ToyConveyor, valve,
pump, fan, and slider. The 2022 development dataset provides fan, gearbox,
bearing, slider, ToyCar, ToyTrain, and valve; the additional training release
provides extra normal training archives for the 2022 classes. Follow the
licenses and citation requirements on the Zenodo pages.

Generated feature roots are also excluded from version control:

```text
asd_dataset_np/
asd_dataset_logmel/
```

Use toy or sample data only for public examples and smoke tests.

## Runtime Requirements

Use Python 3.10.x and install the pinned package set in `requirements.txt`.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

On other machines, keep the listed package versions where possible, but replace
the PyTorch, torchvision, and torchaudio wheels when your CUDA/CPU runtime
requires different builds.

Local modules required by the entry points are included in `models/`,
`shared_backbone/`, `ae_baseline_utils.py`, and `dataloader.py`. BEAT
checkpoints and the upstream BEATs implementation are not distributed here.

## Experimental Platform

The paper experiments used Python 3.10.15 on Linux 6.14.0-37-generic, PyTorch
2.7.1+cu128, CUDA 12.8, cuDNN 90701, and an NVIDIA GeForce RTX 5090. The raw
platform capture is stored in
`docs/platform_snapshot.txt`. Paper-oriented package versions are listed in
`requirements.txt`.

Unless stated otherwise, models trained from scratch use AdamW, a StepLR
scheduler, and 200 training epochs. The checkpoint used for evaluation is
selected by the lowest training loss. No anomaly-labeled validation set is used
for checkpoint selection.

The experiments fix the optimizer and scheduler across proxy tasks. The
controlled variables are model-capacity parameters: AE bottleneck/hidden
dimensions, ResNet depth, and separation conformer-block/channel width.

## Public Entry Points

Use these role-oriented script names for public reproduction:

```text
rename_dataset_files.py       optional filename normalization
prepare_dataset_features.py   33x augmentation verification and feature export
train_shared_backbone.py      shared-backbone proxy training
train_ae.py                   autoencoder proxy training
train_classifier.py           CE / ArcFace classifier proxy training
train_contrastive.py          SimCLR / SimSiam proxy training
train_separation.py           separation proxy training
evaluate_proxy_asd.py         task-specific checkpoint evaluation
evaluate_shared_backbone.py   shared-backbone checkpoint evaluation
evaluate_pretrain.py          BEAT / CED / EAT pre-trained representation evaluation
plot_proxy_asd_summary.py     summary CSV aggregation and figure generation
analyze_proxy_correlations.py device-aggregated correlation table generation
plot_paper_figure4.py         paper-style three-panel proxy/ASD figure
```

Shared helper modules used by those entry points are `ae_baseline_utils.py`,
`dataloader.py`, `models/`, and `shared_backbone/`.

## Default Policy

Bare script defaults are single-run defaults, not a full paper reproduction
grid. For examples and smoke tests, pass paths and small epoch counts
explicitly. Full paper grids are provided by the reproduction shell wrappers in
`scripts/`.

## Reproduction Flow

1. Normalize filenames if needed:

```bash
python rename_dataset_files.py --root ./asd_dataset:wav --devices bearing fan gearbox pump slider ToyCar ToyConveyor ToyTrain valve --mode dry-run
```

2. Preview planned feature counts:

```bash
python prepare_dataset_features.py --root ./asd_dataset --skip-augment --dry-run --features logmel --logmel-mode raw --mel-n-fft 1024 --mel-hop 512 --feature-duration 10.0 --logmel-out ./asd_dataset_logmel --n-mels 128
```

3. Generate file-wise features after dry-run review. Choose one flow:

Use existing augmentation WAVs:
```bash
python prepare_dataset_features.py --root ./asd_dataset --skip-augment --features logmel --logmel-mode raw --mel-n-fft 1024 --mel-hop 512 --feature-duration 10.0 --logmel-out ./asd_dataset_logmel --n-mels 128 --write-manifest --verify --verify-shape --strict-verify --require-aug
```

Generate augmentation WAVs before feature export:
```bash
python prepare_dataset_features.py --root ./asd_dataset --features logmel --logmel-mode raw --mel-n-fft 1024 --mel-hop 512 --feature-duration 10.0 --logmel-out ./asd_dataset_logmel --n-mels 128 --write-manifest --verify --verify-shape --strict-verify --require-aug
```

AE/shared-backbone smoke feature export without augmentation:
```bash
python prepare_dataset_features.py --root ./asd_dataset --skip-augment --exclude-aug --features logmel --logmel-mode raw --mel-n-fft 1024 --mel-hop 512 --feature-duration 10.0 --logmel-out ./asd_dataset_logmel --n-mels 128 --write-manifest --verify --verify-shape --strict-verify
```

Contrastive and separation training require `<device>/aug` files. If those WAVs
already exist, use `--skip-augment --require-aug`; otherwise omit
`--skip-augment`.

The commands above store raw mel-power matrices. Use `--matrix_log_mode raw`
with these features. If features are exported with `--logmel-mode db`, pass
`--matrix_log_mode already_log` during training and evaluation.
`--feature-duration 10.0` gives the 313-frame input used by the paper setup.

Separation training and evaluation read WAV files directly; they do not require
precomputed STFT features.

4. Train proxy models. The main shared-backbone entry point is:

```bash
python train_shared_backbone.py --target_dir ./asd_dataset_logmel --matrix_log_mode raw --mode all --save_model_dir ./saved_proxy_lite
```

Separate task entry points are:

```text
train_ae.py
train_classifier.py
train_contrastive.py
train_separation.py
```

These scripts run the target/configuration specified by CLI options. Use
`--target_classes`, `--margins`, or `--resnets` where available to launch
explicit grids.

Paper-oriented separate-task grids are:

```text
AE              --hidden_dims 64 128 256 --latent_dims 4 8 16      # 9 configs
Classifier CE   --mode ce --resnets resnet18 resnet34 resnet50 resnet101 resnet152
Classifier Arc  --mode arcface --resnets resnet18 resnet34 resnet50 resnet101 resnet152
Contrastive     --mode simclr  --segment_frames 313 --resnets resnet18 resnet34 resnet50 resnet101 resnet152
Contrastive     --mode simsiam --segment_frames 313 --resnets resnet18 resnet34 resnet50 resnet101 resnet152
Separation      --channels 64 128 --cbs 0 1 2 4 --snr_list -5 -4 -3 -2 -1 0 1 2 3 4 5 # 8 configs
```

## Paper Configuration Summary

Full model-specific hyperparameters are listed in
`docs/paper_model_configurations.csv`. The table also expands shared-backbone
rows by task and Lite0-Lite4 index for the separate shared-backbone analysis.

| Family | Configurations | Main varied parameters | Full table |
| --- | ---: | --- | --- |
| Auto-Encoder | 9 | latent 4/8/16 x hidden 64/128/256 | `docs/paper_model_configurations.csv` |
| Classification (CE) | 5 | ResNet18/34/50/101/152 | `docs/paper_model_configurations.csv` |
| Classification (ArcFace) | 5 | ResNet18/34/50/101/152, margin 0.5 | `docs/paper_model_configurations.csv` |
| Separation | 8 | CB 0/1/2/4 x channels 64/128 | `docs/paper_model_configurations.csv` |
| Contrastive SimCLR | 5 | ResNet18/34/50/101/152 | `docs/paper_model_configurations.csv` |
| Contrastive SimSiam | 5 | ResNet18/34/50/101/152 | `docs/paper_model_configurations.csv` |
| Pre-trained | 8 | EAT, BEATs, and CED variants | `docs/paper_model_configurations.csv` |
| Shared backbone | 30 | 6 task heads x Lite0-Lite4 | `docs/paper_model_configurations.csv` |

## Runtime Summary

To measure runtime for every paper configuration, use
`docs/full_runtime_measurements_template.csv`,
`docs/full_runtime_measurement_plan.md`, and
`scripts/measure_full_runtime_grid.sh`. The helper defaults to one-epoch
timing.
After the run completes, create the publishable table with
`scripts/summarize_runtime_grid.py`; it writes the completed runtime table under
`docs/` after all expected rows have finished.

The completed one-epoch runtime table used for release is
`docs/full_runtime_measurements_completed.csv`. All 75 expected rows are marked
`completed`.

| Family | Configurations | Training wall-clock range | Evaluation wall-clock range |
| --- | --- | ---: | ---: |
| Auto-Encoder | 9 | 69.83-74.65 s | N/A |
| Classification CE | 5 | 123.88-186.99 s | N/A |
| Classification ArcFace | 5 | 1288.73-1962.19 s | N/A |
| Separation | 8 | 1186.81-3846.73 s | N/A |
| SimCLR | 5 | 2353.88-2486.72 s | N/A |
| SimSiam | 5 | 2269.57-2486.21 s | N/A |
| Pre-trained | 8 | N/A | 570.00-954.99 s |
| Shared backbone | 30 | 37.37-154.54 s | N/A |

Runtime values are provided as reproducibility information, not as the primary
comparison metric.

The same grids are available as shell wrappers:

```bash
bash scripts/reproduce_public_grid_ae.sh
bash scripts/reproduce_public_grid_classifier.sh
bash scripts/reproduce_public_grid_contrastive.sh
bash scripts/reproduce_public_grid_separation.sh
```

Set `ASD_DATASET_ROOT`, `ASD_LOGMEL_ROOT`, and method-specific save-root
variables before running wrappers. They do not activate conda environments or
contain local machine paths.

5. Evaluate proxy metrics and ASD metrics:

```bash
python evaluate_proxy_asd.py --data_dir ./asd_dataset --model_root ./saved_exp --save_dir ./batch_eval_results
python evaluate_shared_backbone.py --data_dir ./asd_dataset_logmel --matrix_log_mode raw --model_root ./saved_proxy_lite --save_dir ./shared_eval_results
```

For paper-style task-specific evaluation, make the LP split and separation
proxy sample count explicit:

```bash
python evaluate_proxy_asd.py --data_dir ./asd_dataset --model_root ./saved_exp --save_dir ./batch_eval_results --linear_half_split per_section --sep_snr -5 0 5 --sep_proxy_k 1000
python evaluate_shared_backbone.py --data_dir ./asd_dataset_logmel --matrix_log_mode raw --model_root ./saved_proxy_lite --save_dir ./shared_eval_results --linear_half_split per_section --snr_db -5 0 5
```

Paper-oriented proxy and ASD metric names are:

```text
AE proxy      -> test_normal_l1
Classifier    -> global_macro_f1
Contrastive   -> uniformity, sign-adjusted for correlation analysis
Pre-trained   -> pretrain_map / mAP
in-domain LP  -> linear_half_auc
out-domain LP -> linear_loso_auc
MD            -> mah_train_auc
```

For BEAT, CED, or EAT pre-trained representation baselines, use:

```bash
python evaluate_pretrain.py --data_dir ./asd_dataset --models eat ced beat --save_dir ./pretrain_eval_results
bash scripts/reproduce_paper_pretrained.sh
```

Full Figure 4 / Table 9 reproduction requires all eight pre-trained variants.
EAT and CED use public Hugging Face model ids by default. BEAT variants require
external checkpoints:

```bash
REQUIRE_FULL_PRETRAIN_GRID=1 \
BEATS_ITER3_CHECKPOINT="./beats/BEATs_iter3.pt" \
BEATS_ITER3_PLUS_CHECKPOINT="./beats/BEATs_iter3_plus_AS2M.pt" \
bash scripts/reproduce_paper_pretrained.sh
```

Without BEAT checkpoints, BEAT rows cannot be reproduced.

6. Aggregate and visualize:

```bash
python analyze_proxy_correlations.py \
  --summary_csv \
    ./batch_eval_results/results_summary.csv \
    ./pretrain_eval_results/*/results_summary.csv \
  --out_csv ./paper_table9_correlations.csv \
  --cache_prefix ./paper_table9_input \
  --validate_paper_counts \
  --strict_paper_counts
python plot_paper_figure4.py \
  --summary_csv \
    ./batch_eval_results/results_summary.csv \
    ./pretrain_eval_results/*/results_summary.csv \
  --out_path ./paper_figure4.png \
  --cache_csv ./paper_figure4_cache.csv \
  --validate_paper_counts \
  --strict_paper_counts
```

`analyze_proxy_correlations.py` generates the Table 9-style correlation table.
`plot_paper_figure4.py` uses the paper defaults for AE, classifier,
contrastive, pre-trained, and ASD metrics. Use
`plot_proxy_asd_summary.py --print_correlations` only as a diagnostic check.

Shared-backbone summaries are a separate analysis path:

```bash
python plot_proxy_asd_summary.py \
  --summary_csv ./shared_eval_results/results_summary.csv \
  --out_path ./shared_proxy_asd_summary.png \
  --cache_csv ./shared_proxy_asd_cache.csv
```

## Smoke Test

After installing the requirements, run these non-training checks:

```bash
python prepare_dataset_features.py --help
python rename_dataset_files.py --help
python train_ae.py --help
python train_classifier.py --help
python train_contrastive.py --help
python train_separation.py --help
python train_shared_backbone.py --help
python evaluate_proxy_asd.py --help
python evaluate_shared_backbone.py --help
python evaluate_pretrain.py --help
python plot_proxy_asd_summary.py --help
python analyze_proxy_correlations.py --help
python plot_paper_figure4.py --help
python tests/smoke_test_dataloaders.py
```

Do not run full training jobs as a smoke test.
The shared-backbone smoke commands require `timm`; run them after installing
`requirements.txt`.

`tests/smoke_test_dataloaders.py` creates a synthetic two-device dataset and
does not read real ASD data. The same smoke-test policy is summarized in
`docs/smoke_tests.md`.
