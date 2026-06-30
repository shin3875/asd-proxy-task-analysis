# Smoke Tests

Use smoke tests only to check that public entry points and dataloaders are
importable. Do not use full training jobs as smoke tests.

## Help Commands

```bash
python prepare_dataset_features.py --help
python train_ae.py --help
python train_classifier.py --help
python train_contrastive.py --help
python train_separation.py --help
python train_shared_backbone.py --help
python evaluate_proxy_asd.py --help
python evaluate_shared_backbone.py --help
python evaluate_pretrain.py --help
python analyze_proxy_correlations.py --help
python plot_paper_figure4.py --help
```

## Synthetic Dataloader Smoke

```bash
python tests/smoke_test_dataloaders.py
```

This test creates a tiny synthetic two-device dataset and does not read real
ASD data.
