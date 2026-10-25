# results/

Final benchmark outputs comparing all 10 models on the 176-sequence held-out test set (58 positive, 118 negative): the 6 thesis models (DNN/CNN × ESM-2/ProtBERT/ProtT5-XL) plus retrained DePP, DepoScope, DposFinder and PhageDPO. All produced by `../src/benchmark_comparison.py`.

## Files

- `benchmark_metrics.csv` — F1, precision, recall, PR-AUC and accuracy for all 10 models. Current ranking by F1: `dnn_prott5` and `dnn_esm` (0.9913), `dposfinder_retrained` (0.9828), `cnn_prott5` (0.9474), `cnn_esm` (0.9391), `dnn_bert` (0.9358), `phagedpo_retrained` (0.9107), `depp_retrained` (0.9009), `deposcope_retrained` (0.8972), `cnn_bert` (0.8296).
- `benchmark_barplot.png` — precision, recall and F1 for all 10 models, grouped into thesis models and retrained external tools. This is the figure used in the thesis.
- `benchmark_heatmap.png`, `benchmark_dotplot.png`, `benchmark_radar.png` — alternative views of the same metrics.
- `confusion_matrix_*.png` — one confusion matrix per model.
- `error_analysis.csv` — per-sequence predictions and error flags across the six thesis models, produced by `../src/nn/error_analysis.py`.

## Subfolders

- `cdhit/` — CD-HIT clustering output (80% identity), run via command line
- `graphpart/` — GraphPart split assignments and reports (40% and 60%)
- `embeddings/` — protein language model embeddings (full set and train/val/test split)
- `models/`, `history/` — trained model weights and per-epoch training curves
- `tuning/` — Optuna hyperparameter search results
- `multiseed/` — multi-seed stability evaluation of the six thesis models
- `depp_retrain/`, `deposcope_retrain/`, `dposfinder_retrain_trainonly/`, `phagedpo_retrain/` — retrained external tool outputs
- `split_identity/` — train-test identity audit of the GraphPart split
- `figures/` — figures assembled for the thesis

## Notes

- All ten models are evaluated at a fixed decision threshold of 0.5.
- `error_analysis_all_models.csv` is an earlier version of `error_analysis.csv`, produced before the final training runs. It does not match the reported metrics and should not be used.
- `deposcope_retrain_backup/` and `results/dposfinder_retrain/` hold superseded runs, kept only for reference. The benchmark reads `dposfinder_retrain_trainonly/`.