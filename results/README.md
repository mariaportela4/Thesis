# results/

Final benchmark outputs comparing all 8 models on the 178-sequence held-out test set (`../alignment/40_percent/40pct_test.fasta`): the 6 thesis models (DNN/CNN × ESM-2/ProtBERT/ProtT5-XL) plus retrained DepoScope and retrained DePP.

## Files

- `benchmark_metrics.csv` — F1, AUC, precision, recall, etc. for all 8 models. Current leaderboard: `dnn_prott5` best (F1≈0.915, AUC≈0.986), then `cnn_prott5`, `dnn_esm`, `dnn_bert`, then `deposcope_retrained`/`depp_retrained` (F1≈0.832–0.833), with `cnn_esm`/`cnn_bert` lowest.
- `benchmark_barplot.png` — visual comparison of all 8 models across metrics.
- `confusion_matrix_*.png` — one confusion matrix per model (`cnn_bert`, `cnn_esm`, `cnn_prott5`, `dnn_bert`, `dnn_esm`, `dnn_prott5`, `deposcope_retrained`, `depp_retrained`).

All produced by `../src/benchmark_comparison.py`.

## Subfolders

- `cdhit/` — CD-HIT clustering output (80% identity), run via command line
- `graphpart/` — GraphPart split assignments and reports (40% and 60%)
- `embeddings/` — protein language model embeddings (full set and train/val/test split)
- `models/` — trained model weights (6 thesis models)
- `history/` — per-epoch training curves
- `deposcope_retrain/` — retrained DepoScope outputs
- `depp_retrain/` — retrained DePP outputs

## Notes on the DNN+ProtT5 / DNN+ProtBERT recall=1.0 anomaly

Investigated and confirmed **not** to be data leakage: FN=0 with non-trivial precision, and training history (see `history/`) confirms genuine overfitting behavior. Best-checkpoint saving (`../nn/README.md`) is in place, so reported test metrics correspond to the best validation-loss epoch.