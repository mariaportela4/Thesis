# results/models/

Trained weights for the six thesis classifier models (2 architectures × 3 embedding types), produced by `../../nn/train_dnn.py` and `../../nn/train_cnn.py`.

## Files

- `dnn_esm.pt`, `dnn_bert.pt`, `dnn_prott5.pt` — DNN classifier, trained on ESM-2 / ProtBERT / ProtT5-XL embeddings respectively.
- `cnn_esm.pt`, `cnn_bert.pt`, `cnn_prott5.pt` — CNN classifier, trained on the same three embedding types.

## Notes

- Each checkpoint corresponds to the epoch with lowest validation loss (best-checkpoint saving), not the final training epoch — see `../../nn/README.md`.
- `dnn_prott5.pt` is currently the best-performing model on the held-out test set (F1≈0.915, AUC≈0.986) — see `../benchmark_metrics.csv`.
- `cnn_bert.pt` is consistently the weakest of the six (hypothesized architecture–embedding mismatch between CNN and ProtBERT).