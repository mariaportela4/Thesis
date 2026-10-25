# results/models/

Trained weights for the six thesis classifier models (2 architectures × 3 embedding types), produced by `../../src/nn/train_dnn.py` and `../../src/nn/train_cnn.py`.

## Files

- `dnn_esm.pt`, `dnn_bert.pt`, `dnn_prott5.pt` — DNN classifier, trained on ESM-2 / ProtBERT / ProtT5-XL embeddings respectively.
- `cnn_esm.pt`, `cnn_bert.pt`, `cnn_prott5.pt` — CNN classifier, trained on the same three embedding types.

## Notes

- Each checkpoint corresponds to the epoch at which early stopping halted training (patience 5, min delta 1e-4, maximum 50 epochs), not the epoch of lowest validation loss. Both epochs are recorded in `../history/*_summary.json`.
- All six were trained with seed 42; the seed is recorded in each summary file.
- `dnn_esm.pt` and `dnn_prott5.pt` are the best-performing models on the held-out test set (F1 0.9913, no false positives) — see `../benchmark_metrics.csv`.
- `cnn_bert.pt` is the weakest of the six (F1 0.8296), driven by 21 false positives.