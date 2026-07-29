# results/history/

Per-epoch training curves (loss/metrics) for each of the six thesis models, produced by `../../nn/train_dnn.py` and `../../nn/train_cnn.py`.

## Files

- `dnn_esm_history.csv`, `dnn_bert_history.csv`, `dnn_prott5_history.csv`
- `cnn_esm_history.csv`, `cnn_bert_history.csv`, `cnn_prott5_history.csv`

Each file logs per-epoch training/validation loss and metrics.

## Notes

- These curves show late-epoch overfitting for some models (e.g. DNN+ProtT5, DNN+ProtBERT) — this is expected given best-checkpoint saving (`val_loss < best_val_loss`) is used, so the reported test-set model corresponds to the best-validation-loss epoch, not the final one.