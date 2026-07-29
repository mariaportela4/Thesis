# nn/

Model class definitions and training scripts for the two thesis classifier architectures (DNN, CNN), each trained on each of the three embedding types (ESM-2, ProtBERT, ProtT5-XL) — six thesis models in total.

## Files

- `dnn.py` — `DNNClassifier` model class.
- `cnn.py` — `CNNClassifier` model class.
- `metrics.py` — shared evaluation metrics (F1, AUC, precision/recall, etc.) used across training and benchmarking.
- `train_dnn.py` — training script for DNN models (run once per embedding type: ESM-2, ProtBERT, ProtT5-XL).
- `train_cnn.py` — training script for CNN models (same, per embedding type).
- `__pycache__/` — compiled bytecode, not tracked.

## Notes

- Best-checkpoint saving (`val_loss < best_val_loss`) is implemented in both training scripts — the model evaluated in `../results/models/` corresponds to the best validation-loss epoch, not the final epoch. Loss curves in `../results/history/` showing late overfitting are therefore expected and not contradictory with reported test metrics.
- Trained weights are written to `../results/models/`; per-epoch training curves to `../results/history/`.