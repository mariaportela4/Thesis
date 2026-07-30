# src/nn/

Model class definitions, training scripts, hyperparameter tuning, and shared evaluation utilities for the two thesis classifier architectures (DNN, CNN), each trained on each of the three embedding types (ESM-2, ProtBERT, ProtT5-XL) — six thesis models in total.

## Files

- `dnn.py` — `DNNClassifier` model class. Dropout rates (per layer) are configurable, defaulting to the values used before hyperparameter tuning.
- `cnn.py` — `CNNClassifier` model class. Dropout rate is configurable, defaulting to the value used before hyperparameter tuning.
- `metrics.py` — shared evaluation utilities: `evaluate_model` (precision/recall/F1 + confusion matrix), `plot_confusion_matrix` (heatmap, configurable to display and/or save), `create_eval_loaders`, `compare_embeddings`, `plot_metric_by_model`, and training-history plotting.
- `train_dnn.py` — training script for DNN models (run once per embedding type: ESM-2, ProtBERT, ProtT5-XL).
- `train_cnn.py` — training script for CNN models (same, per embedding type).
- `tune_hyperparams.py` — Optuna-based hyperparameter search (lr, batch size, weight decay, dropout) per (architecture, embedding) combination. Minimises validation loss, consistent with the best-checkpoint criterion used in training. Outputs to `../../results/tuning/`.
- `__pycache__/` — compiled bytecode, not tracked.

## Notes

- Best-checkpoint saving (`val_loss < best_val_loss`) is implemented in both training scripts — the model evaluated in `../../results/models/` corresponds to the best validation-loss epoch, not the final epoch. Loss curves in `../../results/history/` showing late overfitting are therefore expected and not contradictory with reported test metrics.
- Trained weights are written to `../../results/models/`; per-epoch training curves to `../../results/history/`.