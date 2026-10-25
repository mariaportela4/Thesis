# src/nn/

Model definitions, training scripts, hyperparameter tuning and evaluation utilities for the two thesis classifier architectures (DNN, CNN), each trained on each of the three embedding types — six models in total.

## Files

- `dnn.py` — `DNNClassifier`: four linear layers (512, 256, 64, 1), BatchNorm on the first two hidden layers, dropout after each.
- `cnn.py` — `CNNClassifier`: two Conv1d blocks (64 and 128 channels, kernel 5) over the embedding dimension, global max-pooling, then a 128 → 64 → 1 head.
- `train_dnn.py`, `train_cnn.py` — training scripts, one run per embedding type. Early stopping with a 50-epoch ceiling (patience 5, min delta 1e-4); all random number generators seeded via `--seed` (default 42).
- `tune_hyperparams.py` — Optuna search (learning rate, batch size, weight decay, dropout) per (architecture, embedding) combination, with a 30-epoch ceiling per trial and a MedianPruner. The objective is the validation loss at the stopping epoch, matching what training actually retains. Outputs to `../../results/tuning/`.
- `multiseed_eval.py` — trains each combination across several seeds and reports mean ± std per metric, for stability analysis. Outputs to `../../results/multiseed/`.
- `error_analysis.py` — per-sequence error analysis across the six models, flagging sequences misclassified by three or more. Outputs `../../results/error_analysis.csv`.
- `metrics.py` — shared evaluation and plotting utilities.

## Notes

- Training keeps the weights of the **stopping** epoch, not the best epoch. Both are recorded in `../../results/history/*_summary.json`.
- The Optuna sampler is seeded, so every search explores the same initial configurations. This is why `cnn_prott5` and `cnn_bert` selected identical hyperparameters despite being separate searches.
- The tuning JSON files contain training parameters (`lr`, `batch_size`) that must be excluded when instantiating a model: `DNNClassifier` accepts only `input_dim` and the three dropout rates; `CNNClassifier` only `input_dim` and `dropout`.