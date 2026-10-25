# results/tuning/

Optuna hyperparameter search results, produced by `../../src/nn/tune_hyperparams.py`.

## Files

`{arch}_{embedding}_best_params.json` — one per (architecture, embedding) combination, recording the selected hyperparameters, the validation loss of the selected trial at its stopping epoch, the seed, the early stopping settings, and that trial's stopping and best epochs.

## Notes

- `best_val_loss` is the loss at the **stopping** epoch of the best trial, which is what the search optimised, not the lowest loss reached during that trial.
- Each search ran 25–30 trials with a 30-epoch ceiling per trial, lower than the 50 epochs allowed in final training.
- The sampler is seeded (42), so all six searches explored the same initial configurations. `cnn_prott5` and `cnn_bert` consequently selected identical hyperparameters, with different resulting losses.
- For `cnn_esm` and `cnn_bert`, the selected trial reached the 30-epoch ceiling while still improving, so their configurations were chosen on truncated runs.
- `.bak` files are superseded copies and are not read by anything.