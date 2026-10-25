# results/history/

Per-epoch training curves and run summaries for each of the six thesis models, produced by `../../src/nn/train_dnn.py` and `../../src/nn/train_cnn.py`.

## Files

- `{arch}_{embedding}_history.csv` — per-epoch training and validation loss.
- `{arch}_{embedding}_summary.json` — seed, stopping epoch and epoch of lowest validation loss for that run.

## Stopping epochs

| Model | Stop epoch | Best epoch |
|---|---|---|
| dnn_esm | 8 | 3 |
| dnn_bert | 11 | 6 |
| dnn_prott5 | 15 | 10 |
| cnn_prott5 | 23 | 18 |
| cnn_bert | 35 | 30 |
| cnn_esm | 48 | 43 |

## Notes

- Training uses early stopping with a 50-epoch ceiling; the weights kept are those of the stopping epoch, so the gap between the two columns above is expected.
- Four of the six runs stopped at exactly the same epoch as the corresponding Optuna trial (see `../tuning/`), which confirms that the seeded runs are reproducible.