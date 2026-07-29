# results/depp_retrain/

DePP (Depolymerase-Predictor), retrained on the thesis dataset for a fair benchmark. The retraining script (`depp_retrain.py`) lives inside the external `Depolymerase-Predictor/DePP_CLI/` repo (gitignored), not in this repository's `src/`.

## Files

- `depp_training_set_custom.csv` — training set formatted for DePP's expected input.
- `depp_test_features.csv` — extracted features for the held-out test set.
- `depp_retrained_predictions.csv` — predictions on the test set.
- `depp_retrain_metrics.csv` — final test-set metrics, consumed by `../../src/benchmark_comparison.py`.

## Compatibility notes

- DePP requires a monkey-patch shim for `ProteinAnalysis.get_amino_acids_percent()`, which was removed in Biopython 1.87+.
- Sequences with ambiguous amino acids (X/B/Z/J/U/O) must be removed before running DePP — it does not handle them natively.

## Notes

- Current benchmark result: F1≈0.832–0.833, in the same range as retrained DepoScope, below the top thesis models.