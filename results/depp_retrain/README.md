# results/depp_retrain/

DePP (Depolymerase-Predictor), retrained on the thesis dataset. Produced by `../../src/depp_retrain.py`, which reuses DePP's own feature extraction and random forest training code from the external clone passed via `--depp-repo` (gitignored).

## Files

- `depp_training_set_custom.csv` — training set in DePP's expected input format.
- `depp_test_features.csv` — extracted features for the held-out test set.
- `depp_retrained_predictions.csv` — test-set predictions.
- `depp_retrain_metrics.csv` — final metrics, read by `../../src/benchmark_comparison.py`.

## Compatibility notes

- DePP relies on `ProteinAnalysis.get_amino_acids_percent()`, removed in recent Biopython versions. The original behaviour is restored at runtime by a shim rather than by editing DePP's source, since the current replacement returns values on a different scale and would alter every feature.
- DePP's feature set is defined only over the 20 standard residues, so sequences containing any other character are dropped before featurisation: 43 of the 1,515 training sequences, and none of the 176 test sequences.

## Notes

- DePP is one of the two baselines built on hand-crafted descriptors rather than language model representations, alongside PhageDPO.
- Benchmark result: F1 0.9009, PR-AUC 0.9626.