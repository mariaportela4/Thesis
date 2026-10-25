# results/deposcope_retrain/

DepoScope, retrained on the thesis dataset. Produced by `../../src/deposcope_retrain.py`.

## Files

- `deposcope_retrained_cnn.pt` — retrained CNN head weights.
- `esm2_token_features.npz` — cached ESM-2 token-level features used as DepoScope's input representation (its architecture consumes residue-level labels, unlike the thesis models, which use pooled embeddings).
- `training_history.json` — per-epoch training history.
- `test_metrics.json` — final test-set metrics.
- `test_probabilities.npy` — predicted probabilities on the held-out test set, read by `../../src/benchmark_comparison.py`.

## Notes

- Only DepoScope's CNN head was retrained. Its ESM-2 component, fine-tuned by the authors on residue-level depolymerase annotations, was kept as released: retraining it would require annotations of that kind, which this dataset does not have.
- Because of that, DepoScope enters the benchmark already trained on labelled examples of the target class, with annotations no other model received.
- Early stopping uses the dataset's own validation partition (the same 88 sequences used by the thesis models), rather than holding out a fraction of the training set as the original procedure does. The head retains the weights of the lowest-validation-loss epoch.
- Benchmark result: F1 0.8972, PR-AUC 0.8953 — the highest precision in the benchmark (0.9796) combined with the lowest recall (0.8276).
- DepoScope's weights are downloaded from Zenodo on first run and are gitignored.
- `../deposcope_retrain_backup/` holds a superseded run and is not read by anything.