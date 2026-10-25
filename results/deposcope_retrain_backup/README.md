# results/deposcope_retrain/

DepoScope, retrained on the thesis dataset. Produced by `../../src/deposcope_retrain.py`.

## Files

- `deposcope_retrained_cnn.pt` — retrained DepoScope model weights.
- `esm2_token_features.npz` — ESM-2 token-level features used as DepoScope's input representation (DepoScope's architecture differs from the thesis models, which use pooled/sequence-level embeddings).
- `training_history.json` — per-epoch training history for the retraining run.
- `test_metrics.json` — final test-set metrics for this model.
- `test_probabilities.npy` — predicted probabilities on the held-out test set, used by `../../src/benchmark_comparison.py`.

## Notes

- Current benchmark result: F1≈0.832–0.833, placing it below all thesis DNN models and `cnn_prott5`, but above `cnn_esm`/`cnn_bert`.
