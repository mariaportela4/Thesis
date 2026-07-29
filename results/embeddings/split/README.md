# results/embeddings/split/

Train/validation/test partitions of each embedding type, aligned to the GraphPart 40% split (`../../../alignment/40_percent/` and `../../../data/final_dataset_40.csv`). Produced by `../../../prepare_split_embeddings.py`.

## Files

For each embedding type (`esm`, `bert`, `prott5`): `{type}_train.npy`, `{type}_val.npy`, `{type}_test.npy`.

Labels (shared across all three embedding types, same row order): `y_train.npy`, `y_val.npy`, `y_test.npy`.

## Notes

- `y_test.npy` is cross-checked against CSV-derived labels in `../../../src/benchmark_comparison.py` as a hard safety check against silent data-alignment bugs.
- These are the arrays consumed directly by `../../../nn/train_dnn.py`, `../../../nn/train_cnn.py`, and the benchmark script.
