# results/embeddings/split/

Train/validation/test partitions of each embedding type, aligned to the GraphPart 40% split. Produced by `../../../src/prepare_split_embeddings.py`.

## Files

For each embedding type (`esm`, `bert`, `prott5`): `{type}_train.npy`, `{type}_val.npy`, `{type}_test.npy`.

Labels, shared across all three types in the same row order: `y_train.npy`, `y_val.npy`, `y_test.npy`.

## Notes

- Partitioning is read from the `split` column of `../../../data/final_dataset_40.csv`, so it is fully deterministic — no random splitting happens here.
- `y_test.npy` is cross-checked against CSV-derived labels in `../../../src/benchmark_comparison.py` as a hard safety check against silent alignment bugs.
- These arrays are consumed directly by `../../../src/nn/train_dnn.py`, `../../../src/nn/train_cnn.py` and the benchmark script.