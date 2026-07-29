# src/

Core pipeline scripts: benchmarking and external tool retraining.

## Files

- `benchmark_comparison.py` — runs all 8 models (6 thesis models + retrained DepoScope + retrained DePP) on the held-out test set (`../alignment/40_percent/40pct_test.fasta`) and produces the final comparison outputs: `../results/benchmark_metrics.csv`, `../results/benchmark_barplot.png`, and per-model `../results/confusion_matrix_*.png`. Includes hard safety checks comparing `y_test.npy` against CSV-derived labels to catch silent data-alignment bugs.
- `deposcope_retrain.py` — retrains DepoScope on the thesis dataset (per advisor instruction, DepoScope is retrained rather than used off-the-shelf, for a fair benchmark). Outputs to `../results/deposcope_retrain/`.
- `make_train_fasta.py` — generates a split-specific FASTA file from `../data/final_dataset_40.csv`, e.g. `python make_train_fasta.py --dataset-csv ../data/final_dataset_40.csv --split test`. Used to (re)produce `../data/train_sequences_clean.fasta` and `../data/test_sequences_clean.fasta` aligned to the GraphPart 40% split.
- `prepare_split_embeddings.py` — splits the full embedding `.npy` files (`../results/embeddings/`) into train/val/test according to the GraphPart assignment, producing `../results/embeddings/split/`.

## Subfolders

- `datasets/` — data-fetching utilities.

## Notes

- DePP retraining (`depp_retrain.py`) lives inside `Depolymerase-Predictor/DePP_CLI/` (external repo, gitignored), not in this folder — see `../results/depp_retrain/README.md`.