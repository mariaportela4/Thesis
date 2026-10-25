# src/

Core pipeline scripts: dataset preparation, benchmarking and external tool retraining.

## Files

- `benchmark_comparison.py` — evaluates all 10 models on the held-out test set, sourced from `../data/final_dataset_40.csv` (split == 'test'), and produces `../results/benchmark_metrics.csv`, the benchmark figures and the per-model confusion matrices. Applies a 0.5 threshold uniformly and cross-checks `y_test.npy` against CSV-derived labels to catch silent alignment bugs.
- `depp_retrain.py` — retrains DePP's random forest, reusing the feature extraction and training code from the DePP repository (cloned externally, passed via `--depp-repo`). Sequences with non-standard residues are dropped before featurisation. Outputs to `../results/depp_retrain/`.
- `deposcope_retrain.py` — retrains DepoScope's CNN head on the thesis dataset, keeping its fine-tuned ESM-2 component frozen. Early stopping uses the dataset's own validation partition. Outputs to `../results/deposcope_retrain/`.
- `dposfinder_retrain.py` — retrains DposFinder through its own `code/main.py`, run as a subprocess in a separate conda environment. Converts the dataset into DposFinder's FASTA format and truncates sequences to 1024 residues. Outputs to `../results/dposfinder_retrain_trainonly/`.
- `phagedpo_retrain.py` — retrains PhageDPO's SVM on its own hand-crafted descriptor set, using the tool's grid search (10-fold CV, MCC). Outputs to `../results/phagedpo_retrain/`.
- `normalize_ambiguous_aa.py` — maps ambiguous residues (B, Z, J, U, O → X) across the merged FASTA, producing `../data/merged_input_normalized.fasta`, the canonical sequence source for every downstream branch.
- `make_train_fasta.py` — writes a split-specific FASTA from the canonical dataset table. Assumes its input FASTA is already normalised.
- `export_split_fasta.py` — writes per-split FASTA files keyed by `orig_id`, used for the pairwise identity audit rather than by the training pipeline.
- `prepare_split_embeddings.py` — partitions the full embedding arrays into train/val/test according to the split column.
- `remove_duplicates.py` — removes exact duplicates from the positive set, with RefSeq > GenBank > PDB priority and one representative chain per PDB structure.

## Subfolders

- `datasets/` — dataset construction and audit scripts
- `embedding/` — protein language model embedding generation
- `nn/` — model definitions, training, tuning and error analysis

## Notes

- All four external tools are retrained on the same training split (1,515 sequences) and evaluated on the same test set, so benchmark differences reflect the modelling approach rather than the data.