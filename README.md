# Thesis – Bacteriophage Depolymerase (DPO) Classifier

Master's thesis project: binary classification of bacteriophage depolymerases vs. non-DPO proteins, building on the PhageDPO framework. Combines three protein language model embeddings (ESM-2 650M, ProtBERT, ProtT5-XL) with two classifier architectures (DNN, CNN) — six thesis models — benchmarked against retrained versions of DePP, DepoScope, DposFinder and PhageDPO.

## Running the pipeline

`run_pipeline.py` drives every stage in the canonical order. `python run_pipeline.py --check` verifies that all paths, environments and external repositories resolve before running anything, and `--list` prints the terminal-only commands (CD-HIT, GraphPart).

Two conda environments are required: the main one (`environment.yml`) and DposFinder's own (`environment-dposfinder.yml`), whose dependencies conflict with the main one.

## Repository structure

Each folder below has its own `README.md` describing its contents in detail.

- `data/` — raw and processed sequence data (positive/negative sets, deduplication, final split-reference CSVs)
- `src/` — pipeline scripts: dataset construction, benchmarking, external tool retraining
  - `src/datasets/` — dataset construction and audit scripts
  - `src/embedding/` — scripts to generate protein language model embeddings
  - `src/nn/` — model class definitions (DNN, CNN), training, tuning, error analysis
- `results/embeddings/` — generated embeddings (full set and train/val/test split `.npy` arrays)
- `results/cdhit/` — CD-HIT clustering output (80% identity)
- `results/graphpart/` — GraphPart split assignments and reports
- `results/models/`, `results/history/` — trained model weights and training curves
- `results/tuning/` — hyperparameter search results (Optuna)
- `results/multiseed/` — multi-seed stability evaluation of the six thesis models
- `results/depp_retrain/`, `results/deposcope_retrain/`, `results/dposfinder_retrain_trainonly/`, `results/phagedpo_retrain/` — retrained external tools (benchmark baselines)
- `results/split_identity/` — train-test identity audit of the GraphPart split
- `notebooks/` — exploratory analysis

## Pipeline order

1. Combine the per-domain positive FASTA files (`src/datasets/combine_domains.py`) → `all_DPO_candidates.fasta` (2,041 entries, 1,949 unique records)
2. Remove hypothetical proteins (`src/datasets/remove_hypothetical_proteins.py`)
3. Remove duplicate sequences (`src/remove_duplicates.py`), with RefSeq > GenBank > PDB priority for the positive set
4. CD-HIT clustering at 80% identity — **run via command line** (see `results/cdhit/README.md`)
5. Class balancing to a 1:2 positive:negative ratio, sampled **from the CD-HIT-clustered negative set** with `random.seed(42)`, then merged (`src/datasets/balance_and_merge.py`) → `data/merged_input.fasta`
6. Normalize ambiguous residues (B, Z, J, U, O → X) (`src/normalize_ambiguous_aa.py`) → `data/merged_input_normalized.fasta`
7. GraphPart split at 40% identity (`--test-ratio 0.1 --val-ratio 0.05`) — **run via command line**, depends on EMBOSS `needleall` (see `results/graphpart/README.md`)
8. Build the canonical dataset table (`src/datasets/build_final_dataset.py`) → `data/final_dataset_40.csv` (train 1,515 / valid 88 / test 176)
9. Export split FASTAs → embeddings → model training → benchmark

A parallel run was also done at 60% GraphPart identity for comparison, but **only the 40% split was used for the final thesis results**.

## Notes for anyone reading this repo

- `data/final_dataset_40.csv` is the single source of truth for the split. Every downstream stage reads it.
- External tool weights and repositories (`deposcope_weights/`, `Depolymerase-Predictor/`, `../DposFinder`) and feature caches are gitignored — see the individual READMEs for how they are regenerated.