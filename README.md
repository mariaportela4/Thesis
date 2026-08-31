# Thesis – Bacteriophage Depolymerase (DPO) Classifier

Master's thesis project: binary classification of bacteriophage depolymerases vs. non-DPO proteins, building on the PhageDPO framework. Combines three protein language model embeddings (ESM-2 650M, ProtBERT, ProtT5-XL) with two classifier architectures (DNN, CNN) — six thesis models — benchmarked against retrained versions of DepoScope and DePP.

## Repository structure

Each folder below has its own `README.md` describing its contents in detail.

- `data/` — raw and processed sequence data (positive/negative sets, deduplication, final split-reference CSVs)
- `src/` — pipeline scripts: benchmarking, external tool retraining, data preparation, deduplication
  - `src/embedding/` — scripts to generate protein language model embeddings
  - `src/nn/` — model class definitions (DNN, CNN), training scripts, hyperparameter tuning, shared evaluation metrics
  - `src/datasets/` — data-fetching utilities
- `results/embeddings/` — generated embeddings (full set and train/val/test split `.npy` arrays)
- `results/cdhit/` — CD-HIT clustering output (80% identity)
- `results/graphpart/` — GraphPart split assignments and reports
  - `results/graphpart/alignment/` — CD-HIT + GraphPart output splits (train/valid/test FASTA), at 40% and 60% identity thresholds
- `results/models/`, `results/history/` — trained model weights and training curves
- `results/tuning/` — hyperparameter search results (Optuna)
- `results/deposcope_retrain/`, `results/depp_retrain/` — retrained external tools (benchmark baselines)
- `notebooks/` — exploratory analysis

## Pipeline order

1. Remove hypothetical proteins from the raw/candidate sequence sets (`all_DPO_candidates.fasta` → `all_DPO_positive.fasta`; `negative.faa` → `negative_nohyp.faa`)
2. Remove duplicate sequences (`src/remove_duplicates.py`) → `all_DPO_positive_dedup.fasta`, `negative_dedup.faa`
3. Normalize ambiguous amino acid residues (B, Z, J, U, O → X) across the deduplicated sequence sets (`src/normalize_ambiguous_aa.py`) — ensures CD-HIT and GraphPart cluster on consistent sequence content
4. CD-HIT clustering at 80% identity — **run via command line**, not from a repo script (see `results/cdhit/README.md`)
5. Class balancing (1:2 positive:negative ratio) — sampled **from the CD-HIT-clustered negative set**, using `random.seed(42)` for reproducibility (e.g. 593 CD-HIT positives → sample 1186 = 593×2 negatives)
6. Merge positive + negative into `data/merged_input_normalized.fasta` (sequential IDs, label/orig_id encoded in the FASTA header)
7. GraphPart split at 40% identity (`--test-ratio 0.1 --val-ratio 0.05`) on `data/merged_input_normalized.fasta` — **run via command line**, depends on EMBOSS `needleall` (see `results/graphpart/README.md` and `logs/README.md`)

A parallel run was also done at 60% GraphPart identity for comparison, but **only the 40% split was used for the final thesis results** (see `results/graphpart/alignment/README.md`).

## Notes for anyone reading this repo
- External tool weights/repos (`deposcope_weights/`, `Depolymerase-Predictor/`) and feature caches are gitignored — see individual READMEs for how they're regenerated.
