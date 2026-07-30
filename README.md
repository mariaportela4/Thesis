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

## Pending (post re-run)

- Steps 3–7 above reflect the pipeline as it will run after moving ambiguous-residue normalization earlier (before CD-HIT). The current `final_dataset_40.csv` split assignments were generated from CD-HIT and GraphPart runs on **non-normalized** sequences — both steps need to be re-run against the normalized input before results are consistent with this documented order.
- Downstream of the split, the following also need to be regenerated once normalized CD-HIT/GraphPart output is available:
  - Protein language model embeddings (`src/embedding/generate_embedding.py`) — whether all three embeddings actually change depends on whether any training sequences contain the residue `J` specifically (the previous embedding extraction already normalized B/Z/J/U/O → X, except J, which was silently dropped rather than replaced — see note in `src/embedding/README.md`)
  - `data/train_sequences_clean.fasta` / `test_sequences_clean.fasta` (`src/make_train_fasta.py`) — previously 1502/178 for DePP/DepoScope (11 training sequences dropped due to ambiguous residues); expected 1513/178 after normalization, matching the thesis models' split counts
  - The six thesis models (DNN/CNN × ESM-2/ProtBERT/ProtT5) — retraining required only if the embeddings actually change (see above)
  - DepoScope and DePP retraining (`src/deposcope_retrain.py`, `depp_retrain.py`) — retraining required regardless, since their training set gains 11 sequences
  - `results/benchmark_metrics.csv` and associated plots (`src/benchmark_comparison.py`) — final step, once all of the above are regenerated
- Hyperparameter tuning (`src/nn/tune_hyperparams.py`) has not yet been run to completion for all six thesis models — `results/tuning/` will be populated after that step.
- `src/datasets/ncbi_fetch.py` needs adjustments to produce a better-balanced positive/negative dataset upstream of CD-HIT (currently a large imbalance, e.g. ~500 positives vs. ~16000 negatives before CD-HIT).
- Additional result visualisations to add: heatmap, dot plot, and/or spider (radar) chart, alongside the existing bar plot in `results/benchmark_barplot.png`.
- Compute mean ± standard deviation across metrics by retraining the six thesis models with multiple random seeds (feasible only for the six thesis models, not for the retrained external tools).
- Benchmark against PhageDPO itself (the framework this thesis builds on), in addition to the current DepoScope/DePP comparison.