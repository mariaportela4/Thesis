# alignment/

Per-split FASTA files written from the GraphPart assignments, at two identity thresholds. Produced by `../../../src/export_split_fasta.py` and used for the pairwise identity audit rather than by the training pipeline itself.

## Subfolders

- `40_percent/` — **used for the final thesis results.**
- `60_percent/` — exploratory comparison run. **Not used in the final thesis results.**

## How these were generated

CD-HIT and GraphPart are both **run via the command line** (see `../README.md` and `../../cdhit/README.md`); `run_pipeline.py` prints the exact commands with `--list`. The FASTA files here are then exported from `../../../data/final_dataset_40.csv`.

Pipeline order leading here: combine domain FASTAs → remove hypothetical proteins → remove duplicates → CD-HIT 80% → class balancing (1:2, sampled from the CD-HIT negative output) → merge → normalise ambiguous residues → GraphPart. See the root `README.md` for the full sequence.