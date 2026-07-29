# results/graphpart/

GraphPart split assignments — the final step of the pipeline, producing the train/valid/test partition used everywhere downstream. **Run via the command line** on `tesla`, dependent on EMBOSS `needleall` for pairwise sequence similarity.

## Files

- `graphpart_assignments_40.csv` / `graphpart_assignments_40_report.json` — **split used for all final thesis results.** Sequence-to-partition (train/valid/test) assignment and run report at 40% identity threshold (`--test-ratio 0.2 --val-ratio 0.1`).
- `graphpart_assignments_60.csv` / `graphpart_assignments_60_report.json` — equivalent output at 60% identity threshold. Exploratory only, **not used in the final thesis results.**

## Command used

Run for both thresholds, using `needle` (global alignment) mode:

```bash
graphpart needle \
  --fasta-file data/merged_input.fasta \
  --threshold 0.4 \
  --out-file results/graphpart/graphpart_assignments_40.csv \
  --labels-name label \
  --test-ratio 0.2 \
  --val-ratio 0.1 \
  --threads 12
```

```bash
graphpart needle \
  --fasta-file data/merged_input.fasta \
  --threshold 0.6 \
  --out-file graphpart_assignments_60.csv \
  --labels-name label \
  --test-ratio 0.2 \
  --val-ratio 0.1 \
  --threads 12
```

### Flags

- `needle` — subcommand selecting global alignment (Needleman-Wunsch, via EMBOSS `needleall`) as the similarity method — a good choice for proteins.
- `--fasta-file` — input FASTA (merged positive + negative set).
- `--threshold` — identity cutoff for the partitioning graph (0.4 or 0.6).
- `--labels-name label` — reads the `label=positive`/`label=negative` field from FASTA headers.
- `--test-ratio 0.2` — 20% of sequences assigned to the test partition.
- `--val-ratio 0.1` — 10% of sequences assigned to the validation partition.
- `--threads 12` — 12 threads.
- output — CSV with per-sequence partition assignment (train/val/test).

## Critical dependency note

GraphPart requires `EMBOSS_ACDROOT` and `EMBOSS_DATA` environment variables to be set correctly (pointing at the conda environment's EMBOSS install) or `needleall` fails **silently**, producing zero graph edges and an invalid random-looking split with no error raised. This was diagnosed and fixed by exporting these variables via the conda activation scripts for `thesis_env`. Always check the edge count in the `_report.json` file before trusting a GraphPart output. See `../../logs/README.md` for the corresponding run logs.

Output feeds into `../../alignment/40_percent/` (and `60_percent/`) and is the reference split recorded in `../../data/final_dataset_40.csv`.