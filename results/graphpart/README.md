# results/graphpart/

GraphPart split assignments — the partitioning step that produces the train/valid/test split used everywhere downstream. **Run via the command line**, dependent on EMBOSS `needleall` for pairwise sequence similarity.

## Files

- `graphpart_assignments_40.csv` / `graphpart_assignments_40_report.json` — **split used for all final thesis results.** Sequence-to-partition assignment and run report at a 40% identity threshold.
- `graphpart_assignments_60.csv` / `graphpart_assignments_60_report.json` — equivalent output at 60% identity. Exploratory only, **not used in the final thesis results.**

GraphPart writes numeric cluster IDs rather than split names; `../../src/datasets/build_final_dataset.py` maps them to train/test/valid by partition size and writes `../../data/final_dataset_40.csv`.

## Command used

Run for both thresholds, using `needle` (global alignment) mode:

```bash
graphpart needle \
  --fasta-file data/merged_input_normalized.fasta \
  --threshold 0.4 \
  --out-file results/graphpart/graphpart_assignments_40.csv \
  --labels-name label \
  --test-ratio 0.1 \
  --val-ratio 0.05 \
  --threads 12
```

```bash
graphpart needle \
  --fasta-file data/merged_input_normalized.fasta \
  --threshold 0.6 \
  --out-file results/graphpart/graphpart_assignments_60.csv \
  --labels-name label \
  --test-ratio 0.1 \
  --val-ratio 0.05 \
  --threads 12
```

### Flags

- `needle` — subcommand selecting global alignment (Needleman-Wunsch, via EMBOSS `needleall`) as the similarity method, appropriate for proteins.
- `--fasta-file` — input FASTA (merged positive + negative set, with ambiguous residues already normalised).
- `--threshold` — identity cutoff for the partitioning graph (0.4 or 0.6).
- `--labels-name label` — reads the `label=positive`/`label=negative` field from the FASTA headers.
- `--test-ratio 0.1` — 10% of sequences assigned to the test partition.
- `--val-ratio 0.05` — 5% of sequences assigned to the validation partition.
- `--threads 12` — 12 threads.

## Resulting split (40%)

| Partition | Positive | Negative | Total |
|---|---|---|---|
| Train | 506 | 1,009 | 1,515 |
| Validation | 29 | 59 | 88 |
| Test | 58 | 118 | 176 |

All 1,779 sequences were retained.

## Critical dependency note

GraphPart requires `EMBOSS_ACDROOT` and `EMBOSS_DATA` to point at the conda environment's EMBOSS install, or `needleall` fails **silently**, producing zero graph edges and an invalid random-looking split with no error raised. These are exported via the conda activation scripts for `thesis_env`. Always check the edge count (`Full graph nr. of edges:`) before trusting a GraphPart output; `run_pipeline.py` verifies this automatically.

The split was independently audited for train-test homology — see `../split_identity/`.