# alignment/

Output of the CD-HIT (80% identity) + GraphPart clustering/splitting stage: final train/valid/test FASTA files, at two GraphPart identity thresholds.

## Subfolders

- `40_percent/` — **used for the final thesis results.** GraphPart run at 40% identity threshold (`--test-ratio 0.2 --val-ratio 0.1`).
- `60_percent/` — GraphPart run at 60% identity threshold, done in parallel for comparison. **Not used in the final thesis results** — kept for reference/exploratory purposes only.

## How these were generated

Both CD-HIT and GraphPart were **run via the command line** on `tesla` (not wrapped in a repo script). GraphPart depends on EMBOSS `needleall`; a missing `EMBOSS_ACDROOT` environment variable caused it to fail silently (zero graph edges → invalid random split) until this was fixed by setting `EMBOSS_ACDROOT`/`EMBOSS_DATA` via the conda activation scripts. See `../logs/README.md` for the corresponding run logs and `../results/graphpart/README.md` for the split assignment outputs.

Pipeline order leading here: remove hypothetical proteins → remove duplicates → CD-HIT 80% → class balancing (1:2, sampled from CD-HIT negative output) → merge → GraphPart (40% or 60%). See root `README.md` for the full, confirmed sequence.