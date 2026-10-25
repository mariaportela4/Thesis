# results/split_identity/

Audit of the GraphPart 40% split, produced by `../../src/datasets/check_split_identity.py`. Verifies that no test sequence is a near-duplicate of a training sequence.

## Files

- `pairwise_identity.csv` — identity for every train-test pair.
- `max_identity_per_test.csv` — the closest training sequence for each test sequence.
- `identity_distribution.png` — the distribution against the 40% threshold.

## Method

Every pair is globally aligned with EMBOSS `needleall`, the same aligner GraphPart uses (EBLOSUM62, gap open 10, gap extend 0.5). Identity is computed as identical residues over alignment length, GraphPart's default denominator. Requires `needleall` on PATH, with `EMBOSS_DATA` set by the `thesis_env` activation scripts.