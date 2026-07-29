# data/negative/

Negative class: non-DPO protein sequences, from raw collection through to the balanced 1:2 (positive:negative) sample used in the pipeline.

## Files, in pipeline order

1. `negative.faa` — raw negative sequence collection, fetched by `../../src/datasets/ncbi_fetch.py` (capped at 30,000 sequences).
2. `negative_ids.txt` — corresponding NCBI protein UIDs, one per line (written by `ncbi_fetch.py` alongside `negative.faa`).
3. `negative_nohyp.faa` — after removal of hypothetical proteins (pipeline step 1): 24,929 kept, 5,071 removed.
4. `negative_dedup.faa` — after duplicate removal (pipeline step 2, `../remove_duplicates.py`): 23,102 unique sequences (1,827 duplicates removed from the 24,929 above).
5. CD-HIT clustering (80% identity) is applied next, producing `../../results/cdhit/negative_cdhit80.fasta` (13,377 sequences) — see `../../results/cdhit/README.md`.
6. `negative_sampled.fasta` — final balanced negative set, sampled **from the CD-HIT-clustered output** (not from `negative_dedup.faa`) at a 1:2 ratio against the CD-HIT-clustered positive count (593 positives → 1,186 negatives sampled), using `random.seed(42)` for reproducibility. This is merged with the positive CD-HIT output to build `../merged_input.fasta`, ahead of GraphPart.

## Notes

- Class balancing happens **after** CD-HIT, not before — the 1:2 ratio is applied to the already-clustered (non-redundant) sequence sets.