# data/negative/

Negative class: phage proteins that are explicitly not depolymerases, from retrieval through to the balanced 1:2 sample.

## Files, in pipeline order

1. `negative.faa` — candidate sequences fetched by `../../src/datasets/ncbi_fetch.py` from NCBI RefSeq, restricted to Caudoviricetes. Depolymerase-related title keywords and the seven positive-class domain accessions are excluded at query time.
2. `negative_ids.txt` — the corresponding NCBI protein UIDs.
3. `negative_nohyp.faa` — after removal of hypothetical proteins.
4. `negative_dedup.faa` — after exact-duplicate removal (first occurrence kept, without the accession-priority rule used for the positive set): 1,854 sequences.
5. CD-HIT at 80% identity is applied next, producing `../../results/cdhit/negative_cdhit80.fasta` (1,237 sequences).
6. `negative_sampled.fasta` — final balanced set, sampled **from the CD-HIT output** at a 1:2 ratio against the clustered positive count (593 positives → 1,186 negatives), using `random.seed(42)`.

## Length-matched sampling

Viral proteins in RefSeq skew strongly towards short sequences, whereas depolymerases are typically large. Sampling uniformly would confound length with class, letting a classifier separate the two by length rather than function. `ncbi_fetch.py` therefore retrieves lengths first for the full candidate pool, divides the positive length distribution into 25-aa strata, and samples negatives per stratum in proportion to the positives it contains.

This reduced the Kolmogorov-Smirnov statistic between the two distributions from 0.292 to 0.195 at the candidate stage. Some residual mismatch remains in the final dataset (D = 0.198).

## Notes

- Class balancing happens **after** CD-HIT, so the 1:2 ratio is applied to the already-clustered sets.