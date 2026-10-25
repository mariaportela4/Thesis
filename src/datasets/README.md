# src/datasets/

Dataset construction scripts, in pipeline order, plus the split audit.

## Files

- `combine_domains.py` — concatenates the per-domain positive FASTA files (`data/positive/domain_*.fasta`, one per CDD/Pfam accession) into `all_DPO_candidates.fasta`. The per-domain files were exported manually from NCBI and are not script-generated. The combined file holds 2,041 entries corresponding to 1,949 unique records, since some proteins carry more than one domain.
- `ncbi_fetch.py` — fetches the negative set from NCBI RefSeq (Caudoviricetes), excluding depolymerase-related keywords and the seven positive-class domain accessions at query time. Rather than taking every hit, it retrieves lengths first and samples per 25-aa stratum so that the negative length distribution matches the positive one, capped at 30,000 candidates. Also reports the Kolmogorov-Smirnov statistic between the two distributions.
- `remove_hypothetical_proteins.py` — drops entries whose description contains "hypothetical".
- `balance_and_merge.py` — subsamples the CD-HIT-clustered negatives to a 1:2 ratio (`random.seed(42)`) and merges both classes into `data/merged_input.fasta` plus a metadata CSV.
- `build_final_dataset.py` — merges GraphPart's cluster assignments into the sequence metadata to produce `data/final_dataset_40.csv`, the canonical split reference. GraphPart writes numeric cluster IDs, which are mapped to train/test/valid by partition size.
- `check_split_identity.py` — audit: globally aligns every train-test pair with EMBOSS `needleall` and reports the maximum identity of each test sequence to any training sequence, against the 40% threshold. Outputs to `../../results/split_identity/`.

## Notes

- Deduplication happens between steps 3 and 4 and lives in `../remove_duplicates.py`, not in this folder.
- `combine_domains.py` does not deduplicate: the 92 records retrieved under more than one domain are removed later, at the deduplication step.