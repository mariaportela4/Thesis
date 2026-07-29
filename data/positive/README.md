# data/positive/

Positive class: bacteriophage depolymerase (DPO) sequences, collected by conserved domain.

## Files, in pipeline order

- `domain_cd20481.fasta`, `domain_cl22684.fasta`, `domain_pfam07212.fasta`, `domain_pfam12217.fasta`, `domain_pfam12219.fasta`, `domain_pfam12708.fasta`, `domain_pfam13472.fasta` — sequences retrieved per CDD/Pfam domain associated with depolymerase activity (domain IDs correspond to entries in `../protein_domains.csv`), manually exported per domain (not script-generated).
- `all_DPO_candidates.fasta` — union of all domain-based candidate sequences (2,041 total), produced by concatenating the `domain_*.fasta` files in `../../notebooks/01_domain_analysis.ipynb` (see `../../notebooks/README.md`).
- `all_DPO_positive.fasta` — after removal of hypothetical proteins (pipeline step 1): 1,764 kept, 277 removed.
- `all_DPO_positive_dedup.fasta` — after duplicate removal (pipeline step 2, `../remove_duplicates.py`): 1,117 unique sequences (647 duplicates removed from the 1,764 above).

CD-HIT clustering (80% identity) is applied next, producing `../../results/cdhit/positive_cdhit80.fasta` (593 sequences) — see `../../results/cdhit/README.md`. That CD-HIT output (not `all_DPO_positive_dedup.fasta`) is what gets merged with the balanced negative set into `../merged_input.fasta`.
