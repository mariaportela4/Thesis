# data/positive/

Positive class: bacteriophage depolymerase (DPO) sequences, retrieved by conserved domain.

## Files, in pipeline order

- `domain_cd20481.fasta`, `domain_cl22684.fasta`, `domain_pfam07212.fasta`, `domain_pfam12217.fasta`, `domain_pfam12219.fasta`, `domain_pfam12708.fasta`, `domain_pfam13472.fasta` — sequences retrieved per CDD/Pfam accession associated with depolymerase activity (see `../protein_domains.csv`), exported manually from NCBI and filtered to viral origin. Not script-generated.
- `all_DPO_candidates.fasta` — concatenation of the seven files above (`../../src/datasets/combine_domains.py`): 2,041 entries corresponding to 1,949 unique records, since some proteins carry more than one domain.
- `all_DPO_positive.fasta` — after removal of hypothetical proteins: 1,764 kept, 277 removed.
- `all_DPO_positive_dedup.fasta` — after exact-duplicate removal (`../../src/remove_duplicates.py`): 1,117 sequences, 647 removed. Of those 647, 80 are records retrieved under more than one domain; the rest are distinct database entries with identical sequences.

CD-HIT clustering at 80% identity is applied next, producing `../../results/cdhit/positive_cdhit80.fasta` (593 sequences). That output, not `all_DPO_positive_dedup.fasta`, is what gets merged with the balanced negative set.

## Deduplication criterion

Duplicate groups are resolved by accession priority (RefSeq > GenBank > PDB), then by the more complete header description within the same category. PDB chains are handled separately: only one representative chain per structure is kept, since their redundancy comes from replication within the crystallographic asymmetric unit.