"""
combine_domain_fasta.py
 
Combines every per-domain positive FASTA (data/positive/domain_*.fasta —
one file per NCBI CDD accession: cd20481, cl22684, pfam07212, pfam12217,
pfam12219, pfam12708, pfam13472) into a single candidate FASTA. 
 
Extracted from 01_domain_analysis.ipynb, section 1.1 — logic unchanged.
 
Usage (from project root):
    python src/combine_domain_fasta.py
 
Input
-----
data/positive/domain_*.fasta
 
Output
------
data/positive/all_DPO_candidates.fasta
"""
from pathlib import Path
 
INPUT_DIR = Path("data/positive")
OUTPUT_FILE = INPUT_DIR / "all_DPO_candidates.fasta"
 
 
def main():
    fasta_files = list(INPUT_DIR.glob("domain_*.fasta"))
 
    with open(OUTPUT_FILE, "w") as outfile:
        for file in fasta_files:
            with open(file, "r") as infile:
                outfile.write(infile.read())
 
    print(f"{len(fasta_files)} files combined into {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()
 












