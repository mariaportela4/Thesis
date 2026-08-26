"""
remove_hypothetical_proteins.py

Removes "hypothetical protein" entries (no experimental/functional evidence)
from the raw positive and negative candidate sets, keeping only sequences
whose description doesn't contain "hypothetical" (case-insensitive).

Extracted from 01_domain_analysis.ipynb, section 1.1.1 — logic unchanged.

Usage (from project root):
    python src/remove_hypothetical_proteins.py

Input
-----
data/positive/all_DPO_candidates.fasta   (produced by combining domain_*.fasta)
data/negative/negative.faa

Output
------
data/positive/all_DPO_positive.fasta
data/negative/negative_nohyp.faa
"""
from pathlib import Path
from Bio import SeqIO

FILES = {
    "positive": {
        "input": Path("data/positive/all_DPO_candidates.fasta"),
        "output": Path("data/positive/all_DPO_positive.fasta"),
    },
    "negative": {
        "input": Path("data/negative/negative.faa"),
        "output": Path("data/negative/negative_nohyp.faa"),
    },
}


def main():
    for label, paths in FILES.items():
        kept = []
        removed = 0

        for record in SeqIO.parse(paths["input"], "fasta"):
            header = record.description.lower()
            if "hypothetical" not in header:
                kept.append(record)
            else:
                removed += 1

        SeqIO.write(kept, paths["output"], "fasta")

        print(f"[{label}] Kept: {len(kept)}")
        print(f"[{label}] Removed: {removed}")


if __name__ == "__main__":
    main()