"""
balance_and_merge.py

1) Subsamples the CD-HIT-clustered negative set 1:2 (positive:negative),
   using random.seed(42) for reproducibility.
2) Merges the CD-HIT-clustered positives + sampled negatives into a single
   FASTA (sequential IDs, label/orig_id encoded in the header) plus a
   companion metadata CSV (product / organism / full_description / sequence
   per row) used downstream by GraphPart and embedding generation.

Extracted from 01_domain_analysis.ipynb, sections 4 and 6 — logic unchanged,
except the negative sample size is now computed from the actual positive
count (n_positives * RATIO) instead of the hardcoded 1185. The section-5 "preliminary merge" cell
(FASTA only, no metadata) is superseded by section 6's full merge and isn't
reproduced here.

Usage (from project root):
    python src/balance_and_merge.py

Input
-----
results/cdhit/positive_cdhit80.fasta
results/cdhit/negative_cdhit80.fasta

Output
------
data/negative/negative_sampled.fasta
data/merged_input.fasta
data/merged_metadata.csv
"""
import random
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO

POS_CDHIT = Path("results/cdhit/positive_cdhit80.fasta")
NEG_CDHIT = Path("results/cdhit/negative_cdhit80.fasta")
NEG_SAMPLED = Path("data/negative/negative_sampled.fasta")
MERGED_FASTA = Path("data/merged_input.fasta")
MERGED_METADATA = Path("data/merged_metadata.csv")

RATIO = 2  # negatives sampled per positive
SEED = 42


def parse_header_info(rec):
    """Split a Biopython record's description into (orig_id, product, organism, full_description)."""
    desc = rec.description
    orig_id = rec.id

    # Strip the repeated accession prefix that Biopython includes in .description
    rest = desc[len(orig_id):].strip() if desc.startswith(orig_id) else desc

    # Extract organism name enclosed in square brackets ([...])
    organism_match = re.search(r"\[(.*?)\]", rest)
    organism = organism_match.group(1) if organism_match else None

    # Remove the organism suffix to isolate the product/function description
    product = re.sub(r"\s*\[.*?\]\s*$", "", rest).strip() if rest else None

    return orig_id, product, organism, desc


def subsample_negatives():
    random.seed(SEED)

    records = list(SeqIO.parse(NEG_CDHIT, "fasta"))
    print(f"Total CD-HIT negatives: {len(records)}")

    n_positives = sum(1 for _ in SeqIO.parse(POS_CDHIT, "fasta"))
    sample_size = n_positives * RATIO

    sampled_records = random.sample(records, sample_size)
    SeqIO.write(sampled_records, NEG_SAMPLED, "fasta")

    print(f"Saved {len(sampled_records)} sampled negatives "
          f"({n_positives} positives x {RATIO}) to {NEG_SAMPLED}")


def merge():
    seq_counter = 1
    rows = []

    with open(MERGED_FASTA, "w") as out:
        # --- Positives ---
        for rec in SeqIO.parse(POS_CDHIT, "fasta"):
            new_id = f"seq{seq_counter:06d}"
            orig_id, product, organism, full_description = parse_header_info(rec)
            seq = str(rec.seq)

            out.write(f">{new_id}|label=positive|orig_id={orig_id}\n{seq}\n")
            rows.append({
                "seq_id": new_id, "label": "positive", "orig_id": orig_id,
                "product": product, "organism": organism,
                "full_description": full_description, "sequence": seq,
            })
            seq_counter += 1

        # --- Negatives ---
        for rec in SeqIO.parse(NEG_SAMPLED, "fasta"):
            new_id = f"seq{seq_counter:06d}"
            orig_id, product, organism, full_description = parse_header_info(rec)
            seq = str(rec.seq)

            out.write(f">{new_id}|label=negative|orig_id={orig_id}\n{seq}\n")
            rows.append({
                "seq_id": new_id, "label": "negative", "orig_id": orig_id,
                "product": product, "organism": organism,
                "full_description": full_description, "sequence": seq,
            })
            seq_counter += 1

    print(f"Merged FASTA saved to: {MERGED_FASTA}")
    print(f"Total sequences: {seq_counter - 1}")

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(MERGED_METADATA, index=False)
    print(f"Metadata saved to: {MERGED_METADATA}")

    # Sanity check — mirrors the notebook's separate verification cell
    pos_count = int((meta_df["label"] == "positive").sum())
    neg_count = int((meta_df["label"] == "negative").sum())
    print(f"Positives: {pos_count}")
    print(f"Negatives: {neg_count}")


def main():
    subsample_negatives()
    merge()


if __name__ == "__main__":
    main()