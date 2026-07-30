"""
normalize_ambiguous_aa.py

Replaces ambiguous amino acid residues (B, Z, J, U, O) with X across all
sequences in the input FASTA. No sequences are dropped. This produces a
single canonical FASTA to be used as the input for every downstream branch
of the pipeline (CD-HIT, GraphPart, embedding extraction, DePP/DepoScope
retraining), ensuring consistent sequence content across all compared models.

Note: DePP's amino acid composition features (via Biopython's
ProteinAnalysis.get_amino_acids_percent) silently ignore X in the
percentage calculation rather than raising an error. 

Usage:
    python normalize_ambiguous_aa.py \
        --input data/merged_input.fasta \
        --output data/merged_input_normalized.fasta
"""

import argparse
from Bio import SeqIO
from Bio.Seq import Seq

AMBIGUOUS_TO_X = set('BZJUO')  # X itself is already the "unknown" code, left as-is


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    records = list(SeqIO.parse(args.input, "fasta"))
    print(f"[+] {len(records)} sequences read from {args.input}")

    n_changed = 0
    n_residues_changed = 0
    for rec in records:
        seq_str = str(rec.seq)
        new_seq_chars = []
        changed = False
        for c in seq_str:
            if c.upper() in AMBIGUOUS_TO_X:
                new_seq_chars.append('X')
                n_residues_changed += 1
                changed = True
            else:
                new_seq_chars.append(c)
        if changed:
            n_changed += 1
            rec.seq = Seq(''.join(new_seq_chars))

    SeqIO.write(records, args.output, "fasta")
    print(f"[+] Saved: {args.output}")
    print(f"    sequences modified (contained B/Z/J/U/O):  {n_changed}")
    print(f"    total residues replaced with X:            {n_residues_changed}")


if __name__ == '__main__':
    main()