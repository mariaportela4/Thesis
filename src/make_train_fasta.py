"""
Generates a clean FASTA for a given dataset split (train/test/valid) from
merged_input_normalized.fasta, filtering by the 'split' column in
final_dataset_40.csv.

All ambiguous residues are normalized to X upstream, once, by
normalize_ambiguous_aa.py. This script assumes its --merged-fasta input is
already normalized, and no longer drops any sequences.

Usage:
    python src/make_train_fasta.py \
        --merged-fasta data/merged_input_normalized.fasta \
        --dataset-csv data/final_dataset_40.csv \
        --output data/train_sequences_clean.fasta \
        --split test
"""

import argparse

import pandas as pd
from Bio import SeqIO


def main():
    parser = argparse.ArgumentParser(description="Generate a FASTA for a given dataset split")
    parser.add_argument('--merged-fasta', required=True)
    parser.add_argument('--dataset-csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='train', help="value of the 'split' column to filter on (default: train)")
    #parser.add_argument('--split', default='test', help="value of the 'split' column to filter on (default: test)")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    split_ids = set(df[df['split'] == args.split]['seq_id'])
    print(f"[+] {len(split_ids)} seq_id in split '{args.split}' from {args.dataset_csv}")

    records = list(SeqIO.parse(args.merged_fasta, "fasta"))
    print(f"[+] {len(records)} sequences read from {args.merged_fasta}")

    # The FASTA id (record.id) is the field before the first '|', e.g. 'seq000001'
    def seq_id_of(record):
        return record.id.split('|')[0]

    kept = []
    dropped_not_in_split = 0

    for rec in records:
        sid = seq_id_of(rec)
        if sid not in split_ids:
            dropped_not_in_split += 1
            continue
        kept.append(rec)

    SeqIO.write(kept, args.output, "fasta")

    print(f"[+] Saved: {args.output}")
    print(f"    kept:                     {len(kept)}")
    print(f"    dropped (outside split):  {dropped_not_in_split}")
    print(f"    expected in split:        {len(split_ids)}")

    if len(kept) != len(split_ids):
        print(f"[!] WARNING: kept count ({len(kept)}) does not match expected split size ({len(split_ids)}) — check for missing seq_ids")


if __name__ == '__main__':
    main()