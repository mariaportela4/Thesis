"""
Generates train_sequences_clean.fasta from merged_input.fasta, filtering
for sequences in split == 'train' in final_dataset_40.csv and removing
sequences containing ambiguous amino acids (X, B, Z, J, U, O) — same logic
used to generate test_sequences_clean.fasta.
 
Usage:
    python make_train_fasta.py \
        --merged-fasta data/merged_input.fasta \
        --dataset-csv data/final_dataset_40.csv \
        --output train_sequences_clean.fasta \
        --split train
"""
 
import argparse
from pathlib import Path
 
import pandas as pd
from Bio import SeqIO
 
AMBIGUOUS_AA = set('XBZJUO')
 
 
def main():
    parser = argparse.ArgumentParser(description="Generate a clean FASTA for a given dataset split")
    parser.add_argument('--merged-fasta', required=True)
    parser.add_argument('--dataset-csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='train', help="value of the 'split' column to filter on (default: train)")
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
    dropped_ambiguous = 0
 
    for rec in records:
        sid = seq_id_of(rec)
        if sid not in split_ids:
            dropped_not_in_split += 1
            continue
        seq = str(rec.seq).rstrip('*')
        if AMBIGUOUS_AA & set(seq.upper()):
            dropped_ambiguous += 1
            continue
        kept.append(rec)
 
    SeqIO.write(kept, args.output, "fasta")
 
    print(f"[+] Saved: {args.output}")
    print(f"    kept:                     {len(kept)}")
    print(f"    dropped (outside split):  {dropped_not_in_split}")
    print(f"    dropped (ambiguous aa):   {dropped_ambiguous}")
    print(f"    expected in split:        {len(split_ids)}")
 
 
if __name__ == '__main__':
    main()