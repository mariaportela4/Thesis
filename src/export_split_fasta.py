"""
export_split_fasta.py

Writes per-split (train/test/valid) FASTA files for both the 40% and 60%
GraphPart datasets, keyed by orig_id — used as input to pairwise
Needleman-Wunsch alignment / audit checks, not by the main training
pipeline itself.

Extracted from 01_domain_analysis.ipynb, section 10 — logic unchanged,
except out_dir now points at results/graphpart/alignment/.

Usage (from project root):
    python src/export_split_fasta.py

Input
-----
data/final_dataset_40.csv   (must have a 'split' column)
data/final_dataset_60.csv   (exploratory only — skipped if missing)

Output
------
results/graphpart/alignment/40_percent/40pct_{train,test,valid}.fasta
results/graphpart/alignment/60_percent/60pct_{train,test,valid}.fasta
"""
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

DATASETS = {
    "40": {
        "csv": Path("data/final_dataset_40.csv"),
        "out_dir": Path("results/graphpart/alignment/40_percent"),
    },
    #"60": {
        #"csv": Path("data/final_dataset_60.csv"),
        #"out_dir": Path("results/graphpart/alignment/60_percent"),
    #},
}


def main():
    for pct, cfg in DATASETS.items():
        csv_path = cfg["csv"]
        if not csv_path.exists():
            print(f"[{pct}%] {csv_path} not found — skipping (exploratory run not present)")
            continue

        df = pd.read_csv(csv_path)
        out_dir = cfg["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGraphPart {pct}%")
        print(df["split"].value_counts())

        for split_name in ["train", "test", "valid"]:
            split_df = df[df["split"] == split_name].copy()

            records = []
            for _, row in split_df.iterrows():
                record_id = str(row["orig_id"])
                sequence = str(row["sequence"])
                records.append(SeqRecord(Seq(sequence), id=record_id, description=""))

            out_path = out_dir / f"{pct}pct_{split_name}.fasta"
            SeqIO.write(records, out_path, "fasta")

            print(f"{split_name}: wrote {len(records)} sequences to {out_path}")


if __name__ == "__main__":
    main()