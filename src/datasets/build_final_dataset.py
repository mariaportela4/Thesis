"""
build_final_dataset.py

Merges GraphPart's cluster assignments back into the sequence metadata to
produce the canonical dataset CSV (data/final_dataset_40.csv), which every
downstream stage uses as its single source of truth for the train/valid/test
split.

Extracted from notebooks/01_domain_analysis.ipynb (section 6.2) so the split
can be regenerated from the command line via run_pipeline.py. Logic is
unchanged; the audit checks that were separate notebook cells are now run
automatically and fail loudly instead of only printing.

GraphPart writes numeric cluster IDs, not split names. With
--test-ratio 0.1 --val-ratio 0.05 the mapping is:

    cluster 0 -> train    (largest partition)
    cluster 1 -> test
    cluster 2 -> valid    (smallest partition)

This mapping is verified against the actual partition sizes before writing:
if train is not the largest partition or test is not larger than valid, the
script aborts rather than silently producing a mislabelled split.

Usage (from project root):
    python src/build_final_dataset.py \
        --metadata data/merged_metadata.csv \
        --assignments results/graphpart/graphpart_assignments_40.csv \
        --output data/final_dataset_40.csv

Input
-----
data/merged_metadata.csv                          -- one row per sequence, keyed by seq_id
results/graphpart/graphpart_assignments_40.csv    -- GraphPart output, keyed by AC

Output
------
data/final_dataset_40.csv   -- metadata + 'cluster' + 'split' columns,
                               deduplicated on orig_id
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CLUSTER_TO_SPLIT = {0.0: "train", 1.0: "test", 2.0: "valid"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/merged_metadata.csv")
    parser.add_argument(
        "--assignments", default="results/graphpart/graphpart_assignments_40.csv"
    )
    parser.add_argument("--output", default="data/final_dataset_40.csv")
    parser.add_argument(
        "--allow-unassigned", action="store_true",
        help="Do not abort if GraphPart failed to assign some sequences to a cluster.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = Path(args.metadata)
    assignments_path = Path(args.assignments)
    output_path = Path(args.output)

    for path in (metadata_path, assignments_path):
        if not path.exists():
            sys.exit(f"Input file not found: {path}")

    meta_df = pd.read_csv(metadata_path)
    assign_df = pd.read_csv(assignments_path)
    print(f"Metadata:    {len(meta_df)} sequences from {metadata_path}")
    print(f"Assignments: {len(assign_df)} sequences from {assignments_path}")

    # Left join so sequences GraphPart could not assign appear with NaN cluster
    # instead of vanishing from the dataset unnoticed.
    df = meta_df.merge(assign_df, left_on="seq_id", right_on="AC", how="left")

    n_unassigned = df["cluster"].isna().sum()
    if n_unassigned:
        message = (
            f"{n_unassigned} sequence(s) were not assigned to any GraphPart cluster. "
            "GraphPart silently excludes sequences it cannot place, so this is real "
            "data loss and should be understood before training."
        )
        if not args.allow_unassigned:
            sys.exit(f"ABORTED: {message}\nRe-run with --allow-unassigned to override.")
        print(f"WARNING: {message}")

    unexpected = set(df["cluster"].dropna().unique()) - set(CLUSTER_TO_SPLIT)
    if unexpected:
        sys.exit(
            f"ABORTED: GraphPart produced unexpected cluster ID(s): {sorted(unexpected)}.\n"
            f"Only {sorted(CLUSTER_TO_SPLIT)} are mapped to split names. Check the "
            "--test-ratio / --val-ratio used for the GraphPart run."
        )

    df["split"] = df["cluster"].map(CLUSTER_TO_SPLIT)

    # Deduplicate on orig_id, matching the notebook's final cleanup step.
    before = len(df)
    df = df.drop_duplicates(subset="orig_id", keep="first")
    if before != len(df):
        print(f"Dropped {before - len(df)} duplicate orig_id row(s)")

    counts = df["split"].value_counts()
    n_train = int(counts.get("train", 0))
    n_test = int(counts.get("test", 0))
    n_valid = int(counts.get("valid", 0))

    # Verify the cluster -> split mapping against the actual partition sizes.
    # If GraphPart ever changes the order in which it numbers its partitions,
    # this catches it instead of producing a silently mislabelled split.
    if not (n_train > n_test > n_valid):
        sys.exit(
            "ABORTED: partition sizes are inconsistent with the assumed cluster "
            f"mapping (train={n_train}, test={n_test}, valid={n_valid}).\n"
            "Expected train > test > valid for --test-ratio 0.1 --val-ratio 0.05. "
            "Inspect the GraphPart output and update CLUSTER_TO_SPLIT before "
            "trusting this split."
        )

    print("\nSplit distribution:")
    for split in ("train", "valid", "test"):
        subset = df[df["split"] == split]
        label_counts = subset["label"].value_counts().to_dict()
        print(f"  {split:<6} {len(subset):>5}   {label_counts}")

        if subset["label"].nunique() < 2:
            sys.exit(
                f"ABORTED: the '{split}' split contains only one class. "
                "A single-class partition makes evaluation meaningless."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved final dataset to: {output_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()