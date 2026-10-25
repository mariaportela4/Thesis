"""
dposfinder_retrain.py

Retrains DposFinder on our own GraphPart 40%-identity
split, so it can be benchmarked on the same 176-sequence test set as the other
three retrained external tools (DepoScope, DePP, PhageDPO) and the six
thesis models.

Source: https://github.com/shenyanxiang/DposFinder
Paper:  Shen, Y., Lun, H., Zhang, Y. et al. 
        DposFinder: an interpretable transformer model for predicting phage-derived polysaccharide depolymerases and their host capsular serotypes. 
        Genome Med 18, 75 (2026). https://doi.org/10.1186/s13073-026-01657-3

DposFinder is trained/evaluated via its own code/main.py, not imported as a
library, so this script's job is purely:
  1. Read final_dataset_40.csv (already has seq_id, label, sequence, split
     columns from the GraphPart 40% run) and convert it into DposFinder's
     expected format: plain FASTA, header ending in the literal "pos" or
     "neg".
    2. Write the GraphPart train split as DposFinder's train_set.fasta. The
     validation split is not used: DposFinder's `final_train` mode has no
     validation phase -- it trains on the whole train_set.fasta for a fixed
     number of epochs and evaluates each epoch on test_set.fasta, which also
     drives its learning-rate scheduler.
  3. Invoke DposFinder's code/main.py --mode final_train as a subprocess,
     exactly the way DepoScope/DePP/PhageDPO retraining was orchestrated.


Usage:
    python dposfinder_retrain.py \
        --dataset_csv results/graphpart/final_dataset_40.csv \
        --dposfinder_repo /path/to/DposFinder \
        --out_dir results/dposfinder_retrain \
        [--num_epochs 15] [--batch_size 16] [--lr 5e-5] [--no_cuda] \
        [--python /path/to/dposfinder-env/bin/python]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_dposfinder_fasta(df: pd.DataFrame, out_path: Path, max_seq_len: int = 1024):
    """
    Writes rows in DposFinder's expected format: header ends in the literal
    'pos' or 'neg' (last 3 characters, per packages/utils.py's label2index,
    which reads label[-3:]).

    DposFinder's DPOSFINDER model (packages/Models.py) does NOT truncate
    sequences before running them through ESM-2, unlike our own embedding
    pipeline which caps at 1024 tokens. Since transformer attention memory
    scales quadratically with sequence length, a handful of unusually long
    proteins can OOM the GPU regardless of batch size. We truncate to
    max_seq_len here for consistency with the 1024-token cap already used
    elsewhere in the thesis pipeline, and to keep training numerically
    stable -- this should be noted as a methods deviation from DposFinder's
    original (untruncated) training setup.
    """
    n_truncated = 0
    with open(out_path, "w") as fh:
        for _, row in df.iterrows():
            label = "pos" if row["label"] == "positive" else "neg"
            seq = row["sequence"]
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
                n_truncated += 1
            # keep seq_id + orig_id in the header for traceability, then the
            # mandatory trailing pos/neg token DposFinder actually parses
            fh.write(f">{row['seq_id']}|{row['orig_id']} {label}\n")
            fh.write(f"{seq}\n")

    if n_truncated:
        print(f"  WARNING: {n_truncated} sequence(s) in {out_path.name} were "
              f"truncated to {max_seq_len} residues (see write_dposfinder_fasta "
              "docstring)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_csv", required=True, type=Path,
                     help="final_dataset_40.csv (seq_id,label,orig_id,...,sequence,split)")
    ap.add_argument("--dposfinder_repo", required=True, type=Path,
                     help="Path to cloned shenyanxiang/DposFinder repo")
    ap.add_argument("--out_dir", required=True, type=Path,
                     help="Where to write DposFinder-format FASTA + results")
    ap.add_argument("--num_epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--name", default="dposfinder_retrained",
                     help="checkpoint/run name passed to DposFinder's main.py")
    ap.add_argument("--python", default=sys.executable,
                     help="interpreter used to run DposFinder's main.py. "
                         "Defaults to the one running this script; set it to "
                         "another environment's python if DposFinder's "
                         "dependencies are installed elsewhere. Must match the "
                         "interpreter later used for --mode predict.")
    ap.add_argument("--max_seq_len", type=int, default=1024,
                     help="truncate sequences longer than this many residues "
                          "before writing FASTA (DposFinder itself doesn't "
                          "truncate, which can OOM on long sequences; default "
                          "matches the 1024-token cap used for all other "
                          "embeddings/models in the thesis pipeline)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.out_dir.resolve()  # must be absolute: main.py runs with
    # cwd=dposfinder_repo, so a relative --data_path would be resolved
    # relative to the wrong directory

    print("[1/3] Loading final_dataset_40.csv...")
    df = pd.read_csv(args.dataset_csv)

    required_cols = {"seq_id", "label", "orig_id", "sequence", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"dataset_csv is missing expected columns: {missing}")

    counts = df["split"].value_counts()
    print("  split sizes found:")
    print(counts.to_string())

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    print(f"  train (DposFinder train_set): {len(train_df)}")
    print(f"  test (DposFinder test_set):        {len(test_df)}")
    if len(test_df) != 176:
        print(f"WARNING: expected 176 test sequences, got {len(test_df)} "
              "-- double check the split column values in dataset_csv",
              file=sys.stderr)
    print("[2/3] Writing DposFinder-format FASTA files...")
    write_dposfinder_fasta(train_df, data_path / "train_set.fasta", args.max_seq_len)
    write_dposfinder_fasta(test_df, data_path / "test_set.fasta", args.max_seq_len)

    print("[3/3] Launching DposFinder final_train...")
    cmd = [
        args.python, str(args.dposfinder_repo / "code" / "main.py"),
        "--mode", "final_train",
        "--data_path", str(data_path) + "/",
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--name", args.name,
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")

    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(args.dposfinder_repo), check=True)


if __name__ == "__main__":
    main()