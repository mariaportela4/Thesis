"""Check sequence identity between the train and test splits.

Every train-test pair is globally aligned with EMBOSS needleall, the aligner used by GraphPart
(EBLOSUM62, gap open 10, gap extend 0.5). Identity = identical residues / alignment length
(GraphPart's default 'full' denominator). For each test sequence, the maximum identity to any
train sequence is reported and plotted against the 40% partitioning threshold.

Requires needleall on PATH (EMBOSS_DATA set by the thesis_env activation scripts).

Outputs (results/split_identity/):
    pairwise_identity.csv       all train-test pairs
    max_identity_per_test.csv   closest train sequence for each test sequence
    identity_distribution.png   distributions with the 40% threshold
"""
import argparse
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_FASTA = "data/train_sequences_clean.fasta"
TEST_FASTA = "data/test_sequences_clean.fasta"
OUT_DIR = "results/split_identity"
THRESHOLD = 40.0


def read_fasta(path):
    seqs, name = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split("|")[0].split()[0]
                seqs[name] = []
            else:
                seqs[name].append(line.upper())
    return {k: "".join(v) for k, v in seqs.items()}


def write_fasta(seqs, path):
    with open(path, "w") as fh:
        for k, s in seqs.items():
            fh.write(f">{k}\n{s}\n")


def run_needle(chunk_path, train_path):
    cmd = ["needleall", "-asequence", chunk_path, "-bsequence", train_path,
           "-gapopen", "10", "-gapextend", "0.5", "-aformat3", "pair", "-auto", "-stdout"]
    res, a, b = [], None, None
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as p:
        for line in p.stdout:
            if line.startswith("# 1:"):
                a = line.split()[2]
            elif line.startswith("# 2:"):
                b = line.split()[2]
            elif line.startswith("# Identity:"):
                m = re.search(r"(\d+)/(\d+)", line)
                res.append((a, b, 100 * int(m[1]) / int(m[2])))
    if p.returncode != 0:
        raise RuntimeError(f"needleall failed on {chunk_path}")
    return res


def needle_identity(train, test, tmp, threads):
    train_path = os.path.join(tmp, "train.fasta")
    write_fasta(train, train_path)
    te_ids = list(test)
    chunks = [c for c in (te_ids[i::threads] for i in range(threads)) if c]
    paths = []
    for i, ids in enumerate(chunks):
        path = os.path.join(tmp, f"test_chunk{i}.fasta")
        write_fasta({k: test[k] for k in ids}, path)
        paths.append(path)
    with ThreadPoolExecutor(threads) as ex:
        results = ex.map(lambda p: run_needle(p, train_path), paths)
    rows = [r for res in results for r in res]
    return pd.DataFrame(rows, columns=["test_id", "train_id", "identity"])


def plot(df, best):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bins = np.arange(0, 101, 2)

    axes[0].hist(best["identity"], bins=bins, color="#1f77b4", alpha=0.8)
    axes[0].set_title("Max identity of each test sequence to the train set")
    axes[0].set_ylabel("Test sequences")

    axes[1].hist(df["identity"], bins=bins, color="#1f77b4", alpha=0.8)
    axes[1].set_title("All train-test pairs")
    axes[1].set_ylabel("Pairs (log scale)")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.axvline(THRESHOLD, color="red", ls="--", lw=1.5, label=f"{THRESHOLD:.0f}% threshold")
        ax.set_xlabel("Sequence identity (%)")
        ax.set_xlim(0, 100)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "identity_distribution.png"), dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    train, test = read_fasta(TRAIN_FASTA), read_fasta(TEST_FASTA)
    print(f"Train: {len(train)} | Test: {len(test)} | Pairs: {len(train) * len(test)}")

    with tempfile.TemporaryDirectory() as tmp:
        df = needle_identity(train, test, tmp, args.threads)
    df.to_csv(os.path.join(OUT_DIR, "pairwise_identity.csv"), index=False)

    best = (df.loc[df.groupby("test_id")["identity"].idxmax()]
              .rename(columns={"train_id": "closest_train_id"})
              .sort_values("identity", ascending=False))
    best.to_csv(os.path.join(OUT_DIR, "max_identity_per_test.csv"), index=False)

    n_over = (best["identity"] > THRESHOLD).sum()
    print(f"Max={best['identity'].max():.1f}% | median of per-test max={best['identity'].median():.1f}% "
          f"| test seqs > {THRESHOLD:.0f}%: {n_over}/{len(best)}")

    plot(df, best)
    print(f"Saved outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()