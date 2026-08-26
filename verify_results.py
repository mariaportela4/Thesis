"""
verify_results.py

It checks, in order:
  1. Exact-duplicate sequences with contradictory labels (data quality)
  2. Current split composition in data/final_dataset_40.csv vs the
     previously-recorded reference (train 1513 / val 88 / test 178)
  3. Per-embedding split .npy shapes vs the CSV split column
     (this is where the 178 -> 176 discrepancy should surface)
  4. Sequence-level leakage: identical sequences appearing in both
     train and test (the classic cause of spuriously perfect metrics)
  5. Re-derives the confusion matrix for each of the 6 models directly
     from the saved checkpoint + saved test embeddings + y_test.npy,
     and diffs it against the numbers read off your screenshot
  6. Whether Optuna's tuned hyperparameters are actually reachable by
     train_cnn.py / train_dnn.py (dropout currently is not wired up
     as a CLI arg in either script, so tuned dropout may be silently
     ignored even if lr/batch_size/weight_decay were passed)

Nothing here modifies any files. Read-only checks + printed report.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(".")
DATA_CSV = ROOT / "data" / "final_dataset_40.csv"
SPLIT_DIR = ROOT / "results" / "embeddings" / "split"
MODELS_DIR = ROOT / "results" / "models"
TUNING_DIR = ROOT / "results" / "tuning"

EMBEDDINGS = ["esm", "bert", "prott5"]
ARCHS = ["dnn", "cnn"]

# What you read off the screenshot (Non-DPO=0 / DPO=1), as (TN, FP, FN, TP)
REPORTED = {
    "cnn_esm":    (114, 4, 1, 57),
    "cnn_bert":   (110, 8, 6, 52),
    "cnn_prott5": (116, 2, 5, 53),
    "dnn_esm":    (118, 0, 1, 57),
    "dnn_bert":   (117, 1, 2, 56),
    "dnn_prott5": (118, 0, 1, 57),
}

# Reference split sizes from the previously-verified 178-sequence benchmark
EXPECTED_SPLIT = {"train": 1513, "valid": 88, "test": 178}
EXPECTED_TEST_LABELS = {"negative": 119, "positive": 59}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    all_ok = True

    # ------------------------------------------------------------------
    section("1. Exact-duplicate sequences with contradictory labels")
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_CSV)
    dup_mask = df["sequence"].duplicated(keep=False)
    dup = df[dup_mask].sort_values("sequence")
    if dup.empty:
        check("No duplicate sequences in dataset", True)
    else:
        n_pairs = dup["sequence"].nunique()
        conflicting = 0
        for seq, grp in dup.groupby("sequence"):
            if grp["label"].nunique() > 1:
                conflicting += 1
        ok = check(
            "No label-conflicting duplicate sequences",
            conflicting == 0,
            f"{n_pairs} duplicate sequence group(s) found, {conflicting} with conflicting labels",
        )
        all_ok &= ok
        if conflicting:
            print("  Conflicting groups (orig_id, label, split):")
            for seq, grp in dup.groupby("sequence"):
                if grp["label"].nunique() > 1:
                    for _, row in grp.iterrows():
                        print(f"    {row['orig_id']:<20} {row['label']:<10} split={row['split']}")

    # ------------------------------------------------------------------
    section("2. Split composition (data/final_dataset_40.csv)")
    # ------------------------------------------------------------------
    split_counts = df["split"].value_counts().to_dict()
    print("Current split counts:", split_counts)
    print("Expected (old, verified 178-test benchmark):", EXPECTED_SPLIT)
    for split_name, expected_n in EXPECTED_SPLIT.items():
        actual_n = split_counts.get(split_name, 0)
        ok = check(f"split == '{split_name}' count", actual_n == expected_n,
                   f"expected {expected_n}, got {actual_n}")
        all_ok &= ok

    test_label_counts = df[df["split"] == "test"]["label"].value_counts().to_dict()
    print("Current test-split label counts:", test_label_counts)
    for label_name, expected_n in EXPECTED_TEST_LABELS.items():
        actual_n = test_label_counts.get(label_name, 0)
        ok = check(f"test label == '{label_name}' count", actual_n == expected_n,
                   f"expected {expected_n}, got {actual_n}")
        all_ok &= ok

    if sum(split_counts.get(s, 0) for s in EXPECTED_SPLIT) != 178 + 1513 + 88:
        print("  NOTE: total row count differs from 1779 — dataset itself may have been regenerated.")

    # ------------------------------------------------------------------
    section("3. Embedding split .npy shapes vs CSV split column")
    # ------------------------------------------------------------------
    csv_test_n = int((df["split"] == "test").sum())
    csv_train_n = int((df["split"] == "train").sum())
    csv_val_n = int((df["split"] == "valid").sum())

    y_test_path = SPLIT_DIR / "y_test.npy"
    if y_test_path.exists():
        y_test = np.load(y_test_path)
        ok = check("y_test.npy length matches CSV test rows", len(y_test) == csv_test_n,
                   f"y_test.npy has {len(y_test)}, CSV test split has {csv_test_n}")
        all_ok &= ok
        print(f"  y_test.npy positive count: {int(y_test.sum())} / {len(y_test)}")
    else:
        check("y_test.npy exists", False, str(y_test_path))
        all_ok = False

    for emb in EMBEDDINGS:
        for split_name, expected_n in [("train", csv_train_n), ("val", csv_val_n), ("test", csv_test_n)]:
            p = SPLIT_DIR / f"{emb}_{split_name}.npy"
            if not p.exists():
                check(f"{p.name} exists", False)
                all_ok = False
                continue
            arr = np.load(p)
            ok = check(f"{p.name} row count matches CSV '{('valid' if split_name=='val' else split_name)}'",
                       arr.shape[0] == expected_n,
                       f"npy has {arr.shape[0]}, CSV has {expected_n}")
            all_ok &= ok

    # ------------------------------------------------------------------
    section("4. Sequence-level leakage between train and test")
    # ------------------------------------------------------------------
    train_seqs = set(df[df["split"] == "train"]["sequence"])
    test_seqs = set(df[df["split"] == "test"]["sequence"])
    val_seqs = set(df[df["split"] == "valid"]["sequence"])

    leak_train_test = train_seqs & test_seqs
    leak_train_val = train_seqs & val_seqs
    leak_val_test = val_seqs & test_seqs

    ok = check("No identical sequences shared between train and test",
               len(leak_train_test) == 0, f"{len(leak_train_test)} shared sequence(s)")
    all_ok &= ok
    ok = check("No identical sequences shared between train and val",
               len(leak_train_val) == 0, f"{len(leak_train_val)} shared sequence(s)")
    all_ok &= ok
    ok = check("No identical sequences shared between val and test",
               len(leak_val_test) == 0, f"{len(leak_val_test)} shared sequence(s)")
    all_ok &= ok

    if leak_train_test:
        leaked_rows = df[df["sequence"].isin(leak_train_test)]
        print("  Leaked orig_ids:")
        print(leaked_rows[["orig_id", "label", "split"]].sort_values("sequence").to_string(index=False))

    # ------------------------------------------------------------------
    section("5. Re-derive confusion matrices from saved checkpoints")
    # ------------------------------------------------------------------
    try:
        sys.path.insert(0, str(ROOT / "src" / "nn"))
        from cnn import CNNClassifier
        from dnn import DNNClassifier
    except ImportError as e:
        print(f"Could not import model classes from src/nn ({e}); skipping section 5.")
        CNNClassifier = DNNClassifier = None

    if CNNClassifier is not None and y_test_path.exists():
        y_test = np.load(y_test_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for arch in ARCHS:
            for emb in EMBEDDINGS:
                key = f"{arch}_{emb}"
                model_path = MODELS_DIR / f"{key}.pt"
                x_path = SPLIT_DIR / f"{emb}_test.npy"
                if not model_path.exists() or not x_path.exists():
                    check(f"{key}: checkpoint + test embeddings present", False,
                          f"missing {model_path if not model_path.exists() else x_path}")
                    all_ok = False
                    continue

                X_test = np.load(x_path)
                input_dim = X_test.shape[1]
                model = (CNNClassifier(input_dim=input_dim) if arch == "cnn"
                         else DNNClassifier(input_dim=input_dim))
                state = torch.load(model_path, map_location=device)
                model.load_state_dict(state)
                model.to(device).eval()

                with torch.no_grad():
                    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                    logits = model(X_t).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                y_true = y_test.astype(int)
                tn = int(((preds == 0) & (y_true == 0)).sum())
                fp = int(((preds == 1) & (y_true == 0)).sum())
                fn = int(((preds == 0) & (y_true == 1)).sum())
                tp = int(((preds == 1) & (y_true == 1)).sum())

                recomputed = (tn, fp, fn, tp)
                reported = REPORTED.get(key)
                ok = check(
                    f"{key}: recomputed confusion matrix matches screenshot",
                    recomputed == reported,
                    f"recomputed TN/FP/FN/TP={recomputed}, screenshot={reported}",
                )
                all_ok &= ok

    # ------------------------------------------------------------------
    section("6. Tuned hyperparameters actually used?")
    # ------------------------------------------------------------------
    if TUNING_DIR.exists():
        found_any = False
        for arch in ARCHS:
            for emb in EMBEDDINGS:
                p = TUNING_DIR / f"{arch}_{emb}_best_params.json"
                if p.exists():
                    found_any = True
                    with open(p) as f:
                        params = json.load(f)
                    print(f"  {p.name}: {params['best_params']} (val_loss={params['best_val_loss']:.4f})")
        if not found_any:
            print("  No results/tuning/*_best_params.json files found — tuning may not have been run yet.")
    else:
        print("  results/tuning/ does not exist — tuning may not have been run yet.")

    print(
        "\n  NOTE: as of the current train_cnn.py / train_dnn.py, neither script exposes a "
        "--dropout / --dropout1/2/3 CLI flag. CNNClassifier(input_dim=...) and "
        "DNNClassifier(input_dim=...) are called with default dropout regardless of what "
        "tune_hyperparams.py found. If these new results were meant to reflect the tuned "
        "dropout values, double-check whether the training scripts were edited locally to "
        "accept them — otherwise only lr/batch_size/weight_decay (if passed via CLI) would "
        "actually differ from the old run."
    )

    # ------------------------------------------------------------------
    section("SUMMARY")
    # ------------------------------------------------------------------
    print("Overall:", "ALL CHECKS PASSED" if all_ok else "ONE OR MORE CHECKS FAILED — see above")


if __name__ == "__main__":
    main()