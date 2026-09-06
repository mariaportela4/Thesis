"""
multiseed_eval.py

Multi-seed evaluation for the six thesis models (cnn/dnn x esm/bert/prott5).

For each (architecture, embedding) combination, trains N independent runs
from scratch with different random seeds, using the best hyperparameters
from results/tuning/<arch>_<embedding>_best_params.json (same auto-load
behaviour as train_cnn.py / train_dnn.py). Each run is evaluated on the
held-out test split with the same metric set/order as benchmark_comparison.py
(f1, precision, recall, auc, accuracy). Output is a mean +/- std summary
per model.

Not run for DepoScope / DePP / PhageDPO retrained: their retraining
pipelines are external and not practical to re-run N times here, so this
covers only the six thesis models.

Place this file at Thesis/src/nn/multiseed_eval.py, alongside train_cnn.py,
train_dnn.py, cnn.py, dnn.py. 
    python src/nn/multiseed_eval.py
    python src/nn/multiseed_eval.py --seeds 13 42 123 2024 777 --epochs 50
    python src/nn/multiseed_eval.py --models cnn_esm dnn_prott5
    python src/nn/multiseed_eval.py --no-tuned-params   # use train_cnn.py/train_dnn.py defaults instead

Outputs
-------
    results/multiseed/<model>_seeds.csv     — one row per seed, per model
    results/multiseed/multiseed_all_runs.csv — all runs, all models, one table
    results/multiseed/multiseed_summary.csv  — mean +/- std per model, all metrics
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from cnn import CNNClassifier
from dnn import DNNClassifier

SPLIT_DIR = Path("results/embeddings/split")
TUNING_DIR = Path("results/tuning")
OUT_DIR = Path("results/multiseed")

DEFAULT_SEEDS = [13, 42, 123, 2024, 777]

# Same defaults as train_cnn.py / train_dnn.py, used only with --no-tuned-params
# or when no results/tuning/<arch>_<embedding>_best_params.json exists.
FALLBACK_PARAMS = {
    "cnn": {"lr": 1e-3, "batch_size": 32, "weight_decay": 1e-5, "dropout": 0.3},
    "dnn": {"lr": 1e-3, "batch_size": 32, "weight_decay": 1e-5,
            "dropout1": 0.4, "dropout2": 0.3, "dropout3": 0.2},
}

MODEL_CONFIGS = [
    ("cnn_esm", "cnn", "esm"),
    ("cnn_bert", "cnn", "bert"),
    ("cnn_prott5", "cnn", "prott5"),
    ("dnn_esm", "dnn", "esm"),
    ("dnn_bert", "dnn", "bert"),
    ("dnn_prott5", "dnn", "prott5"),
]


def set_seed(seed):
    """Seed every source of randomness that affects a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels; slower but reproducible per-seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_params(arch, embedding, use_tuned):
    """Load best Optuna hyperparameters unless --no-tuned-params was passed
    or no tuning file exists for this (arch, embedding) combo."""
    if use_tuned:
        params_path = TUNING_DIR / f"{arch}_{embedding}_best_params.json"
        if params_path.exists():
            with open(params_path) as f:
                data = json.load(f)
            params = dict(FALLBACK_PARAMS[arch])
            params.update(data["best_params"])
            return params, True
    return dict(FALLBACK_PARAMS[arch]), False


def load_data(embedding_name):
    X_train = np.load(SPLIT_DIR / f"{embedding_name}_train.npy")
    X_val = np.load(SPLIT_DIR / f"{embedding_name}_val.npy")
    X_test = np.load(SPLIT_DIR / f"{embedding_name}_test.npy")
    y_train = np.load(SPLIT_DIR / "y_train.npy")
    y_val = np.load(SPLIT_DIR / "y_val.npy")
    y_test = np.load(SPLIT_DIR / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size, seed):
    to_t = lambda a, dt: torch.tensor(a, dtype=dt)

    train_ds = TensorDataset(to_t(X_train, torch.float32), to_t(y_train, torch.float32))
    val_ds = TensorDataset(to_t(X_val, torch.float32), to_t(y_val, torch.float32))
    test_ds = TensorDataset(to_t(X_test, torch.float32), to_t(y_test, torch.float32))

    # Per-run generator so shuffling itself is tied to the run's seed, not
    # just weight init and dropout masks.
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_model(arch, input_dim, params):
    if arch == "cnn":
        return CNNClassifier(input_dim=input_dim, dropout=params["dropout"])
    return DNNClassifier(
        input_dim=input_dim,
        dropout1=params["dropout1"],
        dropout2=params["dropout2"],
        dropout3=params["dropout3"],
    )


def train_one_run(arch, embedding, params, epochs, seed, device):
    """Train a single model from scratch with a given seed. Keeps the
    best-val-loss state_dict in memory only (no .pt files written) and
    returns test-set metrics evaluated at that checkpoint."""
    set_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(embedding)
    input_dim = X_train.shape[1]

    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, params["batch_size"], seed
    )

    model = build_model(arch, input_dim, params).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: without this, CNN runs with Optuna-selected
            # higher LR + low dropout occasionally collapse to predicting
            # the majority class (seen empirically: cnn_bert seeds 123/777,
            # cnn_esm seed 42 all landed at accuracy == negative base rate).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze(1)
                val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Same keys/order as benchmark_comparison.py's results dict.
    metrics = {
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs),
        "accuracy": accuracy_score(all_labels, all_preds),
        "best_val_loss": best_val_loss,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                         help=f"Random seeds to run per model (default: {DEFAULT_SEEDS})")
    parser.add_argument("--epochs", type=int, default=50,
                         help="Epochs per run (default 50, matching final training budget)")
    parser.add_argument("--models", type=str, nargs="+",
                         choices=[m[0] for m in MODEL_CONFIGS],
                         default=[m[0] for m in MODEL_CONFIGS],
                         help="Subset of models to run (default: all six)")
    parser.add_argument("--no-tuned-params", action="store_true",
                         help="Ignore results/tuning/*.json and use train_cnn.py/"
                              "train_dnn.py defaults for every run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seeds: {args.seeds}  ({len(args.seeds)} runs per model)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [c for c in MODEL_CONFIGS if c[0] in args.models]
    all_rows = []

    for name, arch, embedding in configs:
        params, from_tuning = load_params(arch, embedding, use_tuned=not args.no_tuned_params)
        source = "results/tuning" if from_tuning else "fallback defaults"
        print(f"\n=== {name} ({source}) — params: {params} ===")

        rows = []
        for seed in args.seeds:
            print(f"  [seed {seed}] training...", end=" ", flush=True)
            metrics = train_one_run(arch, embedding, params, args.epochs, seed, device)
            row = {"model": name, "arch": arch, "embedding": embedding, "seed": seed, **metrics}
            rows.append(row)
            all_rows.append(row)
            print(f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")

        per_model_df = pd.DataFrame(rows)
        per_model_path = OUT_DIR / f"{name}_seeds.csv"
        per_model_df.to_csv(per_model_path, index=False)
        print(f"  -> saved {per_model_path}")

    # ----------------------------
    # Aggregate mean +/- std across seeds, per model
    # ----------------------------
    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(OUT_DIR / "multiseed_all_runs.csv", index=False)

    metric_cols = ["f1", "precision", "recall", "auc", "accuracy"]
    summary_rows = []
    for name in full_df["model"].unique():
        sub = full_df[full_df["model"] == name]
        row = {"model": name, "n_seeds": len(sub)}
        for m in metric_cols:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=1)  # sample std
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("f1_mean", ascending=False)
    summary_path = OUT_DIR / "multiseed_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[+] All per-seed results: {OUT_DIR / 'multiseed_all_runs.csv'}")
    print(f"[+] Summary (mean +/- std): {summary_path}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()