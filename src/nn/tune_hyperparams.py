"""
tune_hyperparams.py

Optuna hyperparameter search for a single (architecture, embedding) combo.
Objective: minimize the best validation loss reached during a short training
run, consistent with the best-checkpoint criterion used in train_cnn.py /
train_dnn.py.

Search space:
  - lr, batch_size, weight_decay (both architectures)
  - dropout (CNN, single value) or dropout1/2/3 (DNN, per-layer)
Layer widths are kept fixed, given the limited trial budget relative to a
small dataset — widening the search space further risks selecting a
configuration that generalizes poorly despite a good validation score.

Usage:
    python src/nn/tune_hyperparams.py --arch dnn --embedding esm --n-trials 25
    python src/nn/tune_hyperparams.py --arch dnn --embedding bert --n-trials 25
    python src/nn/tune_hyperparams.py --arch dnn --embedding prott5 --n-trials 25
    python src/nn/tune_hyperparams.py --arch cnn --embedding esm --n-trials 25
    python src/nn/tune_hyperparams.py --arch cnn --embedding bert --n-trials 25
    python src/nn/tune_hyperparams.py --arch cnn --embedding prott5 --n-trials 25
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from cnn import CNNClassifier
from dnn import DNNClassifier

def load_data(embedding_name):
    base_path = "results/embeddings/split"
    X_train = np.load(f"{base_path}/{embedding_name}_train.npy")
    X_val = np.load(f"{base_path}/{embedding_name}_val.npy")
    y_train = np.load(f"{base_path}/y_train.npy")
    y_val = np.load(f"{base_path}/y_val.npy")
    return X_train, X_val, y_train, y_val

def make_loaders(X_train, X_val, y_train, y_val, batch_size):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def build_model(arch, input_dim, trial):
    if arch == "cnn":
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        return CNNClassifier(input_dim=input_dim, dropout=dropout)
    else:
        dropout1 = trial.suggest_float("dropout1", 0.2, 0.6)
        dropout2 = trial.suggest_float("dropout2", 0.1, 0.5)
        dropout3 = trial.suggest_float("dropout3", 0.05, 0.4)
        return DNNClassifier(input_dim=input_dim, dropout1=dropout1, dropout2=dropout2, dropout3=dropout3)

def objective(trial, arch, X_train, X_val, y_train, y_val, input_dim, device, max_epochs):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = build_model(arch, input_dim, trial).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader, val_loader = make_loaders(X_train, X_val, y_train, y_val, batch_size)

    best_val_loss = float("inf")
    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze(1)
                val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(val_loader)

        best_val_loss = min(best_val_loss, val_loss)

        # Report to Optuna for pruning (stops clearly unpromising trials early)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["cnn", "dnn"])
    parser.add_argument("--embedding", required=True, choices=["esm", "bert", "prott5"])
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=30,
                         help="Epochs per trial (kept shorter than final training's 50, for search speed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, X_val, y_train, y_val = load_data(args.embedding)
    input_dim = X_train.shape[1]
    print(f"{args.arch}_{args.embedding} — train: {X_train.shape}, val: {X_val.shape}")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(
        lambda t: objective(t, args.arch, X_train, X_val, y_train, y_val, input_dim, device, args.max_epochs),
        n_trials=args.n_trials
    )

    print(f"\n[+] Best params for {args.arch}_{args.embedding}: {study.best_params}")
    print(f"[+] Best val_loss: {study.best_value:.4f}")

    out_dir = Path("results/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.arch}_{args.embedding}_best_params.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_val_loss": study.best_value}, f, indent=2)

    print(f"[+] Saved to {out_dir / f'{args.arch}_{args.embedding}_best_params.json'}")


if __name__ == "__main__":
    main()   
