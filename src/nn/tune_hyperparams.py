import argparse
import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from cnn import CNNClassifier
from dnn import DNNClassifier

"""
tune_hyperparams.py

Optuna hyperparameter search for a single (architecture, embedding) combo.
Objective: minimize the validation loss of the epoch at which early stopping
halts the trial — the same criterion, and the same model, that train_cnn.py /
train_dnn.py would keep for that configuration.

Search space:
  - lr, batch_size, weight_decay (both architectures)
  - dropout (CNN, single value) or dropout1/2/3 (DNN, per-layer)
Layer widths are kept fixed, given the limited trial budget relative to a
small dataset — widening the search space further risks selecting a
configuration that generalizes poorly despite a good validation score.

Reproducibility: the Optuna sampler is seeded, and every trial re-seeds all
random number generators from --seed, so that repeated searches with the same
arguments explore the same configurations and train the same models.

Usage:
    python src/nn/tune_hyperparams.py --arch dnn --embedding esm --n-trials 25
    python src/nn/tune_hyperparams.py --arch dnn --embedding bert --n-trials 25
    python src/nn/tune_hyperparams.py --arch dnn --embedding prott5 --n-trials 25
    python src/nn/tune_hyperparams.py --arch cnn --embedding esm --n-trials 25
    python src/nn/tune_hyperparams.py --arch cnn --embedding bert --n-trials 25
    python src/nn/tune_hyperparams.py --arch cnn --embedding prott5 --n-trials 25
"""

def set_seed(seed):
    """
    Seed every random number generator involved in training and force
    deterministic cuDNN kernels. Called once per trial so that each
    configuration is trained under identical random conditions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(embedding_name):
    base_path = "results/embeddings/split"
    X_train = np.load(f"{base_path}/{embedding_name}_train.npy")
    X_val = np.load(f"{base_path}/{embedding_name}_val.npy")
    y_train = np.load(f"{base_path}/y_train.npy")
    y_val = np.load(f"{base_path}/y_val.npy")
    return X_train, X_val, y_train, y_val


def make_loaders(X_train, X_val, y_train, y_val, batch_size, seed):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )
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


def objective(trial, arch, X_train, X_val, y_train, y_val, input_dim, device,
              max_epochs, patience, min_delta, seed):
    """
    Train one configuration with early stopping and return the validation loss
    of the epoch at which it stopped.

    This mirrors train_cnn.py / train_dnn.py: those scripts keep the weights of
    the stopping epoch, so the score reported here is the score of the model
    that would actually be saved — not the best value seen along the way.
    """
    # Re-seed per trial so configurations differ only in their hyperparameters
    set_seed(seed)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = build_model(arch, input_dim, trial).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader, val_loader = make_loaders(X_train, X_val, y_train, y_val, batch_size, seed)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze(1)
                val_loss += criterion(outputs, y_batch).item() * len(y_batch)
                n_val += len(y_batch)
        val_loss /= n_val

        # Report to Optuna for pruning (stops clearly unpromising trials early)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # --- Early stopping check ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Recorded for diagnosis; the returned score is the stopping epoch's loss
    trial.set_user_attr("stop_epoch", epoch)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_loss", best_val_loss)

    return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["cnn", "dnn"])
    parser.add_argument("--embedding", required=True, choices=["esm", "bert", "prott5"])
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="Epoch ceiling per trial (kept shorter than final training's 50, for search speed)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Epochs with no val_loss improvement before a trial stops. "
                             "Must match the value used in train_cnn.py / train_dnn.py.")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum val_loss decrease that counts as an improvement. "
                             "Must match the value used in train_cnn.py / train_dnn.py.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the Optuna sampler and for every trial's training run.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"[+] Random seed: {args.seed}")
    print(f"[+] Early stopping -- patience={args.patience}, min_delta={args.min_delta}")

    X_train, X_val, y_train, y_val = load_data(args.embedding)
    input_dim = X_train.shape[1]
    print(f"{args.arch}_{args.embedding} — train: {X_train.shape}, val: {X_val.shape}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(
        lambda t: objective(t, args.arch, X_train, X_val, y_train, y_val, input_dim,
                            device, args.max_epochs, args.patience, args.min_delta, args.seed),
        n_trials=args.n_trials
    )

    print(f"\n[+] Best params for {args.arch}_{args.embedding}: {study.best_params}")
    print(f"[+] Best val_loss: {study.best_value:.4f}")
    print(f"[+] Stopped at epoch {study.best_trial.user_attrs['stop_epoch']} "
          f"(best epoch: {study.best_trial.user_attrs['best_epoch']})")

    out_dir = Path("results/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.arch}_{args.embedding}_best_params.json", "w") as f:
        json.dump({
            "best_params": study.best_params,
            "best_val_loss": study.best_value,
            "seed": args.seed,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "stop_epoch": study.best_trial.user_attrs["stop_epoch"],
            "best_epoch": study.best_trial.user_attrs["best_epoch"],
        }, f, indent=2)

    print(f"[+] Saved to {out_dir / f'{args.arch}_{args.embedding}_best_params.json'}")


if __name__ == "__main__":
    main()