from dnn import DNNClassifier

import os
import json
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

"""
Training script for the DNN binary classifier (DNNClassifier) applied to
phage depolymerase (DPO) identification.

Loads pre-split protein embedding arrays, trains the model with
BCEWithLogitsLoss and Adam optimisation, and saves the trained weights
and per-epoch loss history to disk.

Training stops early once the validation loss has not improved by more than
--min-delta for --patience consecutive epochs; the weights saved are those
of the stopping epoch. All random number generators are seeded (--seed) so
that repeated runs with identical arguments produce identical models.

Hyperparameters (lr, batch_size, weight_decay, dropout1-3) are loaded
automatically from results/tuning/dnn_<embedding>_best_params.json,
if that file exists (produced by tune_hyperparams.py). Any value passed
explicitly on the command line overrides the tuned one for that run only.

Usage
-----
    python src/nn/train_dnn.py --embedding esm
        # uses results/tuning/dnn_esm_best_params.json if present,
        # otherwise falls back to the architecture defaults below.

    python src/nn/train_dnn.py --embedding bert --lr 5e-4
        # same, but overrides just the learning rate for this run.

    python src/nn/train_dnn.py --embedding prott5 --no-tuned-params
        # ignores any tuned params file and uses the architecture defaults.

Outputs
-------
    results/models/dnn_<embedding>.pt              — model state dict
    results/history/dnn_<embedding>_history.csv    — epoch / train_loss / val_loss
    results/history/dnn_<embedding>_summary.json   — seed, stopping epoch, best epoch
"""

def set_seed(seed):
    """
    Seed every random number generator involved in training and force
    deterministic cuDNN kernels, so that two runs with the same arguments
    produce identical weights.
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
    X_test = np.load(f"{base_path}/{embedding_name}_test.npy")

    y_train = np.load(f"{base_path}/y_train.npy")
    y_val = np.load(f"{base_path}/y_val.npy")
    y_test = np.load(f"{base_path}/y_test.npy")

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size, seed):
    """
    Convert NumPy arrays into PyTorch DataLoaders.

    The training loader is given an explicitly seeded generator so that the
    shuffled batch order is reproducible across runs.
    """
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def load_best_params(arch, embedding, tuning_dir="results/tuning"):
    """
    Load the best hyperparameters found by tune_hyperparams.py for a given
    (arch, embedding) combination, if the file exists.
    """
    path = f"{tuning_dir}/{arch}_{embedding}_best_params.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["best_params"]


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                epochs, patience, min_delta):
    """
    Run the training loop with early stopping on the validation loss.

    Training stops as soon as ``patience`` consecutive epochs pass without the
    validation loss improving by more than ``min_delta``. The weights left in
    ``model`` are those of the stopping epoch, not of the best epoch; the best
    epoch is still recorded in the returned summary for diagnosis.

    Returns
    -------
    history : list[dict]
        One dict per epoch: ``{"epoch", "train_loss", "val_loss"}``.
    summary : dict
        Stopping epoch, best epoch and the corresponding validation losses.
    """
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_train = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            n_train += len(y_batch)

        train_loss /= n_train

        model.eval()
        val_loss, n_val = 0.0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch).squeeze(1)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * len(y_batch)
                n_val += len(y_batch)

        val_loss /= n_val

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # --- Early stopping check ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                early_stopped = True
                print(f"  --> Early stopping at epoch {epoch} "
                      f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})")
                break

    summary = {
        "early_stopped": early_stopped,
        "stop_epoch": epoch,
        "final_val_loss": val_loss,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    return history, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=["esm", "bert", "prott5"],
        help="Embedding type to train the DNN model."
    )

    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs; early stopping may end training sooner.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Epochs with no val_loss improvement before training stops.")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum val_loss decrease that counts as an improvement.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for weight initialisation, dropout and batch shuffling.")
    # Default to None so tuned values (if present) can fill the gaps;
    # anything the user passes explicitly on the CLI always wins.
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout1", type=float, default=None)
    parser.add_argument("--dropout2", type=float, default=None)
    parser.add_argument("--dropout3", type=float, default=None)
    parser.add_argument(
        "--no-tuned-params",
        action="store_true",
        help="Ignore results/tuning/dnn_<embedding>_best_params.json even if present, "
             "and fall back to the architecture defaults."
    )

    args = parser.parse_args()

    print(f"Training DNN using {args.embedding} embeddings")

    # Seed everything before any model or loader is created
    set_seed(args.seed)
    print(f"[+] Random seed: {args.seed}")

    # --- Resolve hyperparameters: tuned JSON < CLI override < hardcoded defaults ---
    ARCH_DEFAULTS = {
        "lr": 1e-3, "batch_size": 32, "weight_decay": 1e-5,
        "dropout1": 0.4, "dropout2": 0.3, "dropout3": 0.2,
    }

    best_params = None if args.no_tuned_params else load_best_params("dnn", args.embedding)
    if best_params is not None:
        print(f"[+] Loaded tuned hyperparameters from "
              f"results/tuning/dnn_{args.embedding}_best_params.json: {best_params}")
    else:
        print("[!] No tuned hyperparameters found (or --no-tuned-params set) -- "
              "using architecture defaults / CLI values only.")

    def resolve(name):
        cli_val = getattr(args, name)
        if cli_val is not None:
            return cli_val
        if best_params is not None and name in best_params:
            return best_params[name]
        return ARCH_DEFAULTS[name]

    args.lr = resolve("lr")
    args.batch_size = resolve("batch_size")
    args.weight_decay = resolve("weight_decay")
    args.dropout1 = resolve("dropout1")
    args.dropout2 = resolve("dropout2")
    args.dropout3 = resolve("dropout3")

    print(f"[+] Final hyperparameters -- lr={args.lr}, batch_size={args.batch_size}, "
          f"weight_decay={args.weight_decay}, "
          f"dropout=({args.dropout1}, {args.dropout2}, {args.dropout3})")
    print(f"[+] Early stopping -- patience={args.patience}, min_delta={args.min_delta}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.embedding)

    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    train_loader, val_loader, _ = create_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        args.batch_size,
        args.seed
    )

    input_dim = X_train.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DNNClassifier(
        input_dim=input_dim,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        dropout3=args.dropout3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/history", exist_ok=True)

    model_path = f"results/models/dnn_{args.embedding}.pt"

    history, summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta
    )

    # Weights of the stopping epoch, saved once after training ends
    torch.save(model.state_dict(), model_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        f"results/history/dnn_{args.embedding}_history.csv",
        index=False
    )

    summary_path = f"results/history/dnn_{args.embedding}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({**summary, "seed": args.seed}, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: results/history/dnn_{args.embedding}_history.csv")
    print(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()