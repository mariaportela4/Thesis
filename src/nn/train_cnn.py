from cnn import CNNClassifier

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
Training script for the 1-D CNN binary classifier (CNNClassifier) applied
to phage depolymerase (DPO) identification.

Loads pre-split protein embedding arrays, trains the model with
BCEWithLogitsLoss and Adam optimisation, and saves the trained weights
and per-epoch loss history to disk.

Training stops early once the validation loss has not improved by more than
--min-delta for --patience consecutive epochs; the weights saved are those
of the stopping epoch. All random number generators are seeded (--seed) so
that repeated runs with identical arguments produce identical models.

Hyperparameters (lr, batch_size, weight_decay, dropout) are loaded
automatically from results/tuning/cnn_<embedding>_best_params.json,
if that file exists (produced by tune_hyperparams.py). Any value passed
explicitly on the command line overrides the tuned one for that run only.

Usage
-----
    python src/nn/train_cnn.py --embedding esm
         uses results/tuning/cnn_esm_best_params.json if present,
         otherwise falls back to the architecture defaults below.

    python src/nn/train_cnn.py --embedding bert --lr 5e-4
         same, but overrides just the learning rate for this run.

    python src/nn/train_cnn.py --embedding prott5 --no-tuned-params
         ignores any tuned params file and uses the architecture
         defaults (lr=1e-3, batch_size=32, weight_decay=1e-5, dropout=0.3).

Outputs
-------
    results/models/cnn_<embedding>.pt              — model state dict
    results/history/cnn_<embedding>_history.csv    — epoch / train_loss / val_loss
    results/history/cnn_<embedding>_summary.json   — seed, stopping epoch, best epoch
"""

# REPRODUCIBILITY
def set_seed(seed):
    """
    Seed every random number generator involved in training and force
    deterministic cuDNN kernels, so that two runs with the same arguments
    produce identical weights.

    Parameters
    ----------
    seed : int
        Value applied to Python's ``random``, NumPy, and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(embedding_name):
    """
    Load pre-split embedding arrays and their corresponding labels from disk.

    Expects six .npy files under ``results/embeddings/split/``:
      - ``<embedding_name>_train.npy`` / ``y_train.npy``
      - ``<embedding_name>_val.npy``   / ``y_val.npy``
      - ``<embedding_name>_test.npy``  / ``y_test.npy``

    Parameters
    ----------
    embedding_name : str
        Model family identifier used as the filename prefix.
        Must be one of ``"esm"``, ``"bert"``, or ``"prott5"``.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Float32 embedding arrays of shape ``(n_samples, embedding_dim)``.
    y_train, y_val, y_test : np.ndarray
        Float32 binary label arrays of shape ``(n_samples,)``
        (0 = Non-DPO, 1 = DPO).
    """
    base_path = "results/embeddings/split"

    X_train = np.load(f"{base_path}/{embedding_name}_train.npy")
    X_val = np.load(f"{base_path}/{embedding_name}_val.npy")
    X_test = np.load(f"{base_path}/{embedding_name}_test.npy")

    y_train = np.load(f"{base_path}/y_train.npy")
    y_val = np.load(f"{base_path}/y_val.npy")
    y_test = np.load(f"{base_path}/y_test.npy")

    return X_train, X_val, X_test, y_train, y_val, y_test


# DataLoader Creation
def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size, seed):
    """
    Convert NumPy arrays into PyTorch DataLoaders.

    The training loader uses ``shuffle=True`` so the model sees batches in
    a different order each epoch, which reduces the risk of the optimiser
    following spurious gradient patterns tied to data ordering. It is given
    an explicitly seeded generator so that this shuffling is reproducible
    across runs. Validation and test loaders use ``shuffle=False`` to keep
    results aligned with the metadata CSV.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Embedding arrays of shape ``(n_samples, embedding_dim)``.
    y_train, y_val, y_test : np.ndarray
        Binary label arrays of shape ``(n_samples,)``.
    batch_size : int
        Number of samples per mini-batch for all three loaders.
    seed : int
        Seed for the training loader's shuffling generator.

    Returns
    -------
    train_loader : DataLoader
        Shuffled, seeded DataLoader for the training split.
    val_loader : DataLoader
        Ordered DataLoader for the validation split.
    test_loader : DataLoader
        Ordered DataLoader for the test split.
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
        shuffle=True,                        # Shuffle training data each epoch
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False                        # Preserve order for reproducible evaluation
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


# Hyperparameter Loading
def load_best_params(arch, embedding, tuning_dir="results/tuning"):
    """
    Load the best hyperparameters found by tune_hyperparams.py for a given
    (arch, embedding) combination, if the file exists.

    Parameters
    ----------
    arch : str
        "cnn" or "dnn" — used to build the expected filename.
    embedding : str
        "esm", "bert", or "prott5".
    tuning_dir : str
        Directory where tune_hyperparams.py writes its output.

    Returns
    -------
    dict or None
        The "best_params" dict (lr, batch_size, weight_decay, dropout, ...)
        if the file exists, otherwise None.
    """
    path = f"{tuning_dir}/{arch}_{embedding}_best_params.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["best_params"]


# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                epochs, patience, min_delta):
    """
    Run the training loop with early stopping on the validation loss.

    Each epoch consists of:
      1. A training phase — gradients are computed and weights updated.
      2. A validation phase — run under ``torch.no_grad()`` to measure
         generalisation without updating weights.

    The model is switched between ``train()`` and ``eval()`` modes so that
    Batch Normalisation and Dropout behave correctly in each phase.

    Training stops as soon as ``patience`` consecutive epochs pass without the
    validation loss improving by more than ``min_delta``. The weights left in
    ``model`` are those of the stopping epoch, not of the best epoch; the best
    epoch is still recorded in the returned summary for diagnosis.

    Parameters
    ----------
    model : nn.Module
        CNN classifier to train (already moved to ``device``).
    train_loader : DataLoader
        Shuffled DataLoader for the training split.
    val_loader : DataLoader
        Ordered DataLoader for the validation split.
    criterion : nn.Module
        Loss function. ``nn.BCEWithLogitsLoss`` is expected; it applies
        sigmoid internally, which is numerically more stable than
        applying sigmoid before ``nn.BCELoss``.
    optimizer : torch.optim.Optimizer
        Optimiser used to update model parameters (Adam recommended).
    device : torch.device
        Device on which tensors and the model reside.
    epochs : int
        Maximum number of full passes over the training data.
    patience : int
        Epochs with no validation improvement tolerated before stopping.
    min_delta : float
        Minimum decrease in validation loss that counts as an improvement.

    Returns
    -------
    history : list[dict]
        One dict per epoch: ``{"epoch": int, "train_loss": float,
        "val_loss": float}``. Suitable for direct conversion to a DataFrame
        with ``pd.DataFrame(history)``.
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
        help="Embedding type to train the CNN model."
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
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--no-tuned-params",
        action="store_true",
        help="Ignore results/tuning/cnn_<embedding>_best_params.json even if present, "
             "and fall back to the architecture defaults."
    )

    args = parser.parse_args()

    print(f"Training CNN using {args.embedding} embeddings")

    # Seed everything before any model or loader is created
    set_seed(args.seed)
    print(f"[+] Random seed: {args.seed}")

    # --- Resolve hyperparameters: tuned JSON < CLI override < hardcoded defaults ---
    ARCH_DEFAULTS = {
        "lr": 1e-3, "batch_size": 32, "weight_decay": 1e-5, "dropout": 0.3,
    }

    best_params = None if args.no_tuned_params else load_best_params("cnn", args.embedding)
    if best_params is not None:
        print(f"[+] Loaded tuned hyperparameters from "
              f"results/tuning/cnn_{args.embedding}_best_params.json: {best_params}")
    else:
        print("[!] No tuned hyperparameters found (or --no-tuned-params set) -- "
              "using architecture defaults / CLI values only.")

    def resolve(name):
        # 1. Explicit CLI value always wins
        cli_val = getattr(args, name)
        if cli_val is not None:
            return cli_val
        # 2. Otherwise use the tuned value, if available
        if best_params is not None and name in best_params:
            return best_params[name]
        # 3. Fall back to the architecture default
        return ARCH_DEFAULTS[name]

    args.lr = resolve("lr")
    args.batch_size = resolve("batch_size")
    args.weight_decay = resolve("weight_decay")
    args.dropout = resolve("dropout")

    print(f"[+] Final hyperparameters -- lr={args.lr}, batch_size={args.batch_size}, "
          f"weight_decay={args.weight_decay}, dropout={args.dropout}")
    print(f"[+] Early stopping -- patience={args.patience}, min_delta={args.min_delta}")

    # --- Load data ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.embedding)

    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    # --- Build DataLoaders ---
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

    # --- Model, loss, optimiser ---
    input_dim = X_train.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CNNClassifier(input_dim=input_dim, dropout=args.dropout).to(device)

    # BCEWithLogitsLoss fuses sigmoid + BCE in a single numerically stable op
    criterion = nn.BCEWithLogitsLoss()

    # Adam with L2 weight decay (weight_decay) for additional regularisation
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # --- Save outputs ---
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/history", exist_ok=True)

    # Model weights only (state_dict); use CNNClassifier(input_dim)
    model_path = f"results/models/cnn_{args.embedding}.pt"

    # --- Train ---
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

    # Loss history — one row per epoch, columns: epoch / train_loss / val_loss
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        f"results/history/cnn_{args.embedding}_history.csv",
        index=False
    )

    summary_path = f"results/history/cnn_{args.embedding}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({**summary, "seed": args.seed}, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: results/history/cnn_{args.embedding}_history.csv")
    print(f"Run summary saved to: {summary_path}")


if __name__ == "__main__":
    main()