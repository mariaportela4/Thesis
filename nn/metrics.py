import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader

"""
Evaluation utilities for the phage depolymerase (DPO) binary classifier.

Provides:
  - create_eval_loaders  : build PyTorch DataLoaders from pre-split .npy files
  - evaluate_model       : run inference and return precision/recall/F1 + confusion matrix
  - plot_training_history_csv : plot train vs. validation loss from a CSV log
  - create_metrics_table : assemble a tidy DataFrame from a results dictionary
  - plot_confusion_matrix         : heatmap for a single confusion matrix
  - plot_all_confusion_matrices   : heatmap for every entry in a results dictionary
  - compare_embeddings    : rank embedding types by mean metric across models
  - plot_metric_by_embedding : bar chart of a chosen metric grouped by embedding
  - plot_metric_by_model     : bar chart of a chosen metric grouped by model architecture
"""

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def create_eval_loaders(split_dir, embedding_name, batch_size=32):
    X_train = np.load(f"{split_dir}/{embedding_name}_train.npy")
    X_val = np.load(f"{split_dir}/{embedding_name}_val.npy")
    X_test = np.load(f"{split_dir}/{embedding_name}_test.npy")

    y_val = np.load(f"{split_dir}/y_val.npy")
    y_test = np.load(f"{split_dir}/y_test.npy")

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    input_dim = X_train.shape[1]

    return val_loader, test_loader, input_dim

def evaluate_model(model, test_loader, device):
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    metrics = {
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
    }

    cm = confusion_matrix(all_labels, all_preds)

    return metrics, cm


def plot_training_history_csv(csv_path, title="Training History"):
    df = pd.read_csv(csv_path)

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, df["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, df["val_loss"], marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def create_metrics_table(results_dict):
    rows = []

    for name, result in results_dict.items():
        model_name, embedding_name = name.split(" + ")

        metrics = result["metrics"]

        rows.append({
            "model": model_name,
            "embedding": embedding_name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

    df = pd.DataFrame(rows)

    return df


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-DPO", "DPO"],
        yticklabels=["Non-DPO", "DPO"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_all_confusion_matrices(results_dict):
    for name, result in results_dict.items():
        plot_confusion_matrix(
            result["cm"],
            title=f"Confusion Matrix: {name}"
        )


def compare_embeddings(metrics_df):
    embedding_comparison = (
        metrics_df
        .groupby("embedding")[["precision", "recall", "f1_score"]]
        .mean()
        .sort_values(by="mcc", ascending=False)
    )

    return embedding_comparison


def plot_metric_by_embedding(metrics_df, metric="mcc"):
    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=metrics_df,
        x="embedding",
        y=metric,
        errorbar=None
    )

    plt.xlabel("Embedding")
    plt.ylabel(metric.upper())
    plt.title(f"Mean {metric.upper()} by Embedding Type")

    plt.tight_layout()
    plt.show()


def plot_metric_by_model(metrics_df, metric="mcc"):
    plt.figure(figsize=(7, 5))

    sns.barplot(
        data=metrics_df,
        x="model",
        y=metric,
        errorbar=None
    )

    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.title(f"Mean {metric.upper()} by Model Type")

    plt.tight_layout()
    plt.show()