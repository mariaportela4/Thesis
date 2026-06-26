"""
benchmark_comparison.py
-----------------------
Compares the thesis models (ESM-2, ProtBERT, ProtT5 + DNN/CNN) against
external DPO prediction tools (DepoScope, DePP) on the test set.

Outputs:
    - results/benchmark_metrics.csv   — F1, Precision, Recall
    - results/benchmark_barplot.png   — Bar chart (Precision, Recall, F1)

Usage:
    python benchmark_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score,precision_score, recall_score)


# ── Paths ──────────────────────────────────────────────────────────────────────
SPLIT_DIR       = "results/embeddings/split"
RESULTS_DIR     = "results"
TEST_SET_CSV    = "data/test_set.csv"
DEPOSCOPE_CSV   = "results/deposcope_predictions.csv"
DEPP_CSV        = "results/depp_predictions.csv"


# ── Load test labels ───────────────────────────────────────────────────────────
def load_test_labels() -> tuple[np.ndarray, pd.DataFrame]:
    """Load ground-truth labels for the test set."""
    y_test = np.load(f"{SPLIT_DIR}/y_test.npy")
    test_df = pd.read_csv(TEST_SET_CSV)[["seq_id", "orig_id", "label"]].copy()
    test_df["y_true"] = (test_df["label"] == "positive").astype(int)
    return y_test, test_df


# ── Load model probabilities ───────────────────────────────────────────────────
def load_my_models() -> dict[str, np.ndarray]:
    """Load predicted probabilities from the thesis DNN/CNN models."""
    models = {}
    for emb in ["esm", "bert", "prott5"]:
        for arch in ["dnn", "cnn"]:
            path = f"{SPLIT_DIR}/{emb}_{arch}_probs.npy"
            models[f"{emb}_{arch}"] = np.load(path)
    return models


# ── Load external tool predictions ────────────────────────────────────────────
def load_external_tools(test_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Load DepoScope and DePP predictions and align them with the test set.

    Returns:
        deposcope: dict with 'y_true' and 'probs'
        depp:      dict with 'y_true' and 'probs'
    """
    # DepoScope
    ds_df = pd.read_csv(DEPOSCOPE_CSV)[["protein_id", "scores_DepoScope"]]
    ds_merged = test_df.merge(ds_df, left_on="orig_id", right_on="protein_id", how="inner")
    deposcope = {
        "y_true": ds_merged["y_true"].values,
        "probs":  ds_merged["scores_DepoScope"].values,
    }

    # DePP
    dp_df = pd.read_csv(DEPP_CSV)
    dp_df.columns = ["seq_id", "prob_depp"]
    dp_merged = test_df.merge(dp_df, on="seq_id", how="inner")
    depp = {
        "y_true": dp_merged["y_true"].values,
        "probs":  dp_merged["prob_depp"].values,
    }

    return deposcope, depp


# ── Compute metrics ────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute F1, Precision, and Recall."""
    preds = (probs >= threshold).astype(int)
    return {
        "F1":        round(f1_score(y_true, preds), 3),
        "Precision": round(precision_score(y_true, preds, zero_division=0), 3),
        "Recall":    round(recall_score(y_true, preds), 3),
    }


# ── Plot bar chart ─────────────────────────────────────────────────────────────
def plot_barplot(metrics_df: pd.DataFrame, output_path: str) -> None:
    """Bar chart comparing Precision, Recall and F1 across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    n_my  = len(metrics_df) - 2
    colors = ["#4C72B0"] * n_my + ["#DD8452", "#55A868"]

    for ax, metric in zip(axes, ["Precision", "Recall", "F1"]):
        values = metrics_df[metric]
        bars = ax.bar(values.index, values.values, color=colors)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, values.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    plt.suptitle("Benchmark: Thesis Models vs External Tools", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")



# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    y_test, test_df = load_test_labels()
    my_models       = load_my_models()
    deposcope, depp = load_external_tools(test_df)

    # Compute metrics for all models
    all_metrics: dict[str, dict] = {}
    for name, probs in my_models.items():
        all_metrics[name] = compute_metrics(y_test, probs)
    all_metrics["DepoScope"] = compute_metrics(deposcope["y_true"], deposcope["probs"])
    all_metrics["DePP"]      = compute_metrics(depp["y_true"],      depp["probs"])

    metrics_df = pd.DataFrame(all_metrics).T
    print(metrics_df.to_string())
    metrics_df.to_csv(f"{RESULTS_DIR}/benchmark_metrics.csv")
    print(f"\nSaved: {RESULTS_DIR}/benchmark_metrics.csv")

    # Plots
    plot_barplot(metrics_df, f"{RESULTS_DIR}/benchmark_barplot.png")

    print("\nDone!")


if __name__ == "__main__":
    main()