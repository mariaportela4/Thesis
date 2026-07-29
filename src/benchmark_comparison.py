"""
Benchmark comparison across:
  - The 6 thesis models (cnn/dnn x esm/bert/prott5)
  - DepoScope, retrained on the thesis training set
  - DePP, retrained on the thesis training set

All evaluated on the same test split (final_dataset_40.csv, split == 'test').

"""

import json
import sys
from pathlib import Path

# Ensure the project root (~/Thesis) is importable regardless of the CWD
# the script is invoked from, so `nn.cnn` / `nn.dnn` / `nn.metrics` resolve
# correctly even when running `python src/benchmark_comparison.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from nn.cnn import CNNClassifier
from nn.dnn import DNNClassifier
from nn.metrics import plot_confusion_matrix

# ----------------------------
# CONFIG
# ----------------------------
DATASET_CSV = "data/final_dataset_40.csv"
SPLIT_DIR = Path("results/embeddings/split")
MODELS_DIR = Path("results/models")
DEPOSCOPE_DIR = Path("results/deposcope_retrain")
DEPP_DIR = Path("results/depp_retrain")
OUTPUT_DIR = Path("results")

EMBEDDING_DIMS = {
    "esm": None,     # filled in automatically from the .npy shape
    "bert": None,
    "prott5": None,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# 1. Canonical test order (from the CSV itself, same order used to build the .npy splits)
# ----------------------------
df = pd.read_csv(DATASET_CSV)
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"[+] Canonical test set: {len(test_df)} sequences (from {DATASET_CSV})")

label_map = {"positive": 1, "negative": 0}
y_true_from_csv = test_df["label"].map(label_map).values.astype(int)

# Sanity check against y_test.npy, produced by prepare_split_embeddings.py
# using the exact same filtering/order logic. If these don't match, something
# is misaligned and results downstream cannot be trusted.
y_test_npy = np.load(SPLIT_DIR / "y_test.npy").astype(int)
if not np.array_equal(y_true_from_csv, y_test_npy):
    raise ValueError(
        "y_test.npy does not match the labels derived from final_dataset_40.csv "
        "in the same order. Do not trust downstream results until this is resolved "
        "-- check whether prepare_split_embeddings.py was re-run after the CSV changed."
    )
print("[+] y_test.npy matches CSV-derived labels — safe to assume positional alignment.")

y_true = y_test_npy  # from here on, this is the ground truth used for every model


# ----------------------------
# 2. The 6 thesis models: load weights, run inference on X_test
# ----------------------------
results = {}

model_configs = [
    ("cnn_esm", "cnn", "esm"),
    ("cnn_bert", "cnn", "bert"),
    ("cnn_prott5", "cnn", "prott5"),
    ("dnn_esm", "dnn", "esm"),
    ("dnn_bert", "dnn", "bert"),
    ("dnn_prott5", "dnn", "prott5"),
]

for name, arch, embedding in model_configs:
    X_test = np.load(SPLIT_DIR / f"{embedding}_test.npy")
    input_dim = X_test.shape[1]

    if arch == "cnn":
        model = CNNClassifier(input_dim)
    else:
        model = DNNClassifier(input_dim)

    weights_path = MODELS_DIR / f"{name}.pt"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int)
    
    cm = confusion_matrix(y_true, preds)
    plot_confusion_matrix(cm, title=f"Confusion Matrix — {name}")
    cm_path = OUTPUT_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Confusion matrix saved to {cm_path}")
    results[name] = {
        "f1": f1_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "auc": roc_auc_score(y_true, probs),
        "accuracy": accuracy_score(y_true, preds),
        "probs": probs,
        "preds": preds,
    }
    print(f"[+] {name}: F1={results[name]['f1']:.4f}  AUC={results[name]['auc']:.4f}")


# ----------------------------
# 3. DepoScope retrained — positional, same test order assumption (validated above)
# ----------------------------
deposcope_probs = np.load(DEPOSCOPE_DIR / "test_probabilities.npy").squeeze()
deposcope_preds = (deposcope_probs >= 0.5).astype(int)

results["deposcope_retrained"] = {
    "f1": f1_score(y_true, deposcope_preds),
    "precision": precision_score(y_true, deposcope_preds),
    "recall": recall_score(y_true, deposcope_preds),
    "auc": roc_auc_score(y_true, deposcope_probs),
    "accuracy": accuracy_score(y_true, deposcope_preds),
    "probs": deposcope_probs,
    "preds": deposcope_preds,
}
print(f"[+] deposcope_retrained: F1={results['deposcope_retrained']['f1']:.4f}  "
      f"AUC={results['deposcope_retrained']['auc']:.4f}")


# ----------------------------
# 4. DePP retrained — merge by seq_id (robust to ordering)
# ----------------------------
depp_df = pd.read_csv(DEPP_DIR / "depp_retrained_predictions.csv")
depp_df["seq_id"] = depp_df["name"].str.split("|").str[0]

# Align to the canonical test_df order via seq_id merge
merged_depp = test_df[["seq_id", "label"]].merge(depp_df, on="seq_id", how="left")
if merged_depp["Probability_DePol"].isna().any():
    missing = merged_depp[merged_depp["Probability_DePol"].isna()]
    raise ValueError(f"{len(missing)} test sequences missing from DePP predictions: "
                      f"{missing['seq_id'].tolist()[:5]}")

depp_probs = merged_depp["Probability_DePol"].values
depp_preds = (depp_probs >= 0.5).astype(int)

results["depp_retrained"] = {
    "f1": f1_score(y_true, depp_preds),
    "precision": precision_score(y_true, depp_preds),
    "recall": recall_score(y_true, depp_preds),
    "auc": roc_auc_score(y_true, depp_probs),
    "accuracy": accuracy_score(y_true, depp_preds),
    "probs": depp_probs,
    "preds": depp_preds,
}
print(f"[+] depp_retrained: F1={results['depp_retrained']['f1']:.4f}  "
      f"AUC={results['depp_retrained']['auc']:.4f}")


# ----------------------------
# 5. Assemble metrics table
# ----------------------------
metrics_df = pd.DataFrame([
    {
        "model": name,
        "f1": r["f1"],
        "precision": r["precision"],
        "recall": r["recall"],
        "auc": r["auc"],
        "accuracy": r["accuracy"],
    }
    for name, r in results.items()
]).sort_values("f1", ascending=False)

metrics_df.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
print(f"\n[+] Metrics table saved to {OUTPUT_DIR / 'benchmark_metrics.csv'}")
print(metrics_df.to_string(index=False))


# ----------------------------
# 6. Bar plot: Precision, Recall, F1 (as previously requested by the supervisor)
# ----------------------------
EXTERNAL_TOOLS = {"deposcope_retrained", "depp_retrained"}

# Order: thesis models first (by F1, already sorted in metrics_df), then external tools
thesis_rows = metrics_df[~metrics_df["model"].isin(EXTERNAL_TOOLS)]
external_rows = metrics_df[metrics_df["model"].isin(EXTERNAL_TOOLS)].sort_values("f1", ascending=False)
ordered_df = pd.concat([thesis_rows, external_rows])

plot_df = ordered_df.set_index("model")[["precision", "recall", "f1"]]

fig, ax = plt.subplots(figsize=(13, 6.5))

n_models = len(plot_df)
n_metrics = 3
bar_width = 0.25
x = np.arange(n_models)
colors = {"precision": "#4C72B0", "recall": "#DD8452", "f1": "#55A868"}

for i, metric in enumerate(["precision", "recall", "f1"]):
    offset = (i - 1) * bar_width
    ax.bar(x + offset, plot_df[metric], width=bar_width,
           label=metric.capitalize(), color=colors[metric])

# Visually separate the thesis models from the external retrained tools
n_thesis = len(thesis_rows)
if 0 < n_thesis < n_models:
    sep_x = n_thesis - 0.5
    ax.axvline(sep_x, color="gray", linestyle="--", linewidth=1)
    ax.text(n_thesis / 2 - 0.5, 1.05, "Thesis models",
            ha="center", va="bottom", fontsize=10, color="dimgray")
    ax.text(n_thesis + (n_models - n_thesis) / 2 - 0.5, 1.05, "Retrained external tools",
            ha="center", va="bottom", fontsize=10, color="dimgray")

ax.set_xticks(x)
ax.set_xticklabels(plot_df.index, rotation=40, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.set_title("Benchmark comparison: thesis models vs. retrained DepoScope and DePP", pad=28)
ax.legend(title="Metric", loc="lower right", framealpha=0.9)
ax.yaxis.grid(True, linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "benchmark_barplot.png", dpi=200)
print(f"[+] Bar plot saved to {OUTPUT_DIR / 'benchmark_barplot.png'}")
plt.close()


# ----------------------------
# 7. Confusion matrices for the two retrained external tools
# ----------------------------
for name in ["deposcope_retrained", "depp_retrained"]:
    cm = confusion_matrix(y_true, results[name]["preds"])
    plot_confusion_matrix(cm, title=f"Confusion Matrix — {name}")
    cm_path = OUTPUT_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Confusion matrix saved to {cm_path}")