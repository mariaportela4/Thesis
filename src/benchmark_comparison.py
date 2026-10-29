"""
Benchmark comparison across:
  - The 6 thesis models (cnn/dnn x esm/bert/prott5)
  - DepoScope, retrained on the thesis training set
  - DePP, retrained on the thesis training set
  - PhageDPO, retrained on the thesis training set
  - DposFinder, retrained on the thesis training set

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
    accuracy_score, f1_score, precision_score, recall_score, average_precision_score,
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
PHAGEDPO_DIR = Path("results/phagedpo_retrain")
DPOSFINDER_DIR = Path("results/dposfinder_retrain_trainonly")
OUTPUT_DIR = Path("results")

# DposFinder retraining was investigated after producing perfect scores
# across all metrics. Checked and ruled out: train/test sequence overlap
# (0 exact and 0 prefix matches), metrics-computation bugs, and label-parsing
# bugs. Reproduced independently at both batch_size=1 and batch_size=16 (same
# result both times, with a healthy/gradual training curve at batch_size=16:
# 97.7% -> 99.4% -> 100% accuracy over the first 3 epochs, not an instant
# collapse) -- this rules out the batch_size=1 gradient-update-count artifact
# as the explanation. Predicted probabilities remain highly polarized toward
# 0/1 in both runs, consistent with end-to-end fine-tuning of a large
# pretrained transformer on a well-separated binary task, not a training
# artifact. Included in the benchmark with this caveat noted in the thesis
# discussion section. Set to False to exclude again if needed.
INCLUDE_DPOSFINDER = True

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
    cm_path = OUTPUT_DIR / f"confusion_matrix_{name}.png"
    plot_confusion_matrix(cm, title=f"Confusion Matrix — {name}", show=False, save_path=cm_path)
    print(f"[+] Confusion matrix saved to {cm_path}")
    results[name] = {
        "f1": f1_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "pr_auc": average_precision_score(y_true, probs),
        "accuracy": accuracy_score(y_true, preds),
        "probs": probs,
        "preds": preds,
    }
    print(f"[+] {name}: F1={results[name]['f1']:.4f}  PR-AUC={results[name]['pr_auc']:.4f}")


# ----------------------------
# 3. DepoScope retrained — positional, same test order assumption (validated above)
# ----------------------------
deposcope_probs = np.load(DEPOSCOPE_DIR / "test_probabilities.npy").squeeze()
deposcope_preds = (deposcope_probs >= 0.5).astype(int)

results["deposcope_retrained"] = {
    "f1": f1_score(y_true, deposcope_preds),
    "precision": precision_score(y_true, deposcope_preds),
    "recall": recall_score(y_true, deposcope_preds),
    "pr_auc": average_precision_score(y_true, deposcope_probs),
    "accuracy": accuracy_score(y_true, deposcope_preds),
    "probs": deposcope_probs,
    "preds": deposcope_preds,
}
print(f"[+] deposcope_retrained: F1={results['deposcope_retrained']['f1']:.4f}  "
      f"PR-AUC={results['deposcope_retrained']['pr_auc']:.4f}")


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
    "pr_auc": average_precision_score(y_true, depp_probs),
    "accuracy": accuracy_score(y_true, depp_preds),
    "probs": depp_probs,
    "preds": depp_preds,
}
print(f"[+] depp_retrained: F1={results['depp_retrained']['f1']:.4f}  "
      f"PR-AUC={results['depp_retrained']['pr_auc']:.4f}")


# ----------------------------
# 5. PhageDPO retrained — merge by orig_id (robust to ordering AND to the
#    ambiguous-AA sequences that phagedpo_retrain.py drops before featurizing;
#    the test set is currently confirmed free of these, but the merge stays
#    correct either way instead of silently assuming positional alignment)
# ----------------------------
phagedpo_df = pd.read_csv(PHAGEDPO_DIR / "phagedpo_retrained_predictions.csv")

merged_phagedpo = test_df[["orig_id", "label"]].merge(phagedpo_df, on="orig_id", how="left")
if merged_phagedpo["Probability_DPO"].isna().any():
    missing = merged_phagedpo[merged_phagedpo["Probability_DPO"].isna()]
    raise ValueError(f"{len(missing)} test sequences missing from PhageDPO predictions "
                      f"(likely dropped for containing a non-standard amino acid): "
                      f"{missing['orig_id'].tolist()[:5]}")

phagedpo_probs = merged_phagedpo["Probability_DPO"].values
phagedpo_preds = (phagedpo_probs >= 0.5).astype(int)

results["phagedpo_retrained"] = {
    "f1": f1_score(y_true, phagedpo_preds),
    "precision": precision_score(y_true, phagedpo_preds),
    "recall": recall_score(y_true, phagedpo_preds),
    "pr_auc": average_precision_score(y_true, phagedpo_probs),
    "accuracy": accuracy_score(y_true, phagedpo_preds),
    "probs": phagedpo_probs,
    "preds": phagedpo_preds,
}
print(f"[+] phagedpo_retrained: F1={results['phagedpo_retrained']['f1']:.4f}  "
      f"PR-AUC={results['phagedpo_retrained']['pr_auc']:.4f}")


# ----------------------------
# 6. DposFinder retrained — merge by seq_id, parsed from the 'label' column
#    of DposFinder's own predict-mode output (test_set_result.tsv).
#
#    DposFinder's final_train mode only prints aggregate metrics -- it does
#    NOT save per-sequence predictions. Before this section can run, DposFinder's
#    own predict mode must be run once against the retrained checkpoint:
#
#      python code/main.py --mode predict \
#          --data_path <abs path to DPOSFINDER_DIR>/ \
#          --test_data test_set.fasta \
#          --name dposfinder_retrained
#
#    (run from the DposFinder repo root, in the DposFinder conda env)
#
#    That writes test_set_result.tsv into DPOSFINDER_DIR with columns
#    label / prob / pred. The 'label' column is the full FASTA header written
#    by dposfinder_retrain.py: "{seq_id}|{orig_id} pos"/"neg" -- so seq_id is
#    recovered the same way as for DePP (split on "|").
#
#    Gated by INCLUDE_DPOSFINDER (see CONFIG) -- set False to skip cleanly.
# ----------------------------
if INCLUDE_DPOSFINDER:
    dposfinder_tsv = DPOSFINDER_DIR / "test_set_result.tsv"
    if not dposfinder_tsv.exists():
        raise FileNotFoundError(
            f"{dposfinder_tsv} not found. Run DposFinder's own --mode predict "
            "against the retrained checkpoint first (see comment above this "
            "block for the exact command)."
        )

    dposfinder_df = pd.read_csv(dposfinder_tsv, sep="\t")
    dposfinder_df["seq_id"] = dposfinder_df["label"].str.split("|").str[0]

    # only pull seq_id/prob across -- dposfinder_df also has its own 'label'
    # column (the FASTA header), which would collide with test_df's 'label'
    # column (positive/negative) if merged wholesale
    merged_dposfinder = test_df[["seq_id", "label"]].merge(
        dposfinder_df[["seq_id", "prob"]], on="seq_id", how="left"
    )
    if merged_dposfinder["prob"].isna().any():
        missing = merged_dposfinder[merged_dposfinder["prob"].isna()]
        raise ValueError(f"{len(missing)} test sequences missing from DposFinder predictions: "
                          f"{missing['seq_id'].tolist()[:5]}")

    dposfinder_probs = merged_dposfinder["prob"].values
    dposfinder_preds = (dposfinder_probs >= 0.5).astype(int)

    results["dposfinder_retrained"] = {
        "f1": f1_score(y_true, dposfinder_preds),
        "precision": precision_score(y_true, dposfinder_preds),
        "recall": recall_score(y_true, dposfinder_preds),
        "pr_auc": average_precision_score(y_true, dposfinder_probs),
        "accuracy": accuracy_score(y_true, dposfinder_preds),
        "probs": dposfinder_probs,
        "preds": dposfinder_preds,
    }
    print(f"[+] dposfinder_retrained: F1={results['dposfinder_retrained']['f1']:.4f}  "
          f"PR-AUC={results['dposfinder_retrained']['pr_auc']:.4f}")
else:
    print("[!] Skipping DposFinder (INCLUDE_DPOSFINDER=False)")


# ----------------------------
# 7. Assemble metrics table
# ----------------------------
metrics_df = pd.DataFrame([
    {
        "model": name,
        "f1": r["f1"],
        "precision": r["precision"],
        "recall": r["recall"],
        "pr_auc": r["pr_auc"],
        "accuracy": r["accuracy"],
    }
    for name, r in results.items()
]).sort_values("f1", ascending=False)

metrics_df.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
print(f"\n[+] Metrics table saved to {OUTPUT_DIR / 'benchmark_metrics.csv'}")
print(metrics_df.to_string(index=False))

# ----------------------------
# 8. Per-sequence predictions (for bootstrap CIs)
# ----------------------------
preds_df = pd.DataFrame({
    "seq_id": test_df["seq_id"].to_numpy(),
    "label": y_true,
})
for name, r in results.items():
    probs_col = np.asarray(r["probs"]).ravel()
    if len(probs_col) != len(preds_df):
        raise ValueError(
            f"{name}: {len(probs_col)} probabilities for "
            f"{len(preds_df)} test sequences"
        )
    preds_df[name] = probs_col

preds_path = OUTPUT_DIR / "benchmark_predictions.csv"
preds_df.to_csv(preds_path, index=False)
print(f"[+] Per-sequence predictions saved to {preds_path}")

# ----------------------------
# 9. Bar plot: Precision, Recall, F1 (as previously requested by the supervisor)
# ----------------------------
EXTERNAL_TOOLS = {"deposcope_retrained", "depp_retrained", "phagedpo_retrained",
                   "dposfinder_retrained"}

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
ax.set_title("Benchmark comparison: thesis models vs. retrained DepoScope, DePP, PhageDPO and DposFinder", pad=28)
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
# 10. Confusion matrices for the four retrained external tools
# ----------------------------
confusion_matrix_tools = ["deposcope_retrained", "depp_retrained", "phagedpo_retrained"]
if INCLUDE_DPOSFINDER:
    confusion_matrix_tools.append("dposfinder_retrained")

for name in confusion_matrix_tools:
    cm = confusion_matrix(y_true, results[name]["preds"])
    cm_path = OUTPUT_DIR / f"confusion_matrix_{name}.png"
    plot_confusion_matrix(cm, title=f"Confusion Matrix — {name}", show=False, save_path=cm_path)
    print(f"[+] Confusion matrix saved to {cm_path}")


# ----------------------------
# 11. Additional visualizations: heatmap, dot plot, radar small multiples.
#     Reuse ordered_df from section 8 (already sorted: thesis models first
#     by F1, then external tools by F1) so all benchmark figures share the
#     same model ordering.
# ----------------------------
from matplotlib import cm as mpl_cm  # local alias -- `cm` above is already
                                      # bound to sklearn's confusion_matrix

METRICS = ["f1", "precision", "recall", "pr_auc", "accuracy"]
METRIC_LABELS = {"f1": "F1", "precision": "Precision", "recall": "Recall",
                  "pr_auc": "PR-AUC", "accuracy": "Accuracy"}

viz_df = ordered_df.reset_index(drop=True).copy()
viz_df["is_external"] = viz_df["model"].isin(EXTERNAL_TOOLS)


def plot_heatmap(df, out_path):
    data = df[METRICS].values
    n_models, n_metrics = data.shape

    fig, ax = plt.subplots(figsize=(7, 0.5 * n_models + 1.5))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(df["model"])

    for i in range(n_models):
        for j in range(n_metrics):
            val = data[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                     color=text_color, fontsize=8)

    n_thesis = (~df["is_external"]).sum()
    if 0 < n_thesis < n_models:
        ax.axhline(n_thesis - 0.5, color="black", linewidth=1.5)

    ax.set_title("Benchmark metrics heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Heatmap saved to {out_path}")


def plot_dotplot(df, out_path):
    n_models = len(df)
    fig, ax = plt.subplots(figsize=(9, 0.5 * n_models + 1.5))

    y_positions = np.arange(n_models)
    dot_colors = plt.colormaps["tab10"](np.linspace(0, 1, len(METRICS)))

    for y in y_positions:
        ax.plot([0, 1], [y, y], color="lightgray", linewidth=1, zorder=1)

    for j, metric in enumerate(METRICS):
        ax.scatter(df[metric], y_positions, color=dot_colors[j], s=60,
                   label=METRIC_LABELS[metric], zorder=3,
                   edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["model"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Score")
    ax.set_title("Benchmark metrics — dot plot")

    n_thesis = (~df["is_external"]).sum()
    if 0 < n_thesis < n_models:
        ax.axhline(n_thesis - 0.5, color="black", linewidth=1, linestyle="--")

    ax.legend(title="Metric", loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=len(METRICS), framealpha=0.9)
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Dot plot saved to {out_path}")


def plot_radar_small_multiples(df, out_path):
    """
    One small radar (pentagon, 5 metrics) per model, in a grid. A single
    overlaid radar with this many models would be unreadable -- small
    multiples keep each model's shape legible while still allowing visual
    comparison of overall shape/balance across metrics.
    """
    n_models = len(df)
    n_cols = 5
    n_rows = int(np.ceil(n_models / n_cols))

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows),
                              subplot_kw=dict(polar=True))
    axes = np.array(axes).reshape(-1)

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        values = [row[m] for m in METRICS]
        values += values[:1]

        color = "#DD8452" if row["is_external"] else "#4C72B0"
        ax.plot(angles, values, color=color, linewidth=1.5)
        ax.fill(angles, values, color=color, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["0.5", "1.0"], fontsize=6)
        ax.set_title(row["model"], fontsize=9, pad=12)

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Benchmark metrics — per-model radar (blue = thesis models, orange = retrained external tools)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Radar small multiples saved to {out_path}")


plot_heatmap(viz_df, OUTPUT_DIR / "benchmark_heatmap.png")
plot_dotplot(viz_df, OUTPUT_DIR / "benchmark_dotplot.png")
plot_radar_small_multiples(viz_df, OUTPUT_DIR / "benchmark_radar.png")