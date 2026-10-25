"""
Per-sequence error analysis across the six thesis models.

For each of the six thesis models (2 architectures x 3 embeddings),
loads predictions on the held-out test set and identifies sequences
misclassified by three or more models -- candidates for genuinely
difficult cases or noisy labels, as opposed to single-model failures.

Usage:
    python src/nn/error_analysis.py
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch

from nn.cnn import CNNClassifier
from nn.dnn import DNNClassifier

DATASET_CSV = "data/final_dataset_40.csv"
SPLIT_DIR = Path("results/embeddings/split")
MODELS_DIR = Path("results/models")
OUTPUT_CSV = Path("results/error_analysis.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(DATASET_CSV)
test_df = df[df["split"] == "test"].reset_index(drop=True)

label_map = {"positive": 1, "negative": 0}
y_true = test_df["label"].map(label_map).values.astype(int)

model_configs = [
    ("dnn_esm", "dnn", "esm"),
    ("dnn_bert", "dnn", "bert"),
    ("dnn_prott5", "dnn", "prott5"),
    ("cnn_esm", "cnn", "esm"),
    ("cnn_bert", "cnn", "bert"),
    ("cnn_prott5", "cnn", "prott5"),
]

per_model_probs = {}
per_model_preds = {}

for name, arch, embedding in model_configs:
    X_test = np.load(SPLIT_DIR / f"{embedding}_test.npy")
    input_dim = X_test.shape[1]
    model = CNNClassifier(input_dim) if arch == "cnn" else DNNClassifier(input_dim)
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        probs = torch.sigmoid(model(X_tensor).squeeze(1)).cpu().numpy()
    per_model_probs[name] = probs
    per_model_preds[name] = (probs >= 0.5).astype(int)
    print(f"[+] {name}: loaded, {(per_model_preds[name] != y_true).sum()} errors")

result = test_df[["seq_id", "orig_id", "product", "organism", "label"]].copy()
error_count = np.zeros(len(test_df), dtype=int)

for name, _, _ in model_configs:
    result[f"{name}_prob"] = per_model_probs[name]
    is_wrong = per_model_preds[name] != y_true
    result[f"{name}_wrong"] = is_wrong
    error_count += is_wrong.astype(int)

result["n_models_wrong"] = error_count
result = result.sort_values("n_models_wrong", ascending=False)

flagged = result[result["n_models_wrong"] >= 3]
print(f"\n[+] {len(flagged)} sequence(s) misclassified by >= 3 of 6 models:\n")
print(flagged[["seq_id", "orig_id", "product", "organism", "label", "n_models_wrong"]].to_string(index=False))

result.to_csv(OUTPUT_CSV, index=False)
print(f"\n[+] Full per-sequence results saved to {OUTPUT_CSV}")
