import pandas as pd
import numpy as np
from pathlib import Path 

"""
Splits pre-computed protein-sequence embeddings and their labels into
train / validation / test subsets, then saves each subset as a separate
NumPy binary file.

The split assignments are read from the 'split' column of the master CSV
(values: "train", "valid", "test"), so no random splitting is performed
here — the partitioning is fully deterministic and reproducible.

Input
-----
data/final_dataset_40.csv          — master dataset with 'split' and 'label' columns
results/embeddings/<model>_embeddings.npy — one file per model family

Output (all written to results/embeddings/split/)
------
y_train.npy / y_val.npy / y_test.npy          — float32 label arrays (0 = negative, 1 = positive)
<model>_train.npy / <model>_val.npy / <model>_test.npy — embedding sub-arrays per model
"""


# ----------------------------
# CONFIG
# ----------------------------
dataset_path = "data/final_dataset_40.csv"

# Mapping from model name to its pre-computed embedding file.
# Add or remove entries here to include additional model families.

embeddings = {
    "esm": "results/embeddings/esm_embeddings.npy",
    "bert": "results/embeddings/bert_embeddings.npy",
    "prott5": "results/embeddings/prott5_embeddings.npy"
}

output_dir = Path("results/embeddings/split")
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# LOAD DATASET
# ----------------------------
df = pd.read_csv(dataset_path)

print("Dataset shape:", df.shape)

# Inspect the pre-assigned split and class distributions before proceeding.
print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nLabel distribution:")
print(df["label"].value_counts())

# ----------------------------
# LABEL CONVERSION
# ----------------------------
# Map string labels to binary floats required by downstream classifiers.
# float32 is used for direct compatibility with PyTorch loss functions.

label_map = { "negative": 0, "positive": 1}

y_all = df["label"].map(label_map).values.astype(np.float32)

# ----------------------------
# SPLIT INDICES
# ----------------------------

# Derive integer positional indices for each partition from the 'split' column.
# Using the DataFrame index (not row numbers) keeps alignment correct even if
# the CSV is not zero-indexed or contains gaps.

train_idx = df[df["split"] == "train"].index
val_idx = df[df["split"] == "valid"].index
test_idx = df[df["split"] == "test"].index

print("\nTrain samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Test samples:", len(test_idx))

# ----------------------------
# SPLIT EACH EMBEDDING
# ----------------------------
y_saved = False  # only save labels once

for model_name, emb_path in embeddings.items():
    print(f"\nProcessing {model_name}...")

    X = np.load(emb_path)
    print("Original embedding shape:", X.shape)

    meta = pd.read_csv(f"results/embeddings/{model_name}_metadata.csv")

    # Align embedding rows to match df row order by orig_id
    merged = df[["orig_id", "split", "label"]].merge(
    meta[["orig_id"]].reset_index().rename(columns={"index": "emb_idx"}),
    on="orig_id",
    how="inner"
)

    # Guard: all sequences must be matched
    if len(merged) != len(df):
        raise ValueError(
            f"{model_name}: only {len(merged)}/{len(df)} sequences matched — check orig_id alignment"
        )

    X_aligned = X[merged["emb_idx"].values]
    y_aligned = merged["label"].map(label_map).values.astype(np.float32)

    train_mask = merged["split"] == "train"
    val_mask   = merged["split"] == "valid"
    test_mask  = merged["split"] == "test"

    np.save(output_dir / f"{model_name}_train.npy", X_aligned[train_mask])
    np.save(output_dir / f"{model_name}_val.npy",   X_aligned[val_mask])
    np.save(output_dir / f"{model_name}_test.npy",  X_aligned[test_mask])

    print(f"{model_name}_train: {X_aligned[train_mask].shape}")
    print(f"{model_name}_val:   {X_aligned[val_mask].shape}")
    print(f"{model_name}_test:  {X_aligned[test_mask].shape}")

    # Save labels only once (identical across models after alignment)
    if not y_saved:
        np.save(output_dir / "y_train.npy", y_aligned[train_mask])
        np.save(output_dir / "y_val.npy",   y_aligned[val_mask])
        np.save(output_dir / "y_test.npy",  y_aligned[test_mask])
        y_saved = True

print("\nEverything is ready.")