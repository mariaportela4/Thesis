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
# SAVE LABELS
# ----------------------------

# Labels are saved once, shared across all model families.
# Row i of y_train.npy corresponds to row i of <model>_train.npy.

np.save(output_dir / "y_train.npy", y_all[train_idx])
np.save(output_dir / "y_val.npy", y_all[val_idx])
np.save(output_dir / "y_test.npy", y_all[test_idx])

print("\nSaved label files")

# ----------------------------
# SPLIT EACH EMBEDDING
# ----------------------------
for model_name, emb_path in embeddings.items():
    print(f"\nProcessing {model_name}...")

    X = np.load(emb_path)

    print("Original embedding shape:", X.shape)

    # Guard against shape mismatches that would silently produce wrong splits.
    # The number of embedding rows must equal the number of dataset rows.

    if X.shape[0] != len(df):
        raise ValueError(
            f"{model_name}: mismatch between embeddings and dataset rows"
        )

    # Index into the full embedding matrix using the pre-computed split indices.

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    # Persist each subset under a name that encodes both model and partition.
    np.save(output_dir / f"{model_name}_train.npy", X_train)
    np.save(output_dir / f"{model_name}_val.npy", X_val)
    np.save(output_dir / f"{model_name}_test.npy", X_test)

    print(f"{model_name}_train:", X_train.shape)
    print(f"{model_name}_val:", X_val.shape)
    print(f"{model_name}_test:", X_test.shape)

print("\nEverything is ready.")