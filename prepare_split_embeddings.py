import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = "data/final_dataset_40.csv"

EMBEDDINGS = {
    "esm": "results/embeddings/esm_embeddings.npy",
    "bert": "results/embeddings/bert_embeddings.npy",
    "prott5": "results/embeddings/prott5_embeddings.npy"
}

OUTPUT_DIR = Path("results/embeddings/split")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# LOAD DATASET
# ----------------------------
df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)
print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nLabel distribution:")
print(df["label"].value_counts())

# ----------------------------
# LABEL CONVERSION
# ----------------------------
label_map = {
    "negative": 0,
    "positive": 1
}

y_all = df["label"].map(label_map).values.astype(np.float32)

# ----------------------------
# SPLIT INDICES
# ----------------------------
train_idx = df[df["split"] == "train"].index
val_idx = df[df["split"] == "valid"].index
test_idx = df[df["split"] == "test"].index

print("\nTrain samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Test samples:", len(test_idx))

# ----------------------------
# SAVE LABELS
# ----------------------------
np.save(OUTPUT_DIR / "y_train.npy", y_all[train_idx])
np.save(OUTPUT_DIR / "y_val.npy", y_all[val_idx])
np.save(OUTPUT_DIR / "y_test.npy", y_all[test_idx])

print("\nSaved label files")

# ----------------------------
# SPLIT EACH EMBEDDING
# ----------------------------
for model_name, emb_path in EMBEDDINGS.items():
    print(f"\nProcessing {model_name}...")

    X = np.load(emb_path)

    print("Original embedding shape:", X.shape)

    if X.shape[0] != len(df):
        raise ValueError(
            f"{model_name}: mismatch between embeddings and dataset rows"
        )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    np.save(OUTPUT_DIR / f"{model_name}_train.npy", X_train)
    np.save(OUTPUT_DIR / f"{model_name}_val.npy", X_val)
    np.save(OUTPUT_DIR / f"{model_name}_test.npy", X_test)

    print(f"{model_name}_train:", X_train.shape)
    print(f"{model_name}_val:", X_val.shape)
    print(f"{model_name}_test:", X_test.shape)

print("\nEverything is ready.")