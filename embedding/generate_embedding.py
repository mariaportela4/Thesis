from pathlib import Path
import pandas as pd
import numpy as np

from embedding.embedding_extractor import ProteinEmbedder


# =========================
# CONFIG
# =========================
INPUT_CSV = "data/final_dataset.csv"
OUTPUT_DIR = Path("results/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "protein_id"
SEQ_COL = "sequence"
LABEL_COL = "label"

#MODEL_NAME = "facebook/esm2_t12_650M_UR50D" --> verificar no github
#MODEL_TYPE = "esm"   # "esm" or "prott5" or later "protbert"

MAX_LENGTH = 1024
POOLING = "mean"
#BATCH_SIZE = 8

MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
MODEL_TYPE = "prott5"
BATCH_SIZE = 2

#MODEL_NAME = "Rostlab/prot_bert"
#MODEL_TYPE = "protbert"
#BATCH_SIZE = 4

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_CSV)

required_cols = [ID_COL, SEQ_COL]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print(f"Loaded {len(df)} sequences from {INPUT_CSV}")

# Optional: remove empty sequences
df = df[df[SEQ_COL].notna()].copy()
df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip()
df = df[df[SEQ_COL] != ""].copy()

print(f"Remaining sequences after cleaning empty rows: {len(df)}")


# =========================
# LOAD MODEL
# =========================
embedder = ProteinEmbedder(
    model_name=MODEL_NAME,
    model_type=MODEL_TYPE
)

print(f"Using device and model: {MODEL_TYPE} / {MODEL_NAME}")
print(f"Embedding dimension: {embedder.get_embedding_dim()}")


# =========================
# EMBEDDINGS
# =========================
embeddings = embedder.embed_batch(
    sequences=df[SEQ_COL].tolist(),
    max_length=MAX_LENGTH,
    pooling=POOLING,
    batch_size=BATCH_SIZE
)

print("Embeddings shape:", embeddings.shape)


# =========================
# SAVE NUMPY
# =========================
np.save(OUTPUT_DIR / f"{MODEL_TYPE}_embeddings.npy", embeddings)

# Save metadata separately
meta_cols = [ID_COL]
if LABEL_COL in df.columns:
    meta_cols.append(LABEL_COL)

metadata = df[meta_cols].copy()
metadata.to_csv(OUTPUT_DIR / f"{MODEL_TYPE}_metadata.csv", index=False)

print(f"Saved NPY to: {OUTPUT_DIR / f'{MODEL_TYPE}_embeddings.npy'}")
print(f"Saved metadata to: {OUTPUT_DIR / f'{MODEL_TYPE}_metadata.csv'}")
