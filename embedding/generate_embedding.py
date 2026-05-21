from pathlib import Path
import pandas as pd
import numpy as np

from embedding_extractor import ProteinEmbedder

"""
Batch-embedding pipeline for protein sequences.

Loads a CSV dataset, generates fixed-length numerical embeddings for every
sequence using a pre-trained transformer model (via ProteinEmbedder), and
saves the results to disk in two files:

  results/embeddings/<model_type>_embeddings.npy  — NumPy array of shape
                                                     (n_sequences, hidden_size)
  results/embeddings/<model_type>_metadata.csv    — sequence IDs (and labels
                                                     if present) aligned
                                                     row-for-row with the array

Supported models (uncomment the relevant block in CONFIG to switch):
  - ProtBERT          (model_type="bert",    Rostlab/prot_bert)
  - ProtT5-XL         (model_type="prott5",  Rostlab/prot_t5_xl_half_uniref50-enc)
  - ESM-2 650 M       (model_type="esm",     facebook/esm2_t33_650M_UR50D)
"""

# =========================
# CONFIG
# =========================
input_csv = "data/final_dataset_40.csv"
output_dir = Path("results/embeddings")
output_dir.mkdir(parents=True, exist_ok=True)

# Column names expected in INPUT_CSV
id_col = "orig_id"  # Unique sequence identifier
seq_col = "sequence"
label_col = "label"

max_length = 1024  # Maximum tokenised length; longer sequences are truncated
pooling = "mean"    # Pooling strategy (currently informational; mean-pool is implemented inside ProteinEmbedder.embed_sequence)

# ----- Model selection -----
# Uncomment exactly ONE block below to choose which model to run.
# Each block sets MODEL_NAME, MODEL_TYPE, and a suggested BATCH_SIZE.

# -- ESM-2 650M  --
# MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
# MODEL_TYPE = "esm"
# BATCH_SIZE = 8

# -- ProtT5-XL  --
# MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
# MODEL_TYPE = "prott5"
# BATCH_SIZE = 2

# -- ProtBERT  --
MODEL_NAME = "Rostlab/prot_bert"
MODEL_TYPE = "bert"
BATCH_SIZE = 4

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(input_csv)

required_cols = [id_col, seq_col]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print(f"Loaded {len(df)} sequences from {input_csv}")

# Drop rows with missing or blank sequences; the model cannot process them.
df = df[df[seq_col].notna()].copy()
df[seq_col] = df[seq_col].astype(str).str.strip()
df = df[df[seq_col] != ""].copy()

print(f"Remaining sequences after cleaning empty rows: {len(df)}")


# =========================
# LOAD MODEL
# =========================
# ProteinEmbedder handles tokeniser selection, device placement, and
# model-specific pre-processing differences internally

embedder = ProteinEmbedder(
    model_name=MODEL_NAME,
    model_type=MODEL_TYPE
)

print(f"Using device and model: {MODEL_TYPE} / {MODEL_NAME}")
print(f"Embedding dimension: {embedder.get_embedding_dim()}")


# =========================
# EMBEDDINGS
# =========================

# embed_sequences iterates over all sequences and returns a 2-D NumPy array
# of shape (n_sequences, hidden_size) with one row per input sequence.
# NOTE: sequences are processed one at a time inside embed_sequences;

embeddings = embedder.embed_sequences(
    sequences = df[seq_col].tolist(),
    max_length = max_length
)

print("Embeddings shape:", embeddings.shape)


# =========================
# SAVE OUTPUTS
# =========================
# --- NumPy array ---
# Saved as a binary .npy file; load later with np.load(..., allow_pickle=False).
np.save(output_dir / f"{MODEL_TYPE}_embeddings.npy", embeddings)

# --- Metadata CSV ---
# Contains the sequence ID and, when available, the class label.
# Row order matches the embedding array exactly, so row i of the CSV
# corresponds to row i of the .npy file.
meta_cols = [id_col]
if label_col in df.columns:
    meta_cols.append(label_col)

metadata = df[meta_cols].copy()
metadata.to_csv(output_dir / f"{MODEL_TYPE}_metadata.csv", index=False)

print(f"Saved NPY to: {output_dir / f'{MODEL_TYPE}_embeddings.npy'}")
print(f"Saved metadata to: {output_dir / f'{MODEL_TYPE}_metadata.csv'}")
