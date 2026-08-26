import argparse
from pathlib import Path
from Bio import SeqIO
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

Usage:
    python src/embedding/generate_embedding.py --model-type esm
    python src/embedding/generate_embedding.py --model-type bert
    python src/embedding/generate_embedding.py --model-type prott5

CHANGE FROM PREVIOUS VERSION: model selection used to require manually
uncommenting one of three CONFIG blocks. Replaced with --model-type so this
script can be looped over from run_pipeline.py; behaviour and outputs are
otherwise identical.
"""

MODEL_CONFIGS = {
    "esm": {
        "name": "facebook/esm2_t33_650M_UR50D",
        "batch_size": 8,  # informational only — embed_sequences() doesn't batch internally
    },
    "prott5": {
        "name": "Rostlab/prot_t5_xl_half_uniref50-enc",
        "batch_size": 2,
    },
    "bert": {
        "name": "Rostlab/prot_bert",
        "batch_size": 4,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate protein embeddings for one model family.")
    parser.add_argument("--model-type", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--input-csv", default="data/final_dataset_40.csv")
    parser.add_argument("--fasta", default="data/merged_input_normalized.fasta")
    parser.add_argument("--output-dir", default="results/embeddings")
    parser.add_argument("--max-length", type=int, default=1024,
                         help="Max tokenised length; longer sequences are truncated.")
    args = parser.parse_args()

    model_type = args.model_type
    model_name = MODEL_CONFIGS[model_type]["name"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Column names expected in input_csv
    id_col = "orig_id"
    seq_col = "sequence"
    label_col = "label"

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(args.input_csv)

    if id_col not in df.columns:
        raise ValueError(f"Missing required column: {id_col}")

    print(f"Loaded {len(df)} sequences from {args.input_csv}")

    # Load sequences from the normalized FASTA, keyed by orig_id (parsed from header)
    orig_id_marker = 'orig_id='
    seqs_by_orig_id = {}
    for rec in SeqIO.parse(args.fasta, "fasta"):
        idx =  rec.id.find(orig_id_marker)
        if idx == -1:
            continue
        oid = rec.id[idx + len(orig_id_marker):]
        if oid:
            seqs_by_orig_id[oid] =str(rec.seq)
    print(f"Loaded {len(seqs_by_orig_id)} sequences from {args.fasta}")

    df[seq_col] = df[id_col].map(seqs_by_orig_id)

    # Guard: every row in the CSV must find a matching sequence in the FASTA
    missing = df[seq_col].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} rows in {args.input_csv} have no matching sequence in {args.fasta} — "
            f"check orig_id alignment between the two files"
        )

    df[seq_col] = df[seq_col].astype(str).str.strip()
    df = df[df[seq_col] != ""].copy()

    print(f"Remaining sequences after cleaning empty rows: {len(df)}")

    # =========================
    # LOAD MODEL
    # =========================
    embedder = ProteinEmbedder(model_name=model_name, model_type=model_type)

    print(f"Using device and model: {model_type} / {model_name}")
    print(f"Embedding dimension: {embedder.get_embedding_dim()}")

    # =========================
    # EMBEDDINGS
    # =========================
    embeddings = embedder.embed_sequences(
        sequences=df[seq_col].tolist(),
        max_length=args.max_length,
    )

    print("Embeddings shape:", embeddings.shape)

    # =========================
    # SAVE OUTPUTS
    # =========================
    np.save(output_dir / f"{model_type}_embeddings.npy", embeddings)

    meta_cols = [id_col]
    if label_col in df.columns:
        meta_cols.append(label_col)

    metadata = df[meta_cols].copy()
    metadata.to_csv(output_dir / f"{model_type}_metadata.csv", index=False)

    print(f"Saved NPY to: {output_dir / f'{model_type}_embeddings.npy'}")
    print(f"Saved metadata to: {output_dir / f'{model_type}_metadata.csv'}")


if __name__ == "__main__":
    main()