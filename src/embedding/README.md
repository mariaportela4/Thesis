# src/embedding/

Scripts for generating protein language model embeddings (ESM-2 650M, ProtBERT, ProtT5-XL) from the final sequence sets.

## Files

- `embedding_extractor.py` — core embedding extraction logic (loads each pretrained protein language model and computes per-sequence embeddings). Sequence input is expected to already have ambiguous residues normalised to X (see `../normalize_ambiguous_aa.py`); this module validates rather than performs that normalisation itself.
- `generate_embedding.py` — entry-point script that runs extraction over the dataset and writes output to `../../results/embeddings/`. Sequences are sourced from `../../data/merged_input_normalized.fasta`, keyed by `orig_id`; the CSV (`../../data/final_dataset_40.csv`) supplies IDs and labels only.
- `__pycache__/` — compiled bytecode, not tracked.

## Notes

- Embeddings must be regenerated whenever the underlying FASTA changes — misalignment between embeddings and labels causes silent bugs (near-perfect but meaningless validation metrics). Always confirm `.npy` row order matches `../../data/final_dataset_40.csv`.
- Full (unsplit) embeddings land in `../../results/embeddings/`; the train/val/test partitioned versions (aligned to the GraphPart 40% split) are produced separately by `../prepare_split_embeddings.py` into `../../results/embeddings/split/`.
