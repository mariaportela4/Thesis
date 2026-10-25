# src/embedding/

Scripts for generating protein language model embeddings (ESM-2 650M, ProtBERT, ProtT5-XL-U50).

## Files

- `embedding_extractor.py` — `ProteinEmbedder`, the core extraction logic: loads each pretrained model and computes one fixed-length vector per sequence. ProtBERT and ProtT5 require whitespace-separated residues, which this module inserts; ESM-2 tokenises the raw sequence directly. Input is expected to have ambiguous residues already normalised to X, which this module validates rather than performs.
- `generate_embedding.py` — entry point: runs extraction across the dataset and writes `../../results/embeddings/<model>_embeddings.npy` plus a matching metadata CSV. Sequences come from `../../data/merged_input_normalized.fasta`, parsed with marker-based `orig_id=` header matching rather than naive splitting, so accessions containing pipes are not truncated.

## Notes

- A 1024-token cap is applied uniformly across all three models. No padding is used, since sequences are embedded one at a time.
- Embeddings must be regenerated whenever the underlying FASTA changes; misalignment between embeddings and labels causes silent, meaningless metrics.
- The train/val/test partitioned versions are produced separately by `../prepare_split_embeddings.py`.