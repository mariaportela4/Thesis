# results/cdhit/

CD-HIT clustering output at 80% sequence identity. **Run via the command line**. Confirmed pipeline position: runs on the deduplicated, hypothetical-protein-filtered sets, and its output is what gets class-balanced (1:2) and merged for GraphPart — see root `README.md` for the full order.

## Files

- `positive_cdhit80.fasta` / `positive_cdhit80.fasta.clstr` — clustered positive sequences and cluster assignments. 1,117 input → 593 output sequences.
- `negative_cdhit80.fasta` / `negative_cdhit80.fasta.clstr` — clustered negative sequences and cluster assignments. 23,102 input → 13,377 output sequences. This is the set that class balancing later samples 1,186 sequences from (`../../data/negative/negative_sampled.fasta`).
- `positive_cdhit80.fasta` — CD-HIT positive output

## Command used

Run separately on the deduplicated negative and positive sets:

```bash
cd-hit \
  -i data/negative/negative_dedup.faa \
  -o results/cdhit/negative_cdhit80.fasta \
  -c 0.8 \
  -n 5 \
  -d 0 \
  -T 4 \
  -M 0
```

Same command for the positive set, swapping input/output:

```bash
cd-hit \
  -i data/positive/all_DPO_positive_dedup.fasta \
  -o results/cdhit/positive_cdhit80.fasta \
  -c 0.8 \
  -n 5 \
  -d 0 \
  -T 4 \
  -M 0
```


### Flags

- `-i` — input FASTA file
- `-o` — output FASTA file (non-redundant sequences)
- `-c 0.8` — 80% sequence identity clustering threshold
- `-n 5` — word size for clustering (correct value for `-c 0.8`)
- `-d 0` — keeps the full header in the `.clstr` file
- `-T 4` — 4 threads
- `-M 0` — no memory limit (uses all available)

### Confirmed pipeline position

Input files (`negative_dedup.faa`, `all_DPO_positive_dedup.fasta`) are deduplicated **and** already hypothetical-protein-filtered at this point (both earlier steps run before CD-HIT). CD-HIT's own output then feeds directly into class balancing (`../../data/negative/negative_sampled.fasta`, sampled from `negative_cdhit80.fasta`) and the merge step (`../../data/merged_input.fasta`), ahead of GraphPart (`../graphpart/`). This was previously undocumented/uncertain in this repo's READMEs and has now been confirmed against `notebooks/01_domain_analysis.ipynb`.