"""
run_pipeline.py
================
Single entry point to (re)run the Python portions of the DPO classification
pipeline, end to end, in the correct order.
 
Lives OUTSIDE src/, at the project root (/home/mcarvalho/Thesis/run_pipeline.py),
and drives the existing scripts under src/ via subprocess — it does not
reimplement any of their logic.
 
IMPORTANT — this script does NOT run CD-HIT or GraphPart. Both require
external command-line tools (cd-hit, needleall/EMBOSS) and are documented
as separate terminal steps — see notebook_terminal_steps.md (or the
markdown cells it's meant to be pasted into, in 01_domain_analysis.ipynb).
 
 
Usage (run from the project root):

 
    python run_pipeline.py --list                # show all stages + where the
                                                   # terminal-only steps fit in
    python run_pipeline.py --all                  # run every Python stage in order
    python run_pipeline.py --stage dedup normalize
    python run_pipeline.py --stage train           # just the 6 thesis models
    python run_pipeline.py --stage tune            # opt-in, not part of --all
 
Each stage shells out to the real script and stops immediately (non-zero
exit) if that script fails, so a broken run never silently continues into
the next stage.
"""
 
import argparse
import subprocess
import sys
import time
from pathlib import Path
 
ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
 
EMBEDDINGS = ["esm", "bert", "prott5"]
ARCHS = ["dnn", "cnn"]
 
 
def run(cmd, cwd=ROOT, label=None):
    """Run a subprocess, streaming its output, and hard-stop on failure."""
    label = label or " ".join(str(c) for c in cmd)
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}")
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {label}  (exit code {result.returncode}, {elapsed:.1f}s)")
        sys.exit(result.returncode)
    print(f"[OK] {label}  ({elapsed:.1f}s)")
 
 
# ---------------------------------------------------------------------------
# STAGES — each one calls existing scripts under src/. Adjust paths below if
# your on-disk filenames differ (marked with NOTE where I had to assume).
# ---------------------------------------------------------------------------
 
def stage_combine_domain_fasta():
    """Combine per-domain positive FASTAs (domain_*.fasta) into one candidate FASTA."""
    run(
        [PYTHON, "src/combine_domain_fasta.py"],
        label="Combine domain FASTAs",
    )
 
 
def stage_remove_hypothetical():
    """Filter out 'hypothetical protein' entries from the raw candidate sets."""
    run(
        [PYTHON, "src/remove_hypothetical_proteins.py"],
        label="Remove hypothetical proteins",
    )
 
 
def stage_dedup():
    """Remove exact-duplicate sequences from the positive and negative sets."""
    run(
        [PYTHON, "src/remove_duplicates.py",
         "data/positive/all_DPO_positive.fasta",
         "data/positive/all_DPO_positive_dedup.fasta"],
        label="Dedup: positives",
    )
    run(
        [PYTHON, "src/remove_duplicates.py",
         "data/negative/negative_nohyp.faa",
         "data/negative/negative_dedup.faa"],
        label="Dedup: negatives",
    )
 
"""
Run CD-HIT in Terminal

cd-hit \
  -i data/negative/negative_dedup.faa \
  -o results/cdhit/negative_cdhit80.fasta \
  -c 0.8 \
  -n 5 \
  -d 0 \
  -T 4 \
  -M 0

 cd-hit \
  -i data/negative/all_DPO_positive_dedup.fasta \
  -o results/cdhit/positive_cdhit80.fasta \
  -c 0.8 \
  -n 5 \
  -d 0 \
  -T 4 \
  -M 0 
  
  """
 
def stage_balance_merge():
    """Subsample negatives 1:2 and merge CD-HIT-clustered pos+neg into merged_input.fasta.
 
    NOTE: requires CD-HIT (terminal, run on BOTH positive and negative
    dedup sets) to already have produced results/cdhit/positive_cdhit80.fasta
    and results/cdhit/negative_cdhit80.fasta.
    """
    run([PYTHON, "src/balance_and_merge.py"], label="Balance negatives + merge")


 
def stage_normalize():
    """Normalize ambiguous residues (B/Z/J/U/O -> X) on the merged FASTA."""
    run(
        [PYTHON, "src/normalize_ambiguous_aa.py",
         "--input", "data/merged_input.fasta",
         "--output", "data/merged_input_normalized.fasta"],
        label="Normalize ambiguous residues",
    )
 
"""
GraphPart (40% identity) 

graphpart needle \
  --fasta-file data/merged_input_normalized.fasta \
  --threshold 0.4 \
  --out-file results/graphpart/graphpart_assignments_40.csv \
  --labels-name label \
  --test-ratio 0.1 \
  --val-ratio 0.05 \
  --threads 12
 
 """
 
def stage_export_alignment_fasta():
    """Write per-split (train/test/valid) FASTAs for 40%  GraphPart dataset.
 
    Used for Needleman-Wunsch alignment / audit checks, not by the main
    training pipeline.
    """
    run([PYTHON, "src/export_split_fasta.py"], label="Export per-split alignment FASTAs")
 
 
def stage_train_fasta():
    """Regenerate per-split clean FASTAs (used by DePP / DepoScope retraining)."""
    outputs = {
        "train": "data/train_sequences_clean.fasta",
        "valid": "data/valid_sequences_clean.fasta",
        "test": "data/test_sequences_clean.fasta",
    }
    for split, out_path in outputs.items():
        run(
            [PYTHON, "src/make_train_fasta.py",
             "--merged-fasta", "data/merged_input_normalized.fasta",
             "--dataset-csv", "data/final_dataset_40.csv",
             "--output", out_path,
             "--split", split],
            label=f"make_train_fasta: {split}",
        )
 
 
def stage_embeddings():
    """Generate ESM-2 / ProtBERT / ProtT5 embeddings for the full dataset.
 
    NOTE: requires the small argparse patch to src/embedding/generate_embedding.py
    described alongside this script (--model-type flag). Without that patch
    this script only has one hardcoded model and can't be looped over here.
    """
    for model_type in EMBEDDINGS:
        run(
            [PYTHON, "src/embedding/generate_embedding.py", "--model-type", model_type],
            label=f"Embeddings: {model_type}",
        )
 
 
def stage_split_embeddings():
    run([PYTHON, "src/prepare_split_embeddings.py"], label="Split embeddings train/val/test")
 
 
def stage_train():
    """Train all 6 thesis models (2 architectures x 3 embeddings)."""
    for arch in ARCHS:
        script = "train_cnn.py" if arch == "cnn" else "train_dnn.py"
        for emb in EMBEDDINGS:
            run(
                [PYTHON, f"src/nn/{script}", "--embedding", emb],
                label=f"Train {arch.upper()} + {emb}",
            )
 
 
def stage_tune():
    """Optuna hyperparameter search for all 6 (arch, embedding) combos. Opt-in only."""
    for arch in ARCHS:
        for emb in EMBEDDINGS:
            run(
                [PYTHON, "src/nn/tune_hyperparams.py",
                 "--arch", arch, "--embedding", emb, "--n-trials", "25"],
                label=f"Tune {arch.upper()} + {emb}",
            )
 
 
def stage_external():
    """Retrain DepoScope and DePP on the current train split."""
    run([PYTHON, "src/deposcope_retrain.py"], label="Retrain DepoScope")
    run([PYTHON, "src/depp_retrain.py"], label="Retrain DePP")
 
 
def stage_benchmark():
    run([PYTHON, "src/benchmark_comparison.py"], label="Benchmark comparison (all 8 models)")
 
 
STAGES = {
    "combine_domain_fasta": stage_combine_domain_fasta,
    "remove_hypothetical": stage_remove_hypothetical,
    "dedup": stage_dedup,
    "balance_merge": stage_balance_merge,
    "normalize": stage_normalize,
    "export_alignment_fasta": stage_export_alignment_fasta,
    "train_fasta": stage_train_fasta,
    "embeddings": stage_embeddings,
    "split_embeddings": stage_split_embeddings,
    "train": stage_train,
    "tune": stage_tune,
    "external": stage_external,
    "benchmark": stage_benchmark,
}
 
# Order matters. CD-HIT and GraphPart run OUTSIDE this script (terminal),
# slotted in between the stages marked below.
STAGE_ORDER = [
    "combine_domain_fasta",
    "remove_hypothetical",
    "dedup",
    # <-- CD-HIT here (terminal, run on BOTH positive and negative dedup sets) -->
    "balance_merge",
    "normalize",
    # <-- GraphPart here (terminal) -->
    "export_alignment_fasta",
    "train_fasta",
    "embeddings",
    "split_embeddings",
    "train",
    "external",
    "benchmark",
]
# 'tune' is deliberately excluded from --all: it's exploratory and you'd
# want to review results and update src/nn/train_*.py defaults by hand
# before folding tuned hyperparameters back into the main training stage.
 
 
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage", nargs="+", choices=STAGES.keys(),
        help="Run one or more specific stages, in the order given.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run the full pipeline in STAGE_ORDER (excludes 'tune').",
    )
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    args = parser.parse_args()
 
    if args.list or not (args.stage or args.all):
        print("Full pipeline order (Python stages only):\n")
        for s in STAGE_ORDER:
            print(f"  - {s}")
        print("\n  - tune       (optional, opt-in, not part of --all)")
        print("\nTerminal-only steps that must run BETWEEN Python stages:")
        print("  - CD-HIT     (between 'dedup' and 'balance_merge'; run on BOTH pos + neg dedup sets)")
        print("  - GraphPart  (between 'normalize' and 'train_fasta')")
        print("\nRun `python run_pipeline.py --all` or `--stage <name> [<name> ...]`.")
        return
 
    t0 = time.time()
    stage_names = STAGE_ORDER if args.all else args.stage
    for name in stage_names:
        STAGES[name]()
 
    print(f"\nAll requested stages complete in {time.time() - t0:.1f}s")
 
 
if __name__ == "__main__":
    main()
 












