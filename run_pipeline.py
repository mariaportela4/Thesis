"""
run_pipeline.py
================
Single entry point for the full DPO classification pipeline, end to end,
in the canonical order.

Lives at the project root, next to src/ and data/, and drives the existing
scripts under src/ via subprocess -- it does not reimplement any of their
logic. All paths are resolved relative to this file's own directory, so the
project can be cloned anywhere.

Stages
------
The pipeline runs as a sequence of named stages (see STAGE_ORDER). Each one
shells out to a script under src/, or to an external command-line tool, and
stops immediately on a non-zero exit, so a broken run never silently continues
into the next stage. Every script path lives in the SCRIPT PATHS block below,
so a file that moves only needs editing in one place.

Two stages call external tools rather than project scripts: 'cdhit' runs
CD-HIT at 80% identity on both dedup sets, and 'graphpart' partitions the
normalized FASTA at 40% identity. 'tune' is excluded from --all because its
results have to be reviewed before being folded back into training.

Safeguards
----------
CD-HIT and GraphPart can both fail without a non-zero exit code, so they are
checked explicitly:

  * Binary preflight  -- cd-hit / graphpart / needleall must be on PATH.
  * EMBOSS preflight  -- EMBOSS_DATA and EMBOSS_ACDROOT must be set before
                         GraphPart runs. Unset, needleall produces no
                         alignments and the resulting partition is random
                         rather than identity-based.
  * Edge-count check  -- GraphPart's stdout is teed and parsed; a graph with
                         zero edges aborts the run instead of writing a
                         meaningless assignment file.

Usage (run from the project root)
---------------------------------
    python run_pipeline.py --list              # stages, order, and preflight status
    python run_pipeline.py --check             # preflight only, run nothing
    python run_pipeline.py --all               # full pipeline (excludes 'tune')
    python run_pipeline.py --all --dry-run     # print every command, execute nothing
    python run_pipeline.py --stage cdhit graphpart
    python run_pipeline.py --from balance_merge          # resume from a stage
    python run_pipeline.py --from embeddings --until train
    python run_pipeline.py --all --skip embeddings       # reuse existing .npy files
    python run_pipeline.py --stage tune                  # opt-in, not part of --all
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

EMBEDDINGS = ["esm", "bert", "prott5"]
ARCHS = ["dnn", "cnn"]

# ---------------------------------------------------------------------------
# CONFIGURATION -- every path and parameter used by the pipeline lives here.
# ---------------------------------------------------------------------------

# Sequence sources
POS_RAW = "data/positive/all_DPO_positive.fasta"
NEG_RAW = "data/negative/negative_nohyp.faa"
POS_DEDUP = "data/positive/all_DPO_positive_dedup.fasta"
NEG_DEDUP = "data/negative/negative_dedup.faa"

# CD-HIT (80% identity clustering, run on BOTH dedup sets)
CDHIT_IDENTITY = 0.8
POS_CDHIT = "results/cdhit/positive_cdhit80.fasta"
NEG_CDHIT = "results/cdhit/negative_cdhit80.fasta"

# Merge + normalisation
MERGED_FASTA = "data/merged_input.fasta"
MERGED_NORMALIZED = "data/merged_input_normalized.fasta"
MERGED_METADATA = "data/merged_metadata.csv"

# GraphPart (40% identity partitioning)
GRAPHPART_THRESHOLD = 0.4
GRAPHPART_TEST_RATIO = 0.1
GRAPHPART_VAL_RATIO = 0.05
GRAPHPART_ASSIGNMENTS = "results/graphpart/graphpart_assignments_40.csv"
FINAL_DATASET = "data/final_dataset_40.csv"

# Compute
THREADS = 12

# Optuna
N_TRIALS = 25

# ---------------------------------------------------------------------------
# SCRIPT PATHS -- every script this pipeline calls, in one place. If a file
# ever moves, this is the only block that needs editing.
# ---------------------------------------------------------------------------

# Dataset construction
COMBINE_DOMAIN_SCRIPT = "src/datasets/combine_domains.py"
REMOVE_HYPOTHETICAL_SCRIPT = "src/datasets/remove_hypothetical_proteins.py"
REMOVE_DUPLICATES_SCRIPT = "src/remove_duplicates.py"
BALANCE_MERGE_SCRIPT = "src/datasets/balance_and_merge.py"
NORMALIZE_SCRIPT = "src/normalize_ambiguous_aa.py"
BUILD_FINAL_DATASET_SCRIPT = "src/datasets/build_final_dataset.py"
EXPORT_SPLIT_FASTA_SCRIPT = "src/export_split_fasta.py"
MAKE_TRAIN_FASTA_SCRIPT = "src/make_train_fasta.py"

# Embeddings and models
GENERATE_EMBEDDING_SCRIPT = "src/embedding/generate_embedding.py"
PREPARE_SPLIT_EMBEDDINGS_SCRIPT = "src/prepare_split_embeddings.py"
TRAIN_SCRIPTS = {"dnn": "src/nn/train_dnn.py", "cnn": "src/nn/train_cnn.py"}
TUNE_SCRIPT = "src/nn/tune_hyperparams.py"

# DposFinder needs to be told where its cloned repository and output directory
# are; the other three resolve their own paths internally. The clone sits
# outside the project tree, so the path is relative to the project root.
DPOSFINDER_REPO = "../DposFinder"

# Retraining only produces a checkpoint -- DposFinder does not save
# per-sequence predictions during training, so its own predict mode has to be
# invoked afterwards to write test_set_result.tsv, which benchmark_comparison.py
# reads. The run name ties the two calls to the same checkpoint.
DPOSFINDER_RUN_NAME = "dposfinder_retrained"

# Interpreter used for DposFinder's own code/main.py, for BOTH final_train and
# predict. These must match: a checkpoint written under one torch version is
# not reliably loaded by another. DposFinder has its own conda environment
# (thesis_env lacks its dependencies, e.g. logomaker), so its interpreter is
# given explicitly rather than inherited from this pipeline.
DPOSFINDER_PYTHON = "/home/mcarvalho/miniconda3/envs/DposFinder/bin/python"
DPOSFINDER_OUT_DIR = "results/dposfinder_retrain"

# Retraining scripts for the four external baselines, with any arguments they
# require. All four live under src/; the DePP and DposFinder ones reach into
# their own cloned (gitignored) repositories at runtime.
EXTERNAL_RETRAIN_SCRIPTS = [
    ("DepoScope", "src/deposcope_retrain.py", []),
    ("DePP", "src/depp_retrain.py", []),
    ("PhageDPO", "src/phagedpo_retrain.py", []),
    ("DposFinder", "src/dposfinder_retrain.py", [
        "--dataset_csv", FINAL_DATASET,
        "--dposfinder_repo", DPOSFINDER_REPO,
        "--out_dir", DPOSFINDER_OUT_DIR,
        "--name", DPOSFINDER_RUN_NAME,
        "--python", DPOSFINDER_PYTHON,
    ]),
]

# Final evaluation
BENCHMARK_SCRIPT = "src/benchmark_comparison.py"

# Checked by --check / --list, so a missing or moved script is reported up
# front rather than halfway through a long run.
ALL_SCRIPTS = [
    COMBINE_DOMAIN_SCRIPT,
    REMOVE_HYPOTHETICAL_SCRIPT,
    REMOVE_DUPLICATES_SCRIPT,
    BALANCE_MERGE_SCRIPT,
    NORMALIZE_SCRIPT,
    BUILD_FINAL_DATASET_SCRIPT,
    EXPORT_SPLIT_FASTA_SCRIPT,
    MAKE_TRAIN_FASTA_SCRIPT,
    GENERATE_EMBEDDING_SCRIPT,
    PREPARE_SPLIT_EMBEDDINGS_SCRIPT,
    *TRAIN_SCRIPTS.values(),
    TUNE_SCRIPT,
    *[path for _, path, _ in EXTERNAL_RETRAIN_SCRIPTS],
    BENCHMARK_SCRIPT,
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

DRY_RUN = False


class StageError(RuntimeError):
    """Raised when a stage cannot run (missing binary, missing input, bad output)."""


def require_binary(name, hint=""):
    """Abort early if an external command-line tool is not on PATH."""
    if shutil.which(name) is None:
        msg = f"Required external tool '{name}' was not found on PATH."
        if hint:
            msg += f"\n{hint}"
        if DRY_RUN:
            print(f"[DRY-RUN WARNING] {msg}")
            return
        raise StageError(msg)


def require_file(rel_path, produced_by=""):
    """Abort early if a required input file is missing."""
    if not (ROOT / rel_path).exists():
        msg = f"Required input file not found: {rel_path}"
        if produced_by:
            msg += f"\nIt is produced by the '{produced_by}' stage -- run that first."
        raise StageError(msg)


def check_emboss_env():
    """
    GraphPart calls EMBOSS needleall. If EMBOSS_DATA / EMBOSS_ACDROOT are not
    set, needleall still exits 0 but produces no alignments, GraphPart builds a
    graph with zero edges, and the resulting partition is effectively random.
    This is the single most dangerous silent failure in the pipeline, so it is
    a hard error rather than a warning.
    """
    missing = [v for v in ("EMBOSS_DATA", "EMBOSS_ACDROOT") if not os.environ.get(v)]
    if missing:
        if DRY_RUN:
            print(f"[DRY-RUN WARNING] EMBOSS variable(s) not set: {', '.join(missing)}")
            return
        raise StageError(
            f"EMBOSS environment variable(s) not set: {', '.join(missing)}.\n"
            "GraphPart would run to completion but build a zero-edge graph, "
            "producing an invalid random split.\n"
            "They are normally set by a conda activation script -- reactivate "
            "the environment, or export them manually, before running this stage."
        )
    print(f"    EMBOSS_DATA    = {os.environ['EMBOSS_DATA']}")
    print(f"    EMBOSS_ACDROOT = {os.environ['EMBOSS_ACDROOT']}")


def cdhit_word_size(identity):
    """
    CD-HIT's -n (word size) must match the -c identity threshold; the wrong
    word size either crashes or silently degrades clustering. Values from the
    CD-HIT user guide.
    """
    if identity >= 0.7:
        return 5
    if identity >= 0.6:
        return 4
    if identity >= 0.5:
        return 3
    return 2


def run(cmd, cwd=ROOT, label=None, capture=False):
    """
    Run a subprocess, streaming its output live, and hard-stop on failure.

    When capture=True the output is also collected and returned, so a stage can
    inspect what the tool actually reported (used for GraphPart's edge count).
    """
    cmd = [str(c) for c in cmd]
    label = label or " ".join(cmd)
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}")
    print(f"$ {' '.join(cmd)}\n")

    if DRY_RUN:
        print("[DRY-RUN] not executed")
        return ""

    t0 = time.time()
    if capture:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        lines = []
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)
        proc.wait()
        returncode, output = proc.returncode, "".join(lines)
    else:
        returncode, output = subprocess.run(cmd, cwd=cwd).returncode, ""

    elapsed = time.time() - t0
    if returncode != 0:
        print(f"\n[FAILED] {label}  (exit code {returncode}, {elapsed:.1f}s)")
        sys.exit(returncode)
    print(f"\n[OK] {label}  ({elapsed:.1f}s)")
    return output


def run_py(script, *args, label=None):
    """Run one of the project's own scripts with the active interpreter."""
    return run([PYTHON, script, *args], label=label or f"python {script}")


# ---------------------------------------------------------------------------
# STAGES
# ---------------------------------------------------------------------------

def stage_combine_domain_fasta():
    """Combine per-domain positive FASTAs (domain_*.fasta) into one candidate FASTA."""
    run_py(COMBINE_DOMAIN_SCRIPT, label="Combine domain FASTAs")


def stage_remove_hypothetical():
    """Filter out 'hypothetical protein' entries from the raw candidate sets."""
    run_py(REMOVE_HYPOTHETICAL_SCRIPT, label="Remove hypothetical proteins")


def stage_dedup():
    """Remove exact-duplicate sequences (RefSeq > GenBank > PDB priority)."""
    run_py(REMOVE_DUPLICATES_SCRIPT, POS_RAW, POS_DEDUP, label="Dedup: positives")
    run_py(REMOVE_DUPLICATES_SCRIPT, NEG_RAW, NEG_DEDUP, label="Dedup: negatives")


def stage_cdhit():
    """
    Cluster both dedup sets at 80% identity with CD-HIT.

    Input and output paths come from the POS_DEDUP / NEG_DEDUP / POS_CDHIT /
    NEG_CDHIT constants, so the positive and negative runs always read what the
    dedup stage actually wrote. The -n word size is derived from the identity
    threshold, since a mismatched value degrades clustering silently.
    """
    require_binary(
        "cd-hit",
        hint="Install it with:  conda install -c bioconda cd-hit",
    )
    word_size = cdhit_word_size(CDHIT_IDENTITY)
    (ROOT / POS_CDHIT).parent.mkdir(parents=True, exist_ok=True)

    for name, src, dst, produced_by in [
        ("positives", POS_DEDUP, POS_CDHIT, "dedup"),
        ("negatives", NEG_DEDUP, NEG_CDHIT, "dedup"),
    ]:
        if not DRY_RUN:
            require_file(src, produced_by=produced_by)
        run(
            ["cd-hit",
             "-i", src,
             "-o", dst,
             "-c", CDHIT_IDENTITY,
             "-n", word_size,
             "-d", 0,
             "-T", THREADS,
             "-M", 0],
            label=f"CD-HIT {int(CDHIT_IDENTITY * 100)}%: {name}",
        )


def stage_balance_merge():
    """Subsample negatives to a 1:2 pos:neg ratio and merge into merged_input.fasta."""
    if not DRY_RUN:
        require_file(POS_CDHIT, produced_by="cdhit")
        require_file(NEG_CDHIT, produced_by="cdhit")
    run_py(BALANCE_MERGE_SCRIPT, label="Balance negatives + merge")


def stage_normalize():
    """Normalize ambiguous residues (B/Z/J/U/O -> X) on the merged FASTA."""
    run_py(
        NORMALIZE_SCRIPT,
        "--input", MERGED_FASTA,
        "--output", MERGED_NORMALIZED,
        label="Normalize ambiguous residues",
    )


def stage_graphpart():
    """
    Partition the normalized FASTA at 40% identity with GraphPart.

    Runs three preflight checks first (graphpart binary, needleall binary,
    EMBOSS env vars) and then validates the reported edge count, because all
    three failure modes are silent.
    """
    require_binary(
        "graphpart",
        hint="Install it with:  pip install graph-part",
    )
    require_binary(
        "needleall",
        hint="needleall ships with EMBOSS:  conda install -c bioconda emboss",
    )
    print("\nEMBOSS environment:")
    check_emboss_env()

    if not DRY_RUN:
        require_file(MERGED_NORMALIZED, produced_by="normalize")
    (ROOT / GRAPHPART_ASSIGNMENTS).parent.mkdir(parents=True, exist_ok=True)

    output = run(
        ["graphpart", "needle",
         "--fasta-file", MERGED_NORMALIZED,
         "--threshold", GRAPHPART_THRESHOLD,
         "--out-file", GRAPHPART_ASSIGNMENTS,
         "--labels-name", "label",
         "--test-ratio", GRAPHPART_TEST_RATIO,
         "--val-ratio", GRAPHPART_VAL_RATIO,
         "--threads", THREADS],
        label=f"GraphPart {int(GRAPHPART_THRESHOLD * 100)}% identity",
        capture=True,
    )

    if DRY_RUN:
        return

    # Validate the edge count. A zero-edge graph means needleall produced no
    # usable alignments and the partition is random, not identity-based.
    edge_line = next(
        (ln for ln in output.splitlines() if "nr. of edges" in ln.lower()), None
    )
    if edge_line is None:
        print(
            "\n[WARNING] Could not find the edge-count line in GraphPart's output. "
            "Inspect the log manually and confirm the graph is not empty before "
            "trusting the split."
        )
        return

    print(f"\n    GraphPart reported: {edge_line.strip()}")
    digits = "".join(ch if ch.isdigit() else " " for ch in edge_line).split()
    n_edges = int(digits[-1]) if digits else 0
    if n_edges == 0:
        raise StageError(
            "GraphPart built a graph with ZERO edges. needleall produced no "
            "alignments, so the resulting partition is random rather than "
            "identity-based.\n"
            "Check the EMBOSS installation and that EMBOSS_DATA / EMBOSS_ACDROOT "
            "point at a valid EMBOSS share directory, then re-run this stage.\n"
            f"The invalid assignment file at {GRAPHPART_ASSIGNMENTS} should be deleted."
        )
    print(f"    Edge count OK ({n_edges} edges) -- partition is identity-based.")


def stage_integrate_graphpart():
    """
    Merge the GraphPart assignments into the canonical dataset CSV.

    Maps GraphPart's numeric cluster IDs to train/test/valid and writes
    data/final_dataset_40.csv, the single source of truth for the split used by
    every stage downstream.
    """
    if not DRY_RUN:
        require_file(GRAPHPART_ASSIGNMENTS, produced_by="graphpart")
        require_file(MERGED_METADATA)
    run_py(
        BUILD_FINAL_DATASET_SCRIPT,
        "--metadata", MERGED_METADATA,
        "--assignments", GRAPHPART_ASSIGNMENTS,
        "--output", FINAL_DATASET,
        label="Integrate GraphPart assignments -> final_dataset_40.csv",
    )


def stage_export_alignment_fasta():
    """Write per-split FASTAs for Needleman-Wunsch alignment / audit checks."""
    run_py(EXPORT_SPLIT_FASTA_SCRIPT, label="Export per-split alignment FASTAs")


def stage_train_fasta():
    """Regenerate per-split clean FASTAs (used by DePP / DepoScope / PhageDPO retraining)."""
    outputs = {
        "train": "data/train_sequences_clean.fasta",
        "valid": "data/valid_sequences_clean.fasta",
        "test": "data/test_sequences_clean.fasta",
    }
    for split, out_path in outputs.items():
        run_py(
            MAKE_TRAIN_FASTA_SCRIPT,
            "--merged-fasta", MERGED_NORMALIZED,
            "--dataset-csv", FINAL_DATASET,
            "--output", out_path,
            "--split", split,
            label=f"make_train_fasta: {split}",
        )


def stage_embeddings():
    """Generate ESM-2 / ProtBERT / ProtT5 embeddings for the full dataset."""
    for model_type in EMBEDDINGS:
        run_py(
            GENERATE_EMBEDDING_SCRIPT,
            "--model-type", model_type,
            label=f"Embeddings: {model_type}",
        )


def stage_split_embeddings():
    """Split the full embedding arrays into train/val/test by the 'split' column."""
    run_py(PREPARE_SPLIT_EMBEDDINGS_SCRIPT, label="Split embeddings train/val/test")


def stage_train():
    """Train all 6 thesis models (2 architectures x 3 embeddings)."""
    for arch in ARCHS:
        for emb in EMBEDDINGS:
            run_py(
                TRAIN_SCRIPTS[arch],
                "--embedding", emb,
                label=f"Train {arch.upper()} + {emb}",
            )


def stage_tune():
    """Optuna hyperparameter search for all 6 combos. Opt-in only, not part of --all."""
    for arch in ARCHS:
        for emb in EMBEDDINGS:
            run_py(
                TUNE_SCRIPT,
                "--arch", arch,
                "--embedding", emb,
                "--n-trials", N_TRIALS,
                label=f"Tune {arch.upper()} + {emb}",
            )


def run_dposfinder_predict():
    """
    Run DposFinder's own predict mode against the retrained checkpoint.

    Retraining writes a checkpoint but no per-sequence predictions, so this
    second call is what produces test_set_result.tsv (label / prob / pred) in
    the output directory. benchmark_comparison.py fails without it.

    Runs from the DposFinder repository root, since main.py resolves its
    internal paths relative to its own working directory. --data_path is
    therefore passed as an absolute path.
    """
    repo = (ROOT / DPOSFINDER_REPO).resolve()
    data_path = (ROOT / DPOSFINDER_OUT_DIR).resolve()

    if not DRY_RUN and not repo.is_dir():
        raise StageError(
            f"DposFinder repository not found at: {repo}\n"
            "Update DPOSFINDER_REPO at the top of run_pipeline.py, or clone it:\n"
            "    git clone https://github.com/shenyanxiang/DposFinder.git"
        )

    run(
        [DPOSFINDER_PYTHON, "code/main.py",
         "--mode", "predict",
         "--data_path", f"{data_path}/",
         "--test_data", "test_set.fasta",
         "--name", DPOSFINDER_RUN_NAME],
        cwd=repo,
        label="DposFinder: predict on the test set",
    )

    if DRY_RUN:
        return

    tsv = data_path / "test_set_result.tsv"
    if not tsv.exists():
        raise StageError(
            f"DposFinder's predict mode finished but {tsv} was not written.\n"
            "benchmark_comparison.py reads that file. Check main.py's output "
            "above, and that --name matches the checkpoint produced by "
            f"retraining ('{DPOSFINDER_RUN_NAME}')."
        )
    print(f"    Wrote {tsv}")


def stage_external():
    """
    Retrain all four external baselines on the current train split.

    DposFinder takes an extra step: its predict mode has to be run against the
    retrained checkpoint to produce the per-sequence predictions the benchmark
    reads.
    """
    for tool, script, extra_args in EXTERNAL_RETRAIN_SCRIPTS:
        if not DRY_RUN and not (ROOT / script).exists():
            raise StageError(
                f"Retraining script for {tool} not found: {script}\n"
                "If the file has moved, update EXTERNAL_RETRAIN_SCRIPTS at the top "
                "of run_pipeline.py."
            )
        run_py(script, *extra_args, label=f"Retrain {tool}")

        if tool == "DposFinder":
            run_dposfinder_predict()


def stage_benchmark():
    """Evaluate all 10 models on the held-out test set."""
    run_py(BENCHMARK_SCRIPT, label="Benchmark comparison (all 10 models)")


STAGES = {
    "combine_domain_fasta": stage_combine_domain_fasta,
    "remove_hypothetical": stage_remove_hypothetical,
    "dedup": stage_dedup,
    "cdhit": stage_cdhit,
    "balance_merge": stage_balance_merge,
    "normalize": stage_normalize,
    "graphpart": stage_graphpart,
    "integrate_graphpart": stage_integrate_graphpart,
    "export_alignment_fasta": stage_export_alignment_fasta,
    "train_fasta": stage_train_fasta,
    "embeddings": stage_embeddings,
    "split_embeddings": stage_split_embeddings,
    "train": stage_train,
    "external": stage_external,
    "benchmark": stage_benchmark,
    # Opt-in only:
    "tune": stage_tune,
}

# Canonical pipeline order. Everything runs from this script now -- there are
# no terminal-only steps left.
STAGE_ORDER = [
    "combine_domain_fasta",
    "remove_hypothetical",
    "dedup",
    "cdhit",
    "balance_merge",
    "normalize",
    "graphpart",
    "integrate_graphpart",
    "export_alignment_fasta",
    "train_fasta",
    "embeddings",
    "split_embeddings",
    "train",
    "external",
    "benchmark",
]

# 'tune' is deliberately excluded from --all: it's exploratory, and the results
# have to be reviewed before being folded back into the training stage.
OPT_IN_STAGES = ["tune"]

# Stages that depend on external command-line tools.
EXTERNAL_TOOL_STAGES = {
    "cdhit": ["cd-hit"],
    "graphpart": ["graphpart", "needleall"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_stage_list():
    print("Full pipeline order:\n")
    for i, name in enumerate(STAGE_ORDER, 1):
        tools = EXTERNAL_TOOL_STAGES.get(name)
        suffix = f"   [external: {', '.join(tools)}]" if tools else ""
        doc = (STAGES[name].__doc__ or "").strip().splitlines()[0]
        print(f"  {i:2d}. {name:<22}{suffix}")
        print(f"      {doc}")
    print("\nOpt-in stages (not included in --all):")
    for name in OPT_IN_STAGES:
        doc = (STAGES[name].__doc__ or "").strip().splitlines()[0]
        print(f"      {name:<22} {doc}")
    print("\nRun `python run_pipeline.py --all` or `--stage <name> [<name> ...]`.")


def preflight():
    """Report the availability of every external dependency without running anything."""
    print("Preflight checks\n" + "-" * 70)
    ok = True

    for stage, tools in EXTERNAL_TOOL_STAGES.items():
        for tool in tools:
            path = shutil.which(tool)
            status = path if path else "NOT FOUND"
            marker = " " if path else "!"
            print(f"[{marker}] {tool:<16} ({stage:<10}) -> {status}")
            ok = ok and path is not None

    for var in ("EMBOSS_DATA", "EMBOSS_ACDROOT"):
        value = os.environ.get(var)
        marker = " " if value else "!"
        print(f"[{marker}] {var:<16} (graphpart) -> {value or 'NOT SET'}")
        ok = ok and value is not None

    dpos_python = Path(DPOSFINDER_PYTHON)
    marker = " " if dpos_python.exists() else "!"
    print(f"[{marker}] {'DposFinder py':<16} (external ) -> "
          f"{DPOSFINDER_PYTHON if dpos_python.exists() else 'NOT FOUND'}")
    ok = ok and dpos_python.exists()

    dpos_repo = (ROOT / DPOSFINDER_REPO).resolve()
    marker = " " if dpos_repo.is_dir() else "!"
    print(f"[{marker}] {'DposFinder repo':<16} (external ) -> "
          f"{dpos_repo if dpos_repo.is_dir() else 'NOT FOUND'}")
    ok = ok and dpos_repo.is_dir()

    print("-" * 70)
    missing_scripts = []
    for script in ALL_SCRIPTS:
        exists = (ROOT / script).exists()
        marker = " " if exists else "!"
        print(f"[{marker}] {script}")
        if not exists:
            missing_scripts.append(script)
    ok = ok and not missing_scripts

    print("-" * 70)
    print("All checks passed." if ok else "Some checks FAILED -- see the '!' lines above.")
    return ok


def resolve_stage_selection(args):
    """Turn the CLI arguments into an ordered list of stage names."""
    if args.stage:
        selected = args.stage
    elif args.all or args.from_stage or args.until:
        start = STAGE_ORDER.index(args.from_stage) if args.from_stage else 0
        end = STAGE_ORDER.index(args.until) + 1 if args.until else len(STAGE_ORDER)
        if start > end:
            sys.exit(f"--from '{args.from_stage}' comes after --until '{args.until}'.")
        selected = STAGE_ORDER[start:end]
    else:
        return []

    if args.skip:
        selected = [s for s in selected if s not in args.skip]
    return selected


def main():
    global DRY_RUN

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
    parser.add_argument(
        "--from", dest="from_stage", choices=STAGE_ORDER,
        help="Resume the pipeline from this stage onwards.",
    )
    parser.add_argument(
        "--until", choices=STAGE_ORDER,
        help="Stop after this stage.",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=STAGES.keys(), default=[],
        help="Exclude these stages from the selected range.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print every command that would run, without executing anything.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Run the preflight checks (external tools, EMBOSS env, scripts) and exit.",
    )
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    args = parser.parse_args()

    DRY_RUN = args.dry_run

    if args.check:
        sys.exit(0 if preflight() else 1)

    stage_names = resolve_stage_selection(args)
    if args.list or not stage_names:
        print_stage_list()
        print()
        preflight()
        return

    print("Stages to run:")
    for i, name in enumerate(stage_names, 1):
        print(f"  {i:2d}. {name}")
    if DRY_RUN:
        print("\n*** DRY RUN -- no command will be executed ***")

    t0 = time.time()
    for name in stage_names:
        try:
            STAGES[name]()
        except StageError as exc:
            print(f"\n{'=' * 70}")
            print(f"[ABORTED] stage '{name}':\n\n{exc}")
            print("=" * 70)
            sys.exit(1)

    print(f"\nAll requested stages complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()