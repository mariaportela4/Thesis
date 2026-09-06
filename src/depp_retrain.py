"""
depp_retrain.py
===============
Retrains DePP (Random Forest) on the thesis dataset (final_dataset_40.csv),
reusing the original feature extraction and training functions from the
DePP_CLI repository (utils/data_processing.py and
utils/depolymerase_prediction.py), to guarantee the feature vector is
EXACTLY the same one DePP itself uses.

DePP source
-----------
This script reuses DePP's own feature extraction and training functions, so the
DePP repository must be cloned locally. That clone is gitignored (it is
external code), and its location is passed via --depp-repo rather than being
implied by this file's own path.

Ambiguous residues
------------------
DePP's featurization assumes the 20 standard residues, so sequences containing
any other character are dropped before training (not substituted, as the thesis
models do). The filtering happens here and the counts are printed, so the
number removed can be verified on every run.

Usage (from the project root):
    python src/depp_retrain.py
    python src/depp_retrain.py --depp-repo ../Depolymerase-Predictor/DePP_CLI

Setup:
    The DePP repository must be cloned locally first -- see the setup section
    of the project README. It is intentionally not committed to this
    repository.

Notes:
- dataset-csv is final_dataset_40.csv, used only to look up each seq_id's
  'label' (positive/negative -> 1/0).
- FASTA headers look like 'seq000001|label=positive|orig_id=DAU34648.1'.
  Since there are no spaces, Biopython reads the whole header as
  record.name/record.id. The script extracts the real seq_id (the field
  before the first '|') to merge with final_dataset_40.csv. The 3 example
  IDs printed below confirm the expected format.
- Known DePP limitation (documented, not worked around): DePP's
  ProteinAnalysis.get_amino_acids_percent() silently ignores non-standard
  residues, which is why sequences containing them are removed outright
  rather than normalized to X as the thesis models do.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix,
)

# --- Biopython compatibility shim ---
# utils/data_processing.py (from the original DePP repo) calls
# ProteinAnalysis.get_amino_acids_percent(), which was removed in newer
# Biopython versions (>=1.81) in favour of the .amino_acids_percent
# attribute. That attribute also changed scale (0-100 instead of 0-1).
# We restore the old method's exact behaviour here instead of touching
# the cloned DePP source, so this keeps working regardless of which
# Biopython version is installed.
from Bio.SeqUtils.ProtParam import ProteinAnalysis  # noqa: E402

if not hasattr(ProteinAnalysis, 'get_amino_acids_percent'):
    def _get_amino_acids_percent_compat(self):
        return {aa: v / 100.0 for aa, v in self.amino_acids_percent.items()}
    ProteinAnalysis.get_amino_acids_percent = _get_amino_acids_percent_compat
    print("[i] Applied Biopython compatibility shim for ProteinAnalysis.get_amino_acids_percent()")

# Project root, derived from this file's location (src/depp_retrain.py).
ROOT = Path(__file__).resolve().parent.parent

# Standard 20 amino acids — sequences containing anything else are dropped,
# matching the PhageDPO retraining approach (consistent removal, not
# substitution).
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_depp_modules(depp_cli_dir):
    """
    Put the DePP_CLI clone on sys.path and import its feature-extraction and
    training functions. Imported here rather than at module level because the
    repository location is only known after the arguments are parsed.
    """
    depp_cli_dir = Path(depp_cli_dir)
    if not depp_cli_dir.is_absolute():
        depp_cli_dir = (ROOT / depp_cli_dir).resolve()

    if not (depp_cli_dir / "utils").is_dir():
        sys.exit(
            f"DePP_CLI not found at: {depp_cli_dir}\n"
            "This script reuses DePP's own feature extraction, so the DePP "
            "repository must be cloned locally. It is intentionally not part of "
            "this repository -- see the setup section of the README:\n\n"
            "    git clone https://github.com/DamianJM/Depolymerase-Predictor.git\n\n"
            "Then re-run, passing --depp-repo if the clone is not at the default "
            "location."
        )

    sys.path.insert(0, str(depp_cli_dir))
    try:
        from utils.data_processing import (
            read_fasta, calculate_protein_parameters, generate_dataframe, save_dataframe,
        )
        from utils.depolymerase_prediction import train_model, predict_depolymerases
    except ImportError as exc:
        sys.exit(
            f"Found {depp_cli_dir} but could not import DePP's modules: {exc}\n"
            "Check that the clone is complete and that its own dependencies are "
            "installed."
        )

    print(f"[+] Using DePP source from: {depp_cli_dir}")
    return SimpleNamespace(
        read_fasta=read_fasta,
        calculate_protein_parameters=calculate_protein_parameters,
        generate_dataframe=generate_dataframe,
        save_dataframe=save_dataframe,
        train_model=train_model,
        predict_depolymerases=predict_depolymerases,
    )


def filter_ambiguous(fasta_path, out_path):
    """
    Write a copy of fasta_path containing only sequences made up of the 20
    standard amino acids, and report how many were dropped.

    DePP's featurization cannot handle non-standard residues, so these
    sequences are removed rather than normalized.
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    kept = [r for r in records if set(str(r.seq).upper()) <= STANDARD_AA]
    n_dropped = len(records) - len(kept)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(kept, out_path, "fasta")

    print(f"[+] {Path(fasta_path).name}: {len(records)} sequences, "
          f"{n_dropped} dropped for non-standard residues, {len(kept)} kept")
    return out_path, n_dropped


def build_feature_csv(depp, fasta_path, dataset_csv_path, output_csv_path,
                      require_labels=True):
    """
    Reads a FASTA file, computes features using DePP's original functions,
    and (optionally) attaches the DP label from dataset_csv_path
    (final_dataset_40.csv). Returns the resulting DataFrame.
    """
    print(f"[+] Reading {fasta_path} ...")
    records = depp.read_fasta(str(fasta_path))
    print(f"    {len(records)} sequences read")

    print("[+] Computing protein parameters (this can take a few minutes)...")
    params = [depp.calculate_protein_parameters(r) for r in records]
    df = depp.generate_dataframe(params)

    # Quick sanity check on ID format — confirm it matches the CSV's seq_id
    print("[+] Example IDs extracted from the FASTA (record.name):")
    print(df['name'].head(3).tolist())

    df = df.rename(columns={'name': 'Name'})

    # FASTA headers look like 'seq000001|label=positive|orig_id=DAU34648.1'.
    # Since they contain no spaces, Biopython reads the whole header as
    # record.id/record.name. We extract the real seq_id (first field before
    # '|') so we can merge with the CSV.
    df['_seq_id'] = df['Name'].str.split('|').str[0]

    if require_labels:
        labels_df = pd.read_csv(dataset_csv_path)[['seq_id', 'label']]
        label_map = {
            'positive': 1,
            'negative': 0,
        }
        labels_df['DP'] = labels_df['label'].map(label_map)

        n_before = len(df)
        df = df.merge(labels_df[['seq_id', 'DP']], left_on='_seq_id', right_on='seq_id', how='left')

        n_missing = df['DP'].isna().sum()
        if n_missing > 0:
            print(f"[!] WARNING: {n_missing}/{n_before} sequences found no matching label "
                  f"in {dataset_csv_path}. Check the 'Name' vs 'seq_id' format.")
            print(df[df['DP'].isna()]['Name'].head(5).tolist())

        df = df.dropna(subset=['DP'])
        df['DP'] = df['DP'].astype(int)
        df = df.drop(columns=['seq_id'])

    # _seq_id was only needed for the merge — drop it before saving, so it
    # doesn't get treated as a "feature" by train_model (which only drops
    # ['Name', 'DP']).
    df = df.drop(columns=['_seq_id'])

    depp.save_dataframe(df, output_csv_path)
    print(f"[+] Saved: {output_csv_path} ({len(df)} rows, {df.shape[1]} columns)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Retrain DePP on the thesis dataset")
    parser.add_argument(
        '--depp-repo', default='Depolymerase-Predictor/DePP_CLI',
        help="Path to the cloned DePP_CLI folder. Relative paths are resolved "
             "against the project root. The repository is gitignored and must "
             "be cloned separately (see README).",
    )
    parser.add_argument('--train-fasta', default='data/train_sequences_clean.fasta')
    parser.add_argument('--test-fasta', default='data/test_sequences_clean.fasta')
    parser.add_argument('--dataset-csv', default='data/final_dataset_40.csv',
                        help='final_dataset_40.csv (for labels)')
    parser.add_argument('--output-dir', default='results/depp_retrain')
    parser.add_argument(
        '--keep-ambiguous', action='store_true',
        help="Skip the non-standard-residue filter. DePP's featurization will "
             "likely fail — for diagnostics only.",
    )
    args = parser.parse_args()

    depp = load_depp_modules(args.depp_repo)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 0. Drop sequences DePP cannot featurize ---
    train_fasta, test_fasta = args.train_fasta, args.test_fasta
    if not args.keep_ambiguous:
        print("\n[+] Filtering sequences with non-standard residues (DePP requirement)")
        train_fasta, n_train_dropped = filter_ambiguous(
            args.train_fasta, out_dir / 'depp_train_no_ambiguous.fasta'
        )
        test_fasta, n_test_dropped = filter_ambiguous(
            args.test_fasta, out_dir / 'depp_test_no_ambiguous.fasta'
        )
        if n_test_dropped:
            print(f"[!] WARNING: {n_test_dropped} TEST sequence(s) were dropped. The "
                  "test set is meant to be identical across all benchmarked tools, "
                  "and benchmark_comparison.py will fail on the missing predictions.")

    # --- 1. Build training set in DePP's format (features + DP) ---
    train_csv = out_dir / 'depp_training_set_custom.csv'
    build_feature_csv(depp, train_fasta, args.dataset_csv, train_csv, require_labels=True)

    # --- 2. Build the test dataframe (features, no DP — true labels kept separately for evaluation) ---
    test_features_csv = out_dir / 'depp_test_features.csv'
    test_df_with_labels = build_feature_csv(
        depp, test_fasta, args.dataset_csv, test_features_csv, require_labels=True
    )
    # keep the true labels aside, and drop DP before predicting
    y_true = test_df_with_labels['DP'].values
    test_df_for_predict = test_df_with_labels.drop(columns=['DP']).rename(columns={'Name': 'name'})

    # --- 3. Train the model (Random Forest) on the custom training set ---
    print("[+] Training DePP's Random Forest on your training set...")
    model = depp.train_model(str(train_csv))

    # --- 4. Predict on the test set ---
    print("[+] Predicting on the test set...")
    predictions = depp.predict_depolymerases(test_df_for_predict, model)
    predictions_path = out_dir / 'depp_retrained_predictions.csv'
    depp.save_dataframe(predictions, predictions_path)
    print(f"[+] Predictions saved to: {predictions_path}")

    # --- 5. Evaluate (F1, AUC, Precision, Recall, Accuracy) ---
    y_prob = predictions['Probability_DePol'].values
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
    }

    print("\n=== DePP retrained results ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix:")
    print(cm)

    metrics_path = out_dir / 'depp_retrain_metrics.csv'
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"\n[+] Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()