"""
phagedpo_retrain.py
====================
Retrains the original PhageDPO model on your dataset, using PhageDPO's own
feature set and model family, for a fair third external-tool benchmark
(alongside DepoScope and DePP).

Source: https://github.com/mfernandasvieira/phagedpo
Paper:  Vieira et al., "PhageDPO: A machine-learning based computational
        framework for identifying phage depolymerases", Computers in Biology
        and Medicine (2025). https://doi.org/10.1016/j.compbiomed.2025.109864

Architecture (from PhageDPO source — calculate_features.py, ml.py, parameters.py):
  - Features (protein-sequence-derived, computed per residue string):
      * AA_Len                          - sequence length
      * 20 raw amino acid counts        - via a manual 20-letter tally
      * Aromaticity, IsoelectricPoint   - Biopython ProteinAnalysis
      * Helix / Turn / Sheet fractions  - Biopython secondary_structure_fraction()
      * CTD (147 features)              - propy.CTD.CalculateCTD
      * Dipeptide composition (400)     - propy.AAComposition.CalculateDipeptideComposition
    Total: 573 features.
  - Model: scikit-learn Pipeline(StandardScaler -> SelectKBest(f_classif) -> SVC),
    hyperparameters chosen by GridSearchCV (cv=10) optimizing Matthews
    correlation coefficient — this is PhageDPO's own selection procedure
    (see ml.py / parameters.py, `param_shallow()['svm']`), not an
    architecture we chose for it.

Known deviation from the original tool (documented, not "fixed"):
  PhageDPO's original feature set also includes 5 ORF/DNA-derived features
  (per-base counts + %GC) computed from the *nucleotide* coding sequence.
  Our dataset is protein-only (as for DePP/DepoScope), so those 5 features
  are dropped here (573 of the original 578 features remain). This is the
  same category of limitation already documented for DePP's amino-acid-only
  featurization — noted here rather than worked around.

Ambiguous amino acids (B/Z/J/U/O/X):
  Both Biopython's ProteinAnalysis and propy's CTD/AAComposition descriptors
  assume the 20 standard residues and raise/behave incorrectly on anything
  else. Consistent with the DePP retraining approach, sequences containing
  any non-standard residue are removed entirely before feature extraction
  (not substituted or worked around).

Pipeline:
  1. Load train/test splits from final_dataset_40.csv
  2. Drop sequences with any non-standard amino acid (report how many/where)
  3. Compute the 573 PhageDPO features per sequence (cached to disk)
  4. GridSearchCV (cv=10, scoring=MCC) over PhageDPO's own SVM param grid
  5. Refit best pipeline on the full train split
  6. Evaluate on test split -> F1, Precision, Recall, AUC, Accuracy
  7. Save model + metrics to results/phagedpo_retrain/

Usage:
    python phagedpo_retrain.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:
    from propy import CTD, AAComposition
except ImportError as e:
    raise ImportError(
        "propy is required for PhageDPO's CTD / dipeptide-composition features.\n"
        "Install it with:  pip install propy3\n"
        "(propy3 is the Python-3-compatible fork; it is imported as `propy`.)"
    ) from e

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, matthews_corrcoef, make_scorer, classification_report,
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path("/home/mcarvalho/Thesis")
DATA_CSV = BASE_DIR / "data" / "final_dataset_40.csv"
OUT_DIR  = BASE_DIR / "results" / "phagedpo_retrain"

FEATURE_CACHE = OUT_DIR / "phagedpo_features.pkl"   # cached feature DataFrame

RANDOM_STATE = 42
CV_FOLDS     = 10          # matches PhageDPO's own cross-validation setup
THRESHOLD    = 0.5

# Standard 20 amino acids — anything else (B, Z, J, U, O, X, ...) is dropped,
# matching the DePP retraining approach (consistent removal, not substitution).
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1 — FEATURE EXTRACTION (from PhageDPO's calculate_features.py)
# ─────────────────────────────────────────────────────────────

AA_LETTERS = ['G', 'A', 'L', 'V', 'I', 'P', 'F', 'S', 'T', 'C',
              'Y', 'N', 'Q', 'D', 'E', 'R', 'K', 'H', 'W', 'M']


def is_clean_sequence(seq: str) -> bool:
    """True if every residue is one of the 20 standard amino acids."""
    return bool(seq) and set(seq.upper()) <= STANDARD_AA


def count_aa(seq: str) -> dict:
    """Raw amino acid counts (not percentages) — matches PhageDPO's count_aa()."""
    dic = {aa: 0 for aa in AA_LETTERS}
    for ch in seq:
        if ch in dic:
            dic[ch] += 1
    return dic


def sec_st_fraction(seq: str) -> dict:
    helix, turn, sheet = ProteinAnalysis(seq).secondary_structure_fraction()
    return {"Helix": helix, "Turn": turn, "Sheet": sheet}


def compute_phagedpo_features(seq: str) -> dict:
    """
    Compute PhageDPO's full protein-derived feature set for one sequence.
    Mirrors calculate_features.py's add_len_aa / add_aa_count / add_aromaticity /
    add_isoelectric_point / add_secondary_structure_fraction /
    add_composition_transition_distribution / add_di_peptide_composition.
    (ORF/DNA-GC features are excluded — see module docstring.)
    """
    pa = ProteinAnalysis(seq)
    feats = {"AA_Len": len(seq)}
    feats.update(count_aa(seq))
    feats["Aromaticity"] = pa.aromaticity()
    feats["IsoelectricPoint"] = pa.isoelectric_point()
    feats.update(sec_st_fraction(seq))
    feats.update(CTD.CalculateCTD(seq))
    feats.update(AAComposition.CalculateDipeptideComposition(seq))
    return feats


def build_feature_frame(df: pd.DataFrame, id_col: str, seq_col: str = "sequence") -> pd.DataFrame:
    """
    Filters to clean (standard-AA-only) sequences, computes PhageDPO features
    for each, and returns a DataFrame indexed by id_col with a 'label' column
    (1 = positive/DPO, 0 = negative) plus all feature columns.
    """
    n_total = len(df)
    clean = df[df[seq_col].apply(is_clean_sequence)].reset_index(drop=True)
    n_dropped = n_total - len(clean)
    if n_dropped:
        print(f"  Dropped {n_dropped}/{n_total} sequences with non-standard residues "
              f"(B/Z/J/U/O/X) — PhageDPO's feature set cannot be computed for these.")

    rows = []
    for _, row in clean.iterrows():
        feats = compute_phagedpo_features(row[seq_col])
        feats[id_col] = row[id_col]
        feats["label"] = 1 if row["label"] == "positive" else 0
        rows.append(feats)

    out = pd.DataFrame(rows).set_index(id_col)
    return out


def get_or_compute_features(df: pd.DataFrame, id_col: str = "orig_id") -> dict:
    """Load cached train/test feature frames, or compute and cache them."""
    if FEATURE_CACHE.exists():
        print(f"Loading cached PhageDPO features from {FEATURE_CACHE}")
        with open(FEATURE_CACHE, "rb") as f:
            return pickle.load(f)

    print("Computing PhageDPO features (573 per sequence) — this runs once and is cached...")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    print("Train split:")
    train_feats = build_feature_frame(train_df, id_col)
    print("Test split:")
    test_feats = build_feature_frame(test_df, id_col)

    cache = {"train": train_feats, "test": test_feats}
    with open(FEATURE_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Features cached to {FEATURE_CACHE}")
    return cache


# ─────────────────────────────────────────────────────────────
# STEP 2 — MODEL (PhageDPO's SVM pipeline, from ml.py / parameters.py)
# ─────────────────────────────────────────────────────────────

def build_param_grid(n_features: int) -> list:
    """
    PhageDPO's own SVM grid (parameters.py, param_shallow()['svm']['param_grid']),
    with selector__k adapted to our feature count. The original used
    [578, 320, 298, 120] where 578 included the 5 DNA/ORF features we don't
    have; we substitute the true feature-set size (n_features) for 578.
    """
    k_options = sorted({n_features, 320, 298, 120}, reverse=True)
    k_options = [k for k in k_options if k <= n_features]
    return [
        {"clf__C": [0.01, 0.1, 1.0, 10],
         "clf__kernel": ["linear"],
         "selector__k": k_options},
        {"clf__C": [0.01, 0.1, 1.0, 10],
         "clf__kernel": ["rbf"],
         "clf__gamma": ["scale", 0.001, 0.0001],
         "selector__k": k_options},
    ]


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = THRESHOLD) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "F1":        f1_score(y_true, preds,        zero_division=0),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall":    recall_score(y_true, preds,    zero_division=0),
        "AUC":       roc_auc_score(y_true, probs),
        "Accuracy":  accuracy_score(y_true, preds),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ─────────────────────────────────────────
    print(f"Loading dataset from {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"  Total sequences: {len(df)}")
    print(f"  Train: {(df['split']=='train').sum()}, Test: {(df['split']=='test').sum()}")

    # ── 2. Compute / load PhageDPO features ──────────────────
    feats = get_or_compute_features(df)
    train_feats, test_feats = feats["train"], feats["test"]

    feature_cols = [c for c in train_feats.columns if c != "label"]
    X_train, y_train = train_feats[feature_cols].values, train_feats["label"].values
    X_test,  y_test  = test_feats[feature_cols].values,  test_feats["label"].values
    print(f"\nFeature matrix: {X_train.shape[1]} features "
          f"({X_train.shape[0]} train / {X_test.shape[0]} test sequences)")

    # ── 3. Build pipeline + PhageDPO's own grid search ───────
    pipe = Pipeline([
        ("scl", StandardScaler()),
        ("selector", SelectKBest(f_classif)),
        ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
    ])
    param_grid = build_param_grid(X_train.shape[1])

    print(f"\nRunning GridSearchCV (cv={CV_FOLDS}, scoring=MCC) over PhageDPO's SVM grid...")
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=make_scorer(matthews_corrcoef),
        cv=CV_FOLDS,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    print(f"\nBest CV MCC: {gs.best_score_:.4f}")
    print(f"Best params: {gs.best_params_}")

    best_model = gs.best_estimator_

    # ── 4. Evaluate on test set ──────────────────────────────
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_probs)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS — PhageDPO (retrained SVM)")
    print("=" * 50)
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("=" * 50)

    print("\nClassification Report:")
    preds = (test_probs >= THRESHOLD).astype(int)
    print(classification_report(y_test, preds, target_names=["Non-DPO", "DPO"]))

    # ── 5. Save outputs ───────────────────────────────────────
    model_path = OUT_DIR / "phagedpo_retrained_svm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    metrics_path = OUT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    cv_results_path = OUT_DIR / "cv_results.csv"
    pd.DataFrame(gs.cv_results_).to_csv(cv_results_path, index=False)

    best_params_path = OUT_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump({"best_cv_mcc": gs.best_score_, **gs.best_params_}, f, indent=2, default=str)

    probs_path = OUT_DIR / "test_probabilities.npy"
    np.save(probs_path, test_probs)
    # ID-keyed predictions CSV — mirrors depp_retrained_predictions.csv, so
    # benchmark_comparison.py can merge by orig_id instead of relying on
    # positional order.
    predictions_path = OUT_DIR / "phagedpo_retrained_predictions.csv"
    pd.DataFrame({
        "orig_id": test_feats.index,
        "Probability_DPO": test_probs,
    }).to_csv(predictions_path, index=False)

    print(f"\nOutputs saved to {OUT_DIR}/")
    print(f"  Model:        {model_path.name}")
    print(f"  Metrics:      {metrics_path.name}")
    print(f"  CV results:   {cv_results_path.name}")
    print(f"  Best params:  {best_params_path.name}")
    print(f"  Probs (.npy): {probs_path.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()