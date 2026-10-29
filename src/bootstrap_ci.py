"""
Bootstrap confidence intervals for the DPO benchmark.

Resamples the 176-sequence test set with replacement and recomputes
F1, precision and recall for every model, yielding percentile 95% CIs.

The SAME resample indices are applied to every model in each replicate,
so the intervals are paired and the per-replicate differences between
two models are meaningful (see --compare).

No retraining required: this reads the stored test-set predictions.

Usage
-----
    python bootstrap_ci.py --preds results/benchmark_predictions.csv
    python bootstrap_ci.py --preds <file> --compare dnn_esm dposfinder_retrained

Expected input: one row per test sequence, with
  - an identifier column (seq_id)
  - a true-label column (label), 1/0 or positive/negative
  - one column per model holding the predicted PROBABILITY of the
    positive class (preferred), or the binary prediction.

Long format (seq_id, model, label, prob) is also accepted and reshaped.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG - adjust if your column names differ
# --------------------------------------------------------------------------
ID_COL = "seq_id"
LABEL_COL = "label"
THRESHOLD = 0.5          # same threshold used in Section 5.6
N_BOOT = 10000           # 1000 is enough; 10000 stabilises the percentiles
SEED = 42
# Columns that are metadata, not model predictions
NON_MODEL_COLS = {
    ID_COL, LABEL_COL, "orig_id", "product", "organism",
    "full_description", "sequence", "AC", "priority", "label-val",
    "cluster", "split",
}
# Display order for the output table (models not listed are appended)
MODEL_ORDER = [
    "dnn_prott5", "dnn_esm", "dposfinder_retrained", "cnn_prott5",
    "cnn_esm", "dnn_bert", "phagedpo_retrained", "depp_retrained",
    "deposcope_retrained", "cnn_bert",
]
# --------------------------------------------------------------------------


def load_predictions(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (y_true, DataFrame of per-model probabilities)."""
    df = pd.read_csv(path)

    # Long format -> wide
    if {"model"}.issubset(df.columns):
        value_col = next(
            (c for c in ("prob", "probability", "y_prob", "score", "pred")
             if c in df.columns),
            None,
        )
        if value_col is None:
            sys.exit("Long format detected but no prob/score column found.")
        labels = df.groupby(ID_COL)[LABEL_COL].first()
        df = df.pivot(index=ID_COL, columns="model", values=value_col)
        df[LABEL_COL] = labels
        df = df.reset_index()

    missing = {ID_COL, LABEL_COL} - set(df.columns)
    if missing:
        sys.exit(f"Missing expected column(s): {sorted(missing)}. "
                 f"Found: {list(df.columns)}")

    y = df[LABEL_COL]
    if y.dtype == object:
        y = y.astype(str).str.lower().map(
            {"positive": 1, "pos": 1, "1": 1, "true": 1,
             "negative": 0, "neg": 0, "0": 0, "false": 0}
        )
        if y.isna().any():
            sys.exit(f"Could not interpret labels: "
                     f"{df.loc[y.isna(), LABEL_COL].unique()[:5]}")
    y_true = y.to_numpy(dtype=int)

    model_cols = [c for c in df.columns if c not in NON_MODEL_COLS]
    model_cols = [c for c in model_cols
                  if pd.api.types.is_numeric_dtype(df[c])]
    if not model_cols:
        sys.exit(f"No numeric model columns found. Columns: {list(df.columns)}")

    return y_true, df[model_cols]


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1 from binary vectors. NaN where undefined."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    if tp == 0 or np.isnan(precision) or np.isnan(recall):
        f1 = 0.0 if (tp + fp + fn) > 0 else np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, type=Path,
                    help="CSV of test-set predictions")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                    help="report the paired F1 difference A - B")
    ap.add_argument("--out", type=Path, default=Path("results/bootstrap_ci.csv"))
    ap.add_argument("--latex", type=Path, default=Path("results/bootstrap_ci.tex"))
    args = ap.parse_args()

    y_true, probs = load_predictions(args.preds)
    n = len(y_true)
    models = list(probs.columns)

    print(f"Test set: {n} sequences "
          f"({int(y_true.sum())} positive, {int((1 - y_true).sum())} negative)")
    print(f"Models:   {len(models)} -> {', '.join(models)}")
    print(f"Bootstrap: {args.n_boot} replicates, seed {args.seed}, "
          f"threshold {args.threshold}\n")

    # Binarise once; resampling then just reindexes.
    binary = {m: (probs[m].to_numpy() >= args.threshold).astype(int)
              for m in models}

    # Point estimates on the full test set
    point = {m: prf(y_true, binary[m]) for m in models}

    # Paired resampling: one index draw per replicate, shared by all models
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_boot, n))

    boot = {m: np.full((args.n_boot, 3), np.nan) for m in models}
    skipped = 0
    for b in range(args.n_boot):
        take = idx[b]
        yt = y_true[take]
        if yt.sum() == 0:          # no positives -> recall undefined
            skipped += 1
            continue
        for m in models:
            boot[m][b] = prf(yt, binary[m][take])

    if skipped:
        print(f"Note: {skipped} replicate(s) contained no positive "
              f"sequence and were dropped.\n")

    rows = []
    for m in models:
        arr = boot[m]
        row = {"model": m}
        for j, name in enumerate(("precision", "recall", "f1")):
            vals = arr[:, j]
            vals = vals[~np.isnan(vals)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            row[name] = point[m][j]
            row[f"{name}_lo"] = lo
            row[f"{name}_hi"] = hi
            row[f"{name}_sd"] = vals.std(ddof=1)
        rows.append(row)

    res = pd.DataFrame(rows)
    order = [m for m in MODEL_ORDER if m in models]
    order += [m for m in models if m not in order]
    res = res.set_index("model").loc[order].reset_index()

    pd.set_option("display.width", 200)
    print(res[["model", "f1", "f1_lo", "f1_hi",
               "precision", "precision_lo", "precision_hi",
               "recall", "recall_lo", "recall_hi"]].round(4).to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    # LaTeX column for Table 12: "F1 (95% CI)"
    lines = [
        r"% F1 with percentile bootstrap 95% CI - paste into Table 12",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & F1 (95\% CI) & Precision (95\% CI) & Recall (95\% CI) \\",
        r"\midrule",
    ]
    for _, r in res.iterrows():
        name = r["model"].replace("_", r"\_")
        cells = []
        for metric in ("f1", "precision", "recall"):
            cells.append(f"{r[metric]:.4f} "
                         f"[{r[metric + '_lo']:.3f}, {r[metric + '_hi']:.3f}]")
        lines.append(f"\\texttt{{{name}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.latex.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.latex}")

    # Paired difference, if requested
    if args.compare:
        a, b = args.compare
        for m in (a, b):
            if m not in models:
                sys.exit(f"--compare: '{m}' not among {models}")
        diff = boot[a][:, 2] - boot[b][:, 2]
        diff = diff[~np.isnan(diff)]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p_worse = float(np.mean(diff <= 0))
        print(f"\nPaired F1 difference, {a} - {b}:")
        print(f"  point estimate : {point[a][2] - point[b][2]:+.4f}")
        print(f"  95% CI         : [{lo:+.4f}, {hi:+.4f}]")
        print(f"  P(diff <= 0)   : {p_worse:.3f}")
        if lo <= 0 <= hi:
            print("  -> interval includes zero: the two are not "
                  "distinguishable on this test set.")


if __name__ == "__main__":
    main()