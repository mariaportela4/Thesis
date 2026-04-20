import pandas as pd


def keep_best_domain_hits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the best hit per protein and per domain accession.
    Primary: lowest independent_evalue
    Secondary: highest domain_score (if present)
    """
    sort_cols = ["independent_evalue"]
    ascending = [True]

    if "domain_score" in df.columns:
        sort_cols.append("domain_score")
        ascending.append(False)

    df = (
        df.sort_values(sort_cols, ascending=ascending)
          .drop_duplicates(subset=["protein_id", "domain_acc"], keep="first")
          .reset_index(drop=True)
    )
    return df