import pandas as pd


def build_pfam_interpro_master(pfam_freq: pd.DataFrame, interpro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a master table linking Pfam domains with InterPro annotations.

    Parameters
    ----------
    pfam_freq : pd.DataFrame
        Output of summarize_pfam_domains().
    interpro_df : pd.DataFrame
        Full InterProScan dataframe.

    Returns
    -------
    pd.DataFrame
        Master table with Pfam frequency and InterPro mapping.
    """

    # Unique domain signatures from InterProScan
    signatures = interpro_df.drop_duplicates(
        subset=["signature_acc"]
    )[["signature_acc", "interpro_acc", "interpro_desc"]]

    pfam_master = (
        pfam_freq.merge(
            signatures,
            on="signature_acc",
            how="left"
        )
        .rename(columns={
            "signature_acc": "pfam_acc",
            "signature_desc": "pfam_desc"
        })
        .reset_index(drop=True)
    )

    return pfam_master