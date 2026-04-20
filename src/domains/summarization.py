import pandas as pd
from pathlib import Path
from typing import Union


INTERPRO_COLUMNS = [
    "protein_acc", "md5", "seq_len", "analysis", "signature_acc",
    "signature_desc", "start", "end", "score", "status", "date",
    "interpro_acc", "interpro_desc", "go_terms", "pathways"
]


def load_interpro_tsv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load InterProScan TSV output into a DataFrame.
    """
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=INTERPRO_COLUMNS,
        comment="#",
        dtype=str
    )


def summarize_pfam_domains(df: pd.DataFrame, total_proteins: int = None) -> pd.DataFrame:
    """
    Compute Pfam domain frequency (unique proteins per domain).
    """

    pfam = df[df["signature_acc"].fillna("").str.startswith("PF")]

    pfam_freq = (
        pfam.groupby(["signature_acc", "signature_desc"])["protein_acc"]
        .nunique()
        .reset_index(name="num_proteins")
        .sort_values("num_proteins", ascending=False)
        .reset_index(drop=True)
    )

    if total_proteins is None:
        total_proteins = df["protein_acc"].nunique()

    pfam_freq["percent_of_proteins"] = (
        pfam_freq["num_proteins"] / total_proteins * 100
    ).round(2)

    pfam_freq = pfam_freq.rename(columns={
        "signature_acc": "domain_acc",
        "signature_desc": "domain_desc"
    })

    return pfam_freq


def summarize_interpro_domains(df: pd.DataFrame, total_proteins: int = None) -> pd.DataFrame:
    """
    Compute InterPro domain frequency (unique proteins per domain).
    """

    ipr = df[df["interpro_acc"].fillna("").str.startswith("IPR")]

    ipr_freq = (
        ipr.groupby(["interpro_acc", "interpro_desc"])["protein_acc"]
        .nunique()
        .reset_index(name="num_proteins")
        .sort_values("num_proteins", ascending=False)
        .reset_index(drop=True)
    )

    if total_proteins is None:
        total_proteins = df["protein_acc"].nunique()

    ipr_freq["percent_of_proteins"] = (
        ipr_freq["num_proteins"] / total_proteins * 100
    ).round(2)

    ipr_freq = ipr_freq.rename(columns={
        "interpro_acc": "domain_acc",
        "interpro_desc": "domain_desc"
    })

    return ipr_freq

def proteins_without_pfam(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify proteins present in the dataset that do not have any Pfam annotation.
    """
    all_proteins = set(df["protein_acc"].dropna().unique())

    pfam_proteins = set(
        df.loc[df["signature_acc"].fillna("").str.startswith("PF"), "protein_acc"]
        .dropna()
        .unique()
    )

    missing = sorted(all_proteins - pfam_proteins)

    return pd.DataFrame({"protein_acc": missing})


def summarize_pfam_architecture(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Pfam domain architecture per protein, ordered by domain start position.
    """
    pfam = df[df["signature_acc"].fillna("").str.startswith("PF")].copy()
    pfam["start"] = pd.to_numeric(pfam["start"], errors="coerce")

    pfam = pfam.sort_values(["protein_acc", "start", "signature_acc"])

    arch = (
        pfam.groupby("protein_acc")["signature_acc"]
        .apply(lambda x: " | ".join(x.tolist()))
        .reset_index(name="pfam_architecture")
    )

    return arch


def summarize_pfam_architecture_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many proteins share the same Pfam architecture.
    """
    arch = summarize_pfam_architecture(df)

    arch_freq = (
        arch.groupby("pfam_architecture")["protein_acc"]
        .nunique()
        .reset_index(name="num_proteins")
        .sort_values("num_proteins", ascending=False)
        .reset_index(drop=True)
    )

    return arch_freq


def save_results(
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    total_proteins: int = None
) -> None:
    """
    Run all summaries and save output tables.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if total_proteins is None:
        total_proteins = df["protein_acc"].nunique()

    pfam_freq = summarize_pfam_domains(df, total_proteins=total_proteins)
    ipr_freq = summarize_interpro_domains(df, total_proteins=total_proteins)
    no_pfam = proteins_without_pfam(df)
    pfam_arch = summarize_pfam_architecture(df)
    pfam_arch_freq = summarize_pfam_architecture_frequency(df)

    pfam_freq.to_csv(out_dir / "pfam_domain_frequency.tsv", sep="\t", index=False)
    ipr_freq.to_csv(out_dir / "interpro_domain_frequency.tsv", sep="\t", index=False)
    no_pfam.to_csv(out_dir / "proteins_without_pfam.tsv", sep="\t", index=False)
    pfam_arch.to_csv(out_dir / "pfam_architecture_per_protein.tsv", sep="\t", index=False)
    pfam_arch_freq.to_csv(out_dir / "pfam_architecture_frequency.tsv", sep="\t", index=False)

    print(f"Total proteins: {total_proteins}")
    print(f"Proteins without Pfam: {len(no_pfam)}")
    print("Saved files:")
    print(out_dir / "pfam_domain_frequency.tsv")
    print(out_dir / "interpro_domain_frequency.tsv")
    print(out_dir / "proteins_without_pfam.tsv")
    print(out_dir / "pfam_architecture_per_protein.tsv")
    print(out_dir / "pfam_architecture_frequency.tsv")
