from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, List, Dict

import pandas as pd


def parse_domtblout(domtblout_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse an HMMER3 hmmscan/hmmsearch --domtblout file into a tidy pandas DataFrame.

    Notes
    -----
    - In domtblout format, the first 22 whitespace-separated fields are fixed.
      Everything after that corresponds to the target (domain) description.
    - This parser extracts per-domain hits with coordinates on the query (protein),
      which is typically what you want for downstream domain filtering and architecture building.

    Parameters
    ----------
    domtblout_path : str | Path
        Path to the HMMER --domtblout output file.

    Returns
    -------
    pd.DataFrame
        Columns:
        - protein_id
        - domain_name
        - domain_acc
        - independent_evalue
        - domain_score
        - ali_from
        - ali_to
        - description
    """
    domtblout_path = Path(domtblout_path)
    rows: List[Dict[str, object]] = []

    with domtblout_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            # Split into the 22 fixed fields + remainder (description)
            parts = line.rstrip("\n").split(maxsplit=22)

            # Need at least the 22 fixed fields (index 0..21). If fewer, skip.
            if len(parts) < 22:
                continue

            # If description is missing, set it to empty string
            if len(parts) == 22:
                parts.append("")

            (
                target_name,     # 0  target name
                target_acc,      # 1  target accession
                tlen,            # 2  target length
                query_name,      # 3  query name (your protein)
                query_acc,       # 4  query accession
                qlen,            # 5  query length
                full_evalue,     # 6
                full_score,      # 7
                full_bias,       # 8
                dom_num,         # 9
                dom_of,          # 10
                c_evalue,        # 11
                i_evalue,        # 12 independent E-value (domain-level)
                dom_score,       # 13
                dom_bias,        # 14
                hmm_from,        # 15
                hmm_to,          # 16
                ali_from,        # 17 alignment start on query
                ali_to,          # 18 alignment end on query
                env_from,        # 19
                env_to,          # 20
                acc,             # 21
                desc             # 22 description (free text)
            ) = parts

            # Cast cautiously; skip malformed lines rather than crashing the pipeline
            try:
                rows.append(
                    {
                        "protein_id": query_name,
                        "domain_name": target_name,
                        "domain_acc": target_acc,
                        "independent_evalue": float(i_evalue),
                        "domain_score": float(dom_score),
                        "ali_from": int(ali_from),
                        "ali_to": int(ali_to),
                        "description": desc.strip(),
                    }
                )
            except (ValueError, TypeError):
                # If any numeric conversion fails, ignore the line
                continue

    df = pd.DataFrame(rows)

    # Enforce a consistent column order (useful for downstream reproducibility)
    if not df.empty:
        df = df[
            [
                "protein_id",
                "domain_name",
                "domain_acc",
                "independent_evalue",
                "domain_score",
                "ali_from",
                "ali_to",
                "description",
            ]
        ]

    return df