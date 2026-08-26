"""
Remove duplicate (100% identical) sequences from the positive DPO dataset.
 
Selection criterion for the representative of each duplicate group:
    1. RefSeq (YP_, WP_, NP_, ...) > GenBank (other prefixes) > PDB
    2. Within the same category, keep the header with the more complete
       description (more characters after the ID)
    3. PDB entries (pdb|XXXX|CHAIN) are handled separately: only 1
       representative chain per structure is kept (the alphabetically
       lowest chain ID, e.g. chain A), since they represent the same
       protein replicated within the crystallographic asymmetric unit.
"""

import sys
import re
from collections import defaultdict
from Bio import SeqIO

refseq_prefixes = ('YP_','WP_','NP_','XP_','AP_')

def classify_id(rec_id : str) -> str:
    """
    Classify the ID as 'pdb','refseq', or 'genbank'.
    """

    if rec_id.startswith('pdb|'):
        return 'pdb'
    if rec_id.startswith(refseq_prefixes):
        return 'refseq'
    return 'genbank'

def pdb_structure_and_chain(rec_id: str):
    """Extract (structure, chain) from an ID like 'pdb|6E0V|A'."""
    parts = rec_id.split("|")
    # possible formats: pdb|6E0V|A  or  pdb|6E0V|A,B
    structure = parts[1] if len(parts) > 1 else rec_id
    chain = parts[2] if len(parts) > 2 else ""
    return structure, chain

def choose_representative(records):
    """Choose the representative record for a group of identical sequences."""
    priority = {"refseq": 0, "genbank": 1, "pdb": 2}
 
    def sort_key(rec):
        category = classify_id(rec.id)
        # lower priority number = better; longer description = better (more negative)
        desc_len = len(rec.description) - len(rec.id)
        return (priority[category], -desc_len)
 
    return sorted(records, key=sort_key)[0]

def main(input_fasta: str, output_fasta: str):
    all_records = list(SeqIO.parse(input_fasta, "fasta"))
    total_in = len(all_records)
 
    # 1. Separate PDB chains: keep only 1 chain per structure
    pdb_by_structure = defaultdict(list)
    non_pdb_records = []
 
    for rec in all_records:
        if classify_id(rec.id) == "pdb":
            structure, chain = pdb_structure_and_chain(rec.id)
            pdb_by_structure[structure].append((chain, rec))
        else:
            non_pdb_records.append(rec)
 
    pdb_representatives = []
    for structure, chain_recs in pdb_by_structure.items():
        chain_recs.sort(key=lambda x: x[0])  # sort by chain (A, B, C...)
        pdb_representatives.append(chain_recs[0][1])
 
    candidates = non_pdb_records + pdb_representatives
 
    # 2. Group by sequence (exact dedup) and choose a representative
    seq_to_records = defaultdict(list)
    for rec in candidates:
        seq_to_records[str(rec.seq)].append(rec)
 
    final_records = []
    dedup_log = []
    for seq, recs in seq_to_records.items():
        rep = choose_representative(recs)
        final_records.append(rep)
        if len(recs) > 1:
            other_ids = [r.id for r in recs if r.id != rep.id]
            dedup_log.append((rep.id, other_ids))
 
    SeqIO.write(final_records, output_fasta, "fasta")
 
    # 3. Report
    print(f"Total sequences in input:        {total_in}")
    print(f"PDB chains collapsed:            {sum(len(v) for v in pdb_by_structure.values()) - len(pdb_by_structure)}")
    print(f"Final unique sequences:          {len(final_records)}")
    print(f"Exact duplicate groups:          {len(dedup_log)}")
    print(f"\nSaved to: {output_fasta}")
 
    if dedup_log:
        print("\nExamples of collapsed groups (kept representative -> removed):")
        for rep_id, removed in dedup_log[:10]:
            print(f"  {rep_id}  ->  {removed}")
        if len(dedup_log) > 10:
            print(f"  ... and {len(dedup_log) - 10} more groups")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_duplicates.py input.fasta output.fasta")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
    