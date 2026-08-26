import os, certifi, time, random
import numpy as np
from Bio import Entrez, SeqIO

"""
Retrieves a negative dataset of viral protein sequences from NCBI's RefSeq
protein database using Biopython's Entrez interface.

A "negative" sequence, in this context, is a phage protein that is explicitly
NOT a depolymerase (DPO) — i.e. it lacks known depolymerase-related keywords
in its title and has no associated conserved domain hits linked to
depolymerase families.

CHANGE vs. the original version: instead of pulling every hit that matches
the query (~16k-30k sequences, heavily left-skewed toward the 400-500 aa
range — see distribution plot in 01_domain_analysis.ipynb), this version:

  1. Fetches only lengths first (via esummary, cheap) for the full candidate
     pool.
  2. Builds a length histogram of the POSITIVE set as the reference shape.
  3. Samples negatives per length-bin so the final negative length
     distribution matches the positive one, capped at NEG_LIMIT total.
  4. Only then fetches full FASTA records for the selected subset.

This directly targets the imbalance flagged in the README
("ncbi_fetch.py needs adjustments to produce a better-balanced
positive/negative dataset upstream of CD-HIT") and additionally removes the
length-distribution mismatch visible in the domain-analysis notebook, which
would otherwise still bias the CD-HIT-clustered negative pool that class
balancing (step 5) samples from later.

Output
------
data/negative/negative_ids.txt  — one NCBI protein UID per line (selected subset)
data/negative/negative.faa      — corresponding sequences in FASTA format
"""

# ---------- SSL FIX ----------
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ---------- ENTREZ CONFIG ----------
Entrez.email = "pocawaves@gmail.com"
Entrez.tool = "phage_dpo_negative_dataset"

# ---------- PATHS ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
out_dir = os.path.join(ROOT, "data", "negative")
os.makedirs(out_dir, exist_ok=True)

ids_path = os.path.join(out_dir, "negative_ids.txt")
fasta_path = os.path.join(out_dir, "negative.faa")

# Reference positive set used only to compute the target length distribution.
# UPDATE this path if your raw positive candidates fasta lives elsewhere
# (this follows the naming used in the README: data/positive/all_DPO_candidates.fasta).
POSITIVE_FASTA = os.path.join(ROOT, "data", "positive", "all_DPO_candidates.fasta")

# ---------- PARAMETERS ----------
BATCH = 400
SEARCH_CAP = 30000   # ceiling for the initial candidate pool pulled from NCBI
NEG_LIMIT = 3000      # final number of negatives to keep, length-matched to positives
                       # (e.g. ~3000 gives headroom for CD-HIT attrition while
                       # staying far below the old ~16-30k raw pull; tune as needed)
BIN_WIDTH = 25         # amino acids per length bin used for distribution matching
SLEEP = 0.34           # seconds between consecutive API calls
SEED = 42              # same seed used elsewhere in the pipeline (class balancing)

random.seed(SEED)

# ---------- NEGATIVE QUERY ----------

# Searches for viral proteins (taxon 2731619 = Viruses) from RefSeq, then
# excludes any sequence whose title or conserved-domain annotation suggests
# depolymerase activity (lyases, glycanases, hydrolases, etc.). No length
# filter here anymore — length-matching to the positive distribution
# (step 5 below) already constrains which sequences get selected, so a
# fixed SLEN range would only shrink the candidate pool unnecessarily.

query = r'''
(viruses[porgn:txid2731619] AND srcdb_refseq[PROP])
NOT (lyase*[Title] OR pectate*[Title] OR pectin*[Title] OR depolymerase*[Title]
OR glycanase*[Title] OR endoglycosidase*[Title] OR alginate*[Title]
OR rhamnosidase*[Title] OR hyaluronidase*[Title] OR hyaluronate*[Title]
OR eps-degrading*[Title] OR levanase*[Title] OR dextranase*[Title]
OR xylanase*[Title] OR "sgnh hydrolase"[Title] OR "gdsl hydrolase"[Title])
NOT (pfam12708[All Fields] OR cd20481[All Fields] OR pfam12219[All Fields]
OR cl22684[All Fields] OR pfam12217[All Fields] OR pfam13472[All Fields])
'''.strip()

# ---------- 1) SEARCH ----------

# Run an initial esearch with retmax=0 just to obtain the total hit count and
# a server-side WebEnv/QueryKey handle; no records are downloaded yet.

h = Entrez.esearch(db="protein", term=query, retmax=0, usehistory="y")
r = Entrez.read(h)
h.close()

count = int(r["Count"])  # Total sequences matching the query on NCBI
webenv = r["WebEnv"]     # Server-side session token for subsequent calls
query_key = r["QueryKey"] # Key that identifies this query in the session

to_get = min(count, SEARCH_CAP)   # Respect the candidate-pool ceiling

print(f"[NEGATIVE] Total hits available: {count}")
print(f"[NEGATIVE] Candidate pool to pull IDs for: {to_get}")

# ---------- 2) GET CANDIDATE IDS ----------

# Page through the result set in batches of BATCH, collecting NCBI protein
# UIDs. Using the cached WebEnv/QueryKey avoids re-running the full query
# on every request, which is faster and friendlier to NCBI's servers.

candidate_ids = []

for start in range(0, to_get, BATCH):
    h = Entrez.esearch(
        db="protein",
        term=query,
        retstart=start,
        retmax=min(BATCH, to_get - start),
        usehistory="y",
        webenv=webenv,
        query_key=query_key
    )
    rr = Entrez.read(h)
    h.close()

    candidate_ids.extend(rr["IdList"])
    time.sleep(SLEEP)

# Remove duplicates (preserve order)
seen = set()
candidate_ids = [x for x in candidate_ids if not (x in seen or seen.add(x))]

print(f"[NEGATIVE] Candidate IDs retrieved: {len(candidate_ids)}")

# ---------- 3) GET CANDIDATE LENGTHS (esummary, cheap) ----------

# esummary returns docsum metadata (including sequence Length) without
# downloading the full sequence, which is much cheaper than efetch for
# scanning the whole candidate pool.

id_to_length = {}

for start in range(0, len(candidate_ids), BATCH):
    batch = candidate_ids[start:start + BATCH]
    h = Entrez.esummary(db="protein", id=",".join(batch))
    docsums = Entrez.read(h)
    h.close()

    for ds in docsums:
        uid = str(ds.get("Id") or ds.get("Caption") or "")
        length = ds.get("Length")
        if uid and length is not None:
            id_to_length[uid] = int(length)

    time.sleep(SLEEP)

print(f"[NEGATIVE] Lengths retrieved: {len(id_to_length)} / {len(candidate_ids)}")

# ---------- 4) REFERENCE (POSITIVE) LENGTH DISTRIBUTION ----------

if not os.path.exists(POSITIVE_FASTA):
    raise FileNotFoundError(
        f"Positive reference fasta not found at {POSITIVE_FASTA}. "
        "Set POSITIVE_FASTA at the top of this script to the correct path "
        "before running the length-matching step."
    )

pos_lengths = [len(rec.seq) for rec in SeqIO.parse(POSITIVE_FASTA, "fasta")]
print(f"[NEGATIVE] Positive reference sequences: {len(pos_lengths)}")

# ---------- 5) LENGTH-MATCHED, CAPPED SAMPLING ----------


# Bin range is now derived from the positive lengths themselves (rounded
# out to the nearest BIN_WIDTH), since the query no longer restricts
# candidates to a fixed SLEN range. Any candidate outside this range is
# simply never selected (falls outside all bins).
bin_lo = int(np.floor(min(pos_lengths) / BIN_WIDTH) * BIN_WIDTH)
bin_hi = int(np.ceil(max(pos_lengths) / BIN_WIDTH) * BIN_WIDTH)
bin_edges = np.arange(bin_lo, bin_hi + BIN_WIDTH, BIN_WIDTH)
pos_counts, _ = np.histogram(pos_lengths, bins=bin_edges)
pos_proportions = pos_counts / pos_counts.sum()

# Group candidate negative IDs (with a known, in-range length) by bin.
neg_by_bin = {i: [] for i in range(len(bin_edges) - 1)}
for uid, length in id_to_length.items():
    if length < bin_edges[0] or length >= bin_edges[-1]:
        continue
    b = np.digitize(length, bin_edges) - 1
    b = min(b, len(bin_edges) - 2)
    neg_by_bin[b].append(uid)

for b in neg_by_bin:
    random.shuffle(neg_by_bin[b])

# Initial per-bin target, proportional to the positive distribution.
target_counts = np.round(pos_proportions * NEG_LIMIT).astype(int)

selected = []
remaining_target = {}
for b, target in enumerate(target_counts):
    available = neg_by_bin[b]
    take = min(target, len(available))
    selected.extend(available[:take])
    neg_by_bin[b] = available[take:]  # leftover pool for redistribution
    deficit = target - take
    if deficit > 0:
        remaining_target[b] = deficit

# Redistribute any shortfall (bins where positives outnumber available
# negatives) proportionally across bins that still have spare candidates.
deficit_total = sum(remaining_target.values())
if deficit_total > 0:
    spare_bins = [b for b, ids in neg_by_bin.items() if ids]
    spare_total = sum(len(neg_by_bin[b]) for b in spare_bins)
    if spare_total > 0:
        for b in spare_bins:
            share = int(round(deficit_total * (len(neg_by_bin[b]) / spare_total)))
            take = min(share, len(neg_by_bin[b]))
            selected.extend(neg_by_bin[b][:take])
            neg_by_bin[b] = neg_by_bin[b][take:]

# Final cap, in case rounding pushed the total slightly over NEG_LIMIT.
random.shuffle(selected)
selected = selected[:NEG_LIMIT]

sel_lengths = [id_to_length[u] for u in selected]
print(f"[NEGATIVE] Selected (length-matched, capped): {len(selected)}")
print(f"[NEGATIVE] Positive length mean/median/std:  "
      f"{np.mean(pos_lengths):.1f} / {np.median(pos_lengths):.1f} / {np.std(pos_lengths):.1f}")
print(f"[NEGATIVE] Selected negative mean/median/std: "
      f"{np.mean(sel_lengths):.1f} / {np.median(sel_lengths):.1f} / {np.std(sel_lengths):.1f}")

try:
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(pos_lengths, sel_lengths)
    print(f"[NEGATIVE] KS test positive vs. selected negative lengths: "
          f"D={ks_stat:.4f}, p={ks_p:.4f}")
except ImportError:
    pass

ids = selected

# Persist the selected UID list so the FASTA step can be re-run independently.
with open(ids_path, "w") as f:
    f.write("\n".join(ids))

print(f"[NEGATIVE] Saved IDs: {ids_path}")

# ---------- 6) FETCH FASTA (selected subset only) ----------

# Stream the protein sequences in FASTA format, appending each batch
# directly to the output file to avoid holding everything in memory.
# Note: unlike step 1-3 (which used the cached WebEnv/QueryKey for the full
# query), this fetches by explicit UID list since we're only pulling the
# length-matched subset, not the whole result set.

with open(fasta_path, "w") as out:
    for start in range(0, len(ids), BATCH):
        batch = ids[start:start + BATCH]
        h = Entrez.efetch(
            db="protein",
            rettype="fasta",
            retmode="text",
            id=",".join(batch),
        )
        out.write(h.read())
        h.close()
        time.sleep(SLEEP)

print(f"[NEGATIVE] Saved FASTA: {fasta_path}")