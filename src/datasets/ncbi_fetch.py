import os, certifi, time
from Bio import Entrez

"""
Retrieves a negative dataset of viral protein sequences from NCBI's RefSeq
protein database using Biopython's Entrez interface.

A "negative" sequence, in this context, is a phage protein that is explicitly
NOT a depolymerase (DPO) — i.e. it lacks known depolymerase-related keywords
in its title and has no associated conserved domain hits linked to
depolymerase families.

Output
------
data/negative/negative_ids.txt  — one NCBI protein UID per line
data/negative/negative.faa      — corresponding sequences in FASTA format
"""

# ---------- SSL FIX ----------
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ---------- ENTREZ CONFIG ----------
Entrez.email = "pocawaves@gmail.com"
Entrez.tool = "phage_dpo_negative_dataset"

# ---------- PARAMETERS ----------
BATCH = 400
CAP = 30000 # Maximum total sequences to retrieve in one run.
SLEEP = 0.34  # Seconds to wait between consecutive API calls.

# ---------- NEGATIVE QUERY ----------

# Searches for viral proteins (taxon 2731619 = Viruses) from RefSeq with 
# sequence lengths between 400 and 1250 amino acids, then excludes any
# sequence whose title or conserved-domain annotation suggests depolymerase
# activity (lyases, glycanases, hydrolases, etc.).

query = r'''
(viruses[porgn:txid2731619] AND srcdb_refseq[PROP]) AND (400:1250[SLEN])
NOT (lyase*[Title] OR pectate*[Title] OR pectin*[Title] OR depolymerase*[Title]
OR glycanase*[Title] OR endoglycosidase*[Title] OR alginate*[Title]
OR rhamnosidase*[Title] OR hyaluronidase*[Title] OR hyaluronate*[Title]
OR eps-degrading*[Title] OR levanase*[Title] OR dextranase*[Title]
OR xylanase*[Title] OR "sgnh hydrolase"[Title] OR "gdsl hydrolase"[Title])
NOT (pfam12708[All Fields] OR cd20481[All Fields] OR pfam12219[All Fields]
OR cl22684[All Fields] OR pfam12217[All Fields] OR pfam13472[All Fields])
'''.strip()

# ---------- OUTPUT PATHS ----------

# Resolve the project root two levels above this script, then place outputs
# under data/negative/ (created automatically if it doesn't exist).

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
out_dir = os.path.join(ROOT, "data", "negative")
os.makedirs(out_dir, exist_ok=True)

ids_path = os.path.join(out_dir, "negative_ids.txt")
fasta_path = os.path.join(out_dir, "negative.faa")

# ---------- 1) SEARCH ----------

# Run an initial esearch with retmax=0 just to obtain the total hit count and
# a server-side WebEnv/QueryKey handle; no records are downloaded yet.

h = Entrez.esearch(db="protein", term=query, retmax=0, usehistory="y")
r = Entrez.read(h)
h.close()

count = int(r["Count"])  # Total sequences matching the query on NCBI
webenv = r["WebEnv"]     # Server-side session token for subsequent calls
query_key = r["QueryKey"] # Key that identifies this query in the session

to_get = min(count, CAP)   # Respect the CAP ceiling

print(f"[NEGATIVE] Total hits available: {count}")
print(f"[NEGATIVE] Will retrieve up to: {to_get}")

# ---------- 2) GET IDS ----------

# Page through the result set in batches of BATCH, collecting NCBI protein
# UIDs. Using the cached WebEnv/QueryKey avoids re-running the full query
# on every request, which is faster and friendlier to NCBI's servers.

ids = []

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

    ids.extend(rr["IdList"])
    time.sleep(SLEEP)

# Remove duplicates (preserve order)
seen = set()
ids = [x for x in ids if not (x in seen or seen.add(x))]

print(f"[NEGATIVE] Retrieved IDs: {len(ids)}")

# Persist the UID list so the FASTA step can be re-run independently.
with open(ids_path, "w") as f:
    f.write("\n".join(ids))

print(f"[NEGATIVE] Saved IDs: {ids_path}")

# ---------- 3) FETCH FASTA ----------

# Stream the protein sequences in FASTA format, appending each batch
# directly to the output file to avoid holding everything in memory.

with open(fasta_path, "w") as out:
    for start in range(0, len(ids), BATCH):
        h = Entrez.efetch(
            db="protein",
            rettype="fasta",
            retmode="text",
            retstart=start,
            retmax=min(BATCH, len(ids) - start),
            query_key=query_key,
            WebEnv=webenv,
        )
        out.write(h.read())
        h.close()
        time.sleep(SLEEP)

print(f"[NEGATIVE] Saved FASTA: {fasta_path}")