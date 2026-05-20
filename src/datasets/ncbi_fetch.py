import os, certifi, time
from Bio import Entrez

# ---------- SSL FIX ----------
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ---------- ENTREZ CONFIG ----------
Entrez.email = "pocawaves@gmail.com"
Entrez.tool = "phage_dpo_negative_dataset"

# ---------- PARAMETERS ----------
BATCH = 400
CAP = 30000
SLEEP = 0.34

# ---------- NEGATIVE QUERY ----------
query = r'''
(viruses[porgn:txid2731619] AND srcdb_refseq[PROP]) AND (250:1250[SLEN])
NOT (lyase*[Title] OR pectate*[Title] OR pectin*[Title] OR depolymerase*[Title]
OR glycanase*[Title] OR endoglycosidase*[Title] OR alginate*[Title]
OR rhamnosidase*[Title] OR hyaluronidase*[Title] OR hyaluronate*[Title]
OR eps-degrading*[Title] OR levanase*[Title] OR dextranase*[Title]
OR xylanase*[Title] OR "sgnh hydrolase"[Title] OR "gdsl hydrolase"[Title])
NOT (pfam12708[All Fields] OR cd20481[All Fields] OR pfam12219[All Fields]
OR cl22684[All Fields] OR pfam12217[All Fields] OR pfam13472[All Fields])
'''.strip()

# ---------- OUTPUT PATHS ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
out_dir = os.path.join(ROOT, "data", "negative")
os.makedirs(out_dir, exist_ok=True)

ids_path = os.path.join(out_dir, "negative_ids.txt")
fasta_path = os.path.join(out_dir, "negative.faa")

# ---------- 1) SEARCH ----------
h = Entrez.esearch(db="protein", term=query, retmax=0, usehistory="y")
r = Entrez.read(h)
h.close()

count = int(r["Count"])
webenv = r["WebEnv"]
query_key = r["QueryKey"]

to_get = min(count, CAP)

print(f"[NEGATIVE] Total hits available: {count}")
print(f"[NEGATIVE] Will retrieve up to: {to_get}")

# ---------- 2) GET IDS ----------
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

with open(ids_path, "w") as f:
    f.write("\n".join(ids))

print(f"[NEGATIVE] Saved IDs: {ids_path}")

# ---------- 3) FETCH FASTA ----------
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