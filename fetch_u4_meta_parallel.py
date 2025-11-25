from __future__ import annotations

"""
Parallel UniProtKB metadata fetch for U4.

Reads accessions from outputs_samples/U4_idmap_raw.parquet, fetches accession/length/lineage/sequence
via the UniProt search API in parallel (chunked queries to avoid URL size limits), and writes a TSV
out to outputs_samples/U4_metadata.tsv.

Usage:
    python fetch_u4_meta_parallel.py
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

IDMAP_PATH = Path("outputs_samples/U4_idmap_raw.parquet")
OUTPUT_PATH = Path("outputs_samples/U4_metadata.tsv")
CHUNK_SIZE = 80
MAX_WORKERS = 20
API_URL = "https://rest.uniprot.org/uniprotkb/search"


def fetch_chunk(ids: list[str]) -> str | None:
    # Use compact query: accession:(ID1 OR ID2 ...)
    id_query = " OR ".join(ids)
    query = f"accession:({id_query})"
    params = {
        "query": query,
        "fields": "accession,length,lineage,sequence",
        "format": "tsv",
        "size": 500,
    }
    for attempt in range(5):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                sleep_time = int(resp.headers.get("Retry-After", 5))
                time.sleep(sleep_time)
                continue
            resp.raise_for_status()
            return resp.text
        except Exception:
            time.sleep(1 * (attempt + 1))
    return None


def load_existing_entries() -> set[str]:
    """If a partial metadata file exists, load already-seen accessions so we can resume."""
    if not OUTPUT_PATH.exists():
        return set()
    print(f"[fetch-meta] Found existing metadata: {OUTPUT_PATH} (resume mode)")
    try:
        existing = pd.read_csv(OUTPUT_PATH, sep="\t", usecols=["Entry"])["Entry"].dropna().unique()
    except Exception:
        print("[fetch-meta] Warning: failed to read existing file, starting fresh.")
        return set()
    existing_set = set(existing.tolist())
    print(f"[fetch-meta] Existing entries: {len(existing_set)}")
    return existing_set


def main() -> None:
    print(f"[fetch-meta] Loading accessions from {IDMAP_PATH}...")
    idmap = pd.read_parquet(IDMAP_PATH)
    accessions = idmap["accession"].dropna().unique().tolist()
    print(f"[fetch-meta] Unique accessions: {len(accessions)}")

    existing_entries = load_existing_entries()
    if existing_entries:
        accessions = [a for a in accessions if a not in existing_entries]
        print(f"[fetch-meta] Remaining to fetch after resume filter: {len(accessions)}")
    else:
        print("[fetch-meta] No existing metadata found; full fetch.")

    if not accessions:
        print("[fetch-meta] Nothing to do; all accessions already fetched.")
        return

    chunks = [accessions[i : i + CHUNK_SIZE] for i in range(0, len(accessions), CHUNK_SIZE)]
    print(f"[fetch-meta] Split into {len(chunks)} chunks of {CHUNK_SIZE}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header_written = bool(existing_entries)
    mode = "a" if header_written else "w"
    with open(OUTPUT_PATH, mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="meta-download", unit="chunk"):
                data = future.result()
                if not data:
                    continue
                lines = data.strip().split("\n")
                if not lines:
                    continue
                if not header_written:
                    f_out.write(data)
                    header_written = True
                else:
                    if len(lines) > 1:
                        f_out.write("\n" + "\n".join(lines[1:]))
    print(f"[fetch-meta] Done. Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
