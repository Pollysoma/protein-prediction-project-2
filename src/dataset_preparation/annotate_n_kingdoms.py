from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_THROTTLE_SECONDS = 0.25  # keep <4 req/s to stay polite with UniProt
DEFAULT_TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"

# -----------------------------------------------------------------------------
# Imports & fallbacks (allow standalone execution)
# -----------------------------------------------------------------------------
try:
    from membership_splits.bins import assign_length_bin, make_length_bins
    from membership_splits.fasta import load_fasta
    from membership_splits.uniprot import UniprotIdMapper
except ImportError:
    print("Warning: 'membership_splits' library not found. Using internal fallbacks.")
    
    class UniprotIdMapper:
        def __init__(self, cache): pass
        def map_accessions(self, accs): return {}

    def load_fasta(p: Path) -> pd.DataFrame:
        ids, seqs, lens = [], [], []
        curr_id, curr_seq = None, []
        with open(p, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if curr_id:
                        ids.append(curr_id)
                        full_seq = "".join(curr_seq)
                        seqs.append(full_seq)
                        lens.append(len(full_seq))
                    curr_id = line.split()[0][1:]
                    curr_seq = []
                else:
                    curr_seq.append(line)
            if curr_id:
                ids.append(curr_id)
                full_seq = "".join(curr_seq)
                seqs.append(full_seq)
                lens.append(len(full_seq))
        return pd.DataFrame({"accession": ids, "sequence": seqs, "length": lens})

    def make_length_bins(l): return [], []
    def assign_length_bin(l, b): return 0

# -----------------------------------------------------------------------------
# API & classification logic
# -----------------------------------------------------------------------------

def get_session():
    """
    Creates a requests session with robust retry logic to handle
    API rate limits or transient network errors.
    """
    session = requests.Session()
    retry = Retry(
        total=5, 
        backoff_factor=0.5, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "ProteinClassifier/5.0 (archive_fallback_enabled)",
        "Accept": "application/json"
    })
    return session

def throttled_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """
    Respectful wrapper around session.get that enforces a small delay
    between calls so we do not hammer UniProt/NCBI endpoints.
    """
    if API_THROTTLE_SECONDS:
        time.sleep(API_THROTTLE_SECONDS)
    return session.get(url, **kwargs)

def classify_lineage_list(lineage_list: List[str]) -> str:
    """
    Determines Kingdom from a list of taxonomic names.
    Example: ['Bacteria', 'Actinomycetota', ...] -> 'Bacteria'
    """
    names = {n.lower() for n in lineage_list}
    
    # Priority Order
    if "metazoa" in names or "animalia" in names: return "Metazoa"
    if "viridiplantae" in names or "plantae" in names: return "Viridiplantae"
    if "fungi" in names: return "Fungi"
    if "bacteria" in names: return "Bacteria"
    if "archaea" in names: return "Archaea"
    if "viruses" in names: return "Viruses"
    
    # Broad Eukaryota check
    if "eukaryota" in names: return "Other Eukaryota"
    
    return "Unknown"

def fetch_taxonomy_lineage(
    session: requests.Session,
    taxid: int,
    cache: Dict[int, List[str]],
) -> List[str]:
    """
    Resolve a taxid to its lineage. Results are cached to avoid issuing
    hundreds of repeated taxonomy lookups.
    """
    if not taxid:
        return []
    if taxid in cache:
        return cache[taxid]

    url = f"https://rest.uniprot.org/taxonomy/{taxid}"
    try:
        resp = throttled_get(session, url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            raw_lineage = data.get("lineage", [])
            lineage: List[str] = []
            for node in raw_lineage:
                if isinstance(node, dict):
                    name = node.get("scientificName")
                else:
                    name = str(node)
                if name:
                    lineage.append(name)
            if "scientificName" in data:
                lineage.append(data["scientificName"])
            cache[taxid] = lineage
            return lineage
    except requests.RequestException:
        pass

    cache[taxid] = []
    return []

def fetch_uniparc_fallback(
    session: requests.Session,
    accession: str,
    lineage_cache: Dict[int, List[str]],
) -> Tuple[str, Optional[int]]:
    """
    Search UniParc for deleted/obsolete entries.
    Uses the common taxon metadata to map back into the taxonomy API.
    """
    url = "https://rest.uniprot.org/uniparc/search"
    params = {
        "query": accession,
        "size": 2,
    }

    try:
        resp = throttled_get(session, url, params=params, timeout=20)
        if resp.status_code == 404:
            return "Unknown", None
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"  [Archive Error] {accession}: {exc}")
        return "Unknown", None

    for entry in data.get("results", []):
        # Prefer explicit taxonomy IDs if provided.
        tax_sources = entry.get("commonTaxons", [])
        if not tax_sources:
            tax_entry = entry.get("organism", {})
            if tax_entry:
                tax_sources = [tax_entry]

        for tax_entry in tax_sources:
            taxid = (
                tax_entry.get("commonTaxonId")
                or tax_entry.get("taxonId")
            )
            lineage = fetch_taxonomy_lineage(session, taxid, lineage_cache)
            if not lineage and tax_entry.get("commonTaxon"):
                lineage = [tax_entry["commonTaxon"]]
            kingdom = classify_lineage_list(lineage)
            if kingdom != "Unknown":
                return kingdom, taxid

    return "Unknown", None

def resolve_kingdom_robust(
    session: requests.Session,
    accession: str,
    lineage_cache: Dict[int, List[str]],
    api_cache: Dict[str, Tuple[str, Optional[int]]],
) -> Tuple[str, Optional[int]]:
    """
    Main resolution strategy:
    1. Live UniProtKB (Fastest for active proteins)
    2. UniParc Archive (Fallback for deleted/merged proteins)
    """
    if accession in api_cache:
        return api_cache[accession]

    # 1. Try Active Database
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        resp = throttled_get(session, url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            organism = data.get("organism", {})
            taxid = organism.get("taxonId")
            lineage = organism.get("lineage", [])
            if "scientificName" in organism:
                lineage.insert(0, organism["scientificName"])
            
            kingdom = classify_lineage_list(lineage)
            if kingdom != "Unknown" or taxid:
                api_cache[accession] = (kingdom, taxid)
                return kingdom, taxid
    except requests.RequestException:
        # API error or timeout, move to fallback logic
        pass

    # 2. If we are here, the protein is likely Deleted or Merged. check Archive.
    kingdom, taxid = fetch_uniparc_fallback(session, accession, lineage_cache)
    api_cache[accession] = (kingdom, taxid)
    return kingdom, taxid

# -----------------------------------------------------------------------------
# LOCAL DATA LOADERS
# -----------------------------------------------------------------------------

def load_taxdump_minimal(tar_path: Path) -> Tuple[Dict, Dict]:
    """
    Loads a minimal subset of the NCBI Taxdump (nodes + names).
    The tarball can be retrieved from DEFAULT_TAXDUMP_URL.
    """
    if not tar_path.exists():
        print(
            f"Warning: Taxdump not found at {tar_path}. "
            f"Download it from {DEFAULT_TAXDUMP_URL} or run the pipeline downloader."
        )
        return {}, {}
    import tarfile
    taxa = {}
    names = {}
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            f = tar.extractfile("nodes.dmp")
            if f:
                for line in f:
                    p = line.decode("utf-8").split("|")
                    taxa[int(p[0])] = (int(p[1]), p[2].strip())
            f = tar.extractfile("names.dmp")
            if f:
                for line in f:
                    p = line.decode("utf-8").split("|")
                    if p[3].strip() == "scientific name":
                        names[int(p[0])] = p[1].strip()
    except Exception:
        pass
    return taxa, names

def kingdom_from_local(tid: int, taxa: dict, names: dict) -> str:
    if tid not in taxa: return "Unknown"
    lineage = []
    curr = tid
    while True:
        if curr in names: lineage.append(names[curr])
        parent, rank = taxa.get(curr, (1, "no_rank"))
        if parent == curr or parent == 1: break
        curr = parent
    return classify_lineage_list(lineage)

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def annotate(args: argparse.Namespace) -> pd.DataFrame:
    print(f"Loading FASTA: {args.n_fasta}")
    n_df = load_fasta(args.n_fasta)

    if "length" not in n_df:
        n_df["length"] = n_df["sequence"].str.len()

    bins, _ = make_length_bins(n_df["length"].tolist()) if len(n_df) else ([], [])
    if len(bins):
        n_df["length_bin"] = n_df["length"].apply(lambda l: assign_length_bin(l, bins))
    else:
        n_df["length_bin"] = 0
    
    print("Loading local Taxonomy (if available)...")
    taxa, names = load_taxdump_minimal(args.taxdump)
    mapper = UniprotIdMapper(args.uniprot_cache)
    mapped = mapper.map_accessions(n_df["accession"].tolist())
    
    session = get_session()
    lineage_cache: Dict[int, List[str]] = {}
    api_cache: Dict[str, Tuple[str, Optional[int]]] = {}
    
    kingdoms = []
    ur50_ids = []
    taxids = []
    unknown_accessions: List[str] = []
    
    total = len(n_df)
    print(f"Processing {total} sequences (with archive fallback & throttled API access)...")
    
    for idx, acc in enumerate(n_df["accession"]):
        if idx % 25 == 0 or idx == total - 1:
            print(f"  Progress: {idx}/{total}...", end="\r")

        curr_kingdom = "Unknown"
        curr_taxid = None
        curr_ur50 = None
        
        # 1. Try Local Mapper (Fastest)
        entry = mapped.get(acc)
        if entry:
            curr_taxid = entry.get("taxon_id")
            curr_ur50 = entry.get("ur50_id") or entry.get("uniref50")
        
        # 2. Try Local Taxonomy Classification
        if curr_taxid and taxa:
            curr_kingdom = kingdom_from_local(curr_taxid, taxa, names)
            
        # 3. If still Unknown, use Robust API (Live + Archive)
        if curr_kingdom == "Unknown":
            api_kingdom, api_taxid = resolve_kingdom_robust(
                session,
                acc,
                lineage_cache=lineage_cache,
                api_cache=api_cache,
            )
            
            if api_kingdom != "Unknown":
                curr_kingdom = api_kingdom
            if api_taxid:
                curr_taxid = api_taxid
        
        if curr_kingdom == "Unknown":
            unknown_accessions.append(acc)
                
        kingdoms.append(curr_kingdom)
        taxids.append(curr_taxid)
        ur50_ids.append(curr_ur50)

    n_df["ur50_id"] = ur50_ids
    n_df["taxid"] = taxids
    n_df["kingdom"] = kingdoms
    
    # Output Saving
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            n_df.to_parquet(args.output, index=False)
        except ImportError:
            n_df.to_csv(str(args.output).replace(".parquet", ".csv"), index=False)

    print("\n" + "="*30)
    print("FINAL KINGDOM COUNTS")
    print("="*30)
    print(n_df["kingdom"].value_counts())
    
    # Report missing if any
    missing_count = (n_df["kingdom"] == "Unknown").sum()
    if missing_count > 0:
        print(f"WARNING: {missing_count} still Unknown. (Likely UniParc-only or invalid IDs)")
        print(f"  Accessions: {unknown_accessions}")
    
    if args.report:
        Path(args.report).write_text(json.dumps({
            "kingdom_counts": n_df["kingdom"].value_counts().to_dict(),
            "unknown_accessions": unknown_accessions,
            "taxdump_path": str(args.taxdump),
            "taxdump_source": DEFAULT_TAXDUMP_URL,
        }, indent=2))
    
    return n_df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fasta", type=Path, default=Path("newly_discovered_dissimilar_proteins_2024.fasta"))
    parser.add_argument("--taxdump", type=Path, default=Path("data/raw/taxdump.tar.gz"))
    parser.add_argument("--uniprot-cache", type=Path, default=Path("data/cache/uniprot_idmap.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/annotated_N.parquet"))
    parser.add_argument("--report", type=Path, default=Path("outputs/annotated_N_summary.json"))
    return parser.parse_args()

if __name__ == "__main__":
    annotate(parse_args())
