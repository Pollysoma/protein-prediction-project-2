from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def classify_lineage_list(lineage_list: List[str]) -> str:
    """
    Determines kingdom from a list of taxonomic names.
    Example: ['Bacteria', 'Actinomycetota', ...] -> 'Bacteria'
    """
    names = {n.lower() for n in lineage_list}

    if "metazoa" in names or "animalia" in names:
        return "Metazoa"
    if "viridiplantae" in names or "plantae" in names:
        return "Viridiplantae"
    if "fungi" in names:
        return "Fungi"
    if "bacteria" in names:
        return "Bacteria"
    if "archaea" in names:
        return "Archaea"
    if "viruses" in names:
        return "Viruses"
    if "eukaryota" in names:
        return "Other Eukaryota"
    return "Unknown"


def load_taxdump_minimal(tar_path: Path) -> Tuple[Dict[int, Tuple[int, str]], Dict[int, str]]:
    """
    Load the minimal subset of NCBI taxdump needed for kingdom classification.
    Returns (taxa, names) dictionaries.
    """
    if not tar_path.exists():
        print(
            f"Warning: Taxdump not found at {tar_path}. "
            "Download it from https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz."
        )
        return {}, {}

    import tarfile

    taxa: Dict[int, Tuple[int, str]] = {}
    names: Dict[int, str] = {}
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            nodes = tar.extractfile("nodes.dmp")
            if nodes:
                for line in nodes:
                    p = line.decode("utf-8").split("|")
                    taxa[int(p[0])] = (int(p[1]), p[2].strip())
            raw_names = tar.extractfile("names.dmp")
            if raw_names:
                for line in raw_names:
                    p = line.decode("utf-8").split("|")
                    if p[3].strip() == "scientific name":
                        names[int(p[0])] = p[1].strip()
    except Exception as exc:  # pragma: no cover - best effort loader
        print(f"Warning: failed to parse taxdump at {tar_path}: {exc}")
    return taxa, names


def kingdom_from_local(tid: int, taxa: Dict[int, Tuple[int, str]], names: Dict[int, str]) -> str:
    if not tid or tid not in taxa:
        return "Unknown"
    lineage: List[str] = []
    curr = tid
    while True:
        if curr in names:
            lineage.append(names[curr])
        parent, _ = taxa.get(curr, (1, "no_rank"))
        if parent == curr or parent == 1:
            break
        curr = parent
    return classify_lineage_list(lineage)
