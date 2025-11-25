from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, Iterator, Optional

HEADER_TAX_RE = re.compile(r"Tax=([^=]+?)\s+TaxID=")
HEADER_TAXID_RE = re.compile(r"TaxID=(\d+)")
HEADER_REP_RE = re.compile(r"RepID=([^\s]+)")


def parse_header(header: str) -> Dict[str, Optional[str]]:
    primary, _, rest = header.partition(" ")
    tax_name = None
    taxid = None
    rep_id = None

    tax_match = HEADER_TAX_RE.search(header)
    if tax_match:
        tax_name = tax_match.group(1).strip()
    taxid_match = HEADER_TAXID_RE.search(header)
    if taxid_match:
        taxid = int(taxid_match.group(1))
    rep_match = HEADER_REP_RE.search(header)
    if rep_match:
        rep_id = rep_match.group(1)

    return {
        "ur50_id": primary,
        "tax_name": tax_name,
        "taxid": taxid,
        "rep_id": rep_id,
    }


def iter_uniref50_entries(
    fasta_path: Path,
    *,
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Optional[str]]]:
    with gzip.open(fasta_path, "rt") as handle:
        header = None
        seq_len = 0
        emitted = 0

        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    record = parse_header(header)
                    record["length"] = seq_len
                    yield record
                    emitted += 1
                    if limit and emitted >= limit:
                        return
                header = line[1:]
                seq_len = 0
            else:
                seq_len += len(line)

        if header is not None and (not limit or emitted < limit):
            record = parse_header(header)
            record["length"] = seq_len
            yield record
