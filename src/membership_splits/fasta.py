from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_fasta(path: Path) -> pd.DataFrame:
    """
    Read a FASTA file and return a DataFrame with columns:
      - accession: header token (first word without '>')
      - description: remainder of the header line
      - sequence: amino-acid string
      - length: computed from the sequence
    """
    records: List[Dict[str, str]] = []
    header = None
    desc = ""
    seq_chunks: List[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(seq_chunks)
                    records.append(
                        {
                            "accession": header,
                            "description": desc,
                            "sequence": sequence,
                            "length": len(sequence),
                        }
                    )
                parts = line[1:].split(maxsplit=1)
                header = parts[0]
                desc = parts[1] if len(parts) == 2 else ""
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if header is not None:
        sequence = "".join(seq_chunks)
        records.append(
            {
                "accession": header,
                "description": desc,
                "sequence": sequence,
                "length": len(sequence),
            }
        )

    return pd.DataFrame.from_records(records)
