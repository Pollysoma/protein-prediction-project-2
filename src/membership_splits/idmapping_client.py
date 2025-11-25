from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


IDM_URL = "https://rest.uniprot.org/idmapping"


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "membership-splits/0.1"})
    return s


def submit_id_mapping(ids: List[str], *, from_ns: str = "UniRef50", to_ns: str = "UniProtKB") -> str:
    sess = _session()
    payload = {"from": from_ns, "to": to_ns, "ids": ",".join(ids)}
    resp = sess.post(f"{IDM_URL}/run", data=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["jobId"]


def wait_for_job(job_id: str, *, poll: float = 5.0, timeout: float = 900.0) -> None:
    sess = _session()
    start = time.time()
    while True:
        resp = sess.get(f"{IDM_URL}/status/{job_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("jobStatus")
        # The API sometimes responds with only "results" once the job is ready.
        if status in {"FINISHED"} or ("results" in data and status is None):
            return
        if status in {"FAILED", "ERROR"}:
            raise RuntimeError(f"idmapping job {job_id} failed: {data}")
        if time.time() - start > timeout:
            raise TimeoutError(f"idmapping job {job_id} exceeded timeout")
        time.sleep(poll)


def _extract_next_cursor(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    match = re.search(r"cursor=([^&>]+)", link_header)
    if match:
        return match.group(1)
    return None


def fetch_idmap_results(job_id: str, *, batch_size: int = 500) -> pd.DataFrame:
    """
    Stream the mapping results (UR50 -> UniProtKB accession) with pagination.
    """
    sess = _session()
    params = {"format": "tsv", "size": batch_size}
    url = f"{IDM_URL}/results/{job_id}"
    records: List[Dict[str, str]] = []
    cursor = None
    total = None
    progress: Optional[tqdm] = None

    while True:
        if cursor:
            params["cursor"] = cursor
        resp = sess.get(url, params=params, timeout=120)
        resp.raise_for_status()
        if total is None:
            raw_total = resp.headers.get("X-Total-Results")
            total = int(raw_total) if raw_total else None
            if total:
                progress = tqdm(total=total, desc="idmap-results", unit="map")
        text = resp.text.strip()
        lines = text.splitlines()
        if len(lines) <= 1:
            break
        for line in lines[1:]:
            from_id, to_id = line.split("\t")
            records.append({"from": from_id, "to": to_id})
        if progress:
            progress.update(len(lines) - 1)
        cursor = _extract_next_cursor(resp.headers.get("Link"))
        if not cursor:
            break
    if progress:
        progress.close()
    return pd.DataFrame.from_records(records)


def fetch_uniprot_metadata(
    accessions: List[str],
    *,
    fields: str = "accession,length,lineage,sequence",
    batch_size: int = 100,
    desc: str = "uniprot-meta",
) -> pd.DataFrame:
    """
    Fetch UniProtKB metadata (and sequences) for a list of accessions using the stream endpoint.
    Queries are batched to keep URLs short and to respect API limits.
    """
    sess = _session()
    rows: List[pd.DataFrame] = []
    total = len(accessions)
    progress = tqdm(total=total, desc=desc, unit="acc")
    for i in range(0, total, batch_size):
        chunk = accessions[i : i + batch_size]
        query = " OR ".join(f"accession:{acc}" for acc in chunk)
        resp = sess.get(
            "https://rest.uniprot.org/uniprotkb/stream",
            params={"format": "tsv", "fields": fields, "query": query},
            timeout=120,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t")
        rows.append(df)
        progress.update(len(chunk))
    progress.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stream_results(job_id: str, *, fields: str = "from,accession,length,lineage", format: str = "tsv") -> pd.DataFrame:
    sess = _session()
    params = {"fields": fields, "format": format}
    resp = sess.get(f"{IDM_URL}/results/stream/{job_id}", params=params, timeout=120)
    resp.raise_for_status()
    if format == "tsv":
        return pd.read_csv(io.StringIO(resp.text), sep="\t")
    raise ValueError("Only TSV format supported")
