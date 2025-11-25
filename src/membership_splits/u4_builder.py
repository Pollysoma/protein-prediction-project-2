from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .bins import assign_length_bin
from .taxonomy import kingdom_from_local, load_taxdump_minimal
from .streaming import SequenceHasher


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "membership-splits/0.1"})
    return s


def fetch_uniref_cluster(session: requests.Session, cluster_id: str, cursor: str | None = None) -> dict:
    url = f"https://rest.uniprot.org/uniref/{cluster_id}"
    params = {"format": "json", "size": 500}
    if cursor:
        params["cursor"] = cursor
    resp = session.get(url, params=params, timeout=30)
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    return resp.json()


def extract_members(payload: dict) -> Tuple[List[Tuple[str, Optional[int]]], Optional[str], Optional[int]]:
    members = []
    for entry in payload.get("members", []):
        seq = (entry.get("sequence") or {}).get("value")
        taxid = (entry.get("organism") or {}).get("taxId")
        if seq:
            members.append((seq, taxid))
    rep = payload.get("representativeMember")
    if rep:
        seq = (rep.get("sequence") or {}).get("value")
        taxid = (rep.get("organism") or {}).get("taxId")
        if seq:
            members.append((seq, taxid))
    next_cursor = payload.get("nextCursor") or (payload.get("links") or {}).get("next")
    size = payload.get("size") or payload.get("memberCount")
    return members, next_cursor, size


class U4Builder:
    """
    Builds U4 homologs by querying UniRef50 clusters and selecting one member
    per S4 sequence that matches kingdom/length_bin and is not already used.
    """

    def __init__(
        self,
        *,
        taxdump: Path,
        length_edges: List[int],
        cache_path: Path,
        seed: int = 123,
    ) -> None:
        self.session = _session()
        self.taxa, self.names = load_taxdump_minimal(taxdump)
        self.length_edges = length_edges
        self.cache_path = cache_path
        self.random = random.Random(seed)
        self.cache: Dict[str, List[Tuple[str, Optional[int]]]] = self._load_cache()
        self.used = SequenceHasher()

    def _load_cache(self) -> Dict[str, List[Tuple[str, Optional[int]]]]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text())
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        try:
            tmp.replace(self.cache_path)
        except PermissionError:
            pass

    def _members_for(self, cluster_id: str) -> List[Tuple[str, Optional[int]]]:
        if cluster_id in self.cache:
            return self.cache[cluster_id]
        all_members: List[Tuple[str, Optional[int]]] = []
        cursor = None
        while True:
            payload = fetch_uniref_cluster(self.session, cluster_id, cursor=cursor)
            if not payload:
                break
            members, next_cursor, size = extract_members(payload)
            all_members.extend(members)
            if size and len(all_members) >= size:
                break
            if next_cursor:
                cursor = next_cursor
            else:
                break
            time.sleep(0.1)
        self.cache[cluster_id] = all_members
        if len(self.cache) % 50 == 0:
            self._save_cache()
        return all_members

    def find_match(
        self,
        ur50_id: str,
        target_kingdom: str,
        target_bin: int,
    ) -> Optional[Tuple[str, str, int, int]]:
        members = self._members_for(ur50_id)
        self.random.shuffle(members)
        for seq, taxid in members:
            if self.used.seen(seq):
                continue
            kingdom = kingdom_from_local(int(taxid), self.taxa, self.names) if taxid else "Unknown"
            if kingdom != target_kingdom:
                continue
            length = len(seq)
            bin_idx = assign_length_bin(length, self.length_edges)
            if bin_idx != target_bin:
                continue
            self.used.add(seq)
            return seq, kingdom, length, bin_idx
        return None
