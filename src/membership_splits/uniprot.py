from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, MutableMapping, Optional, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm


class UniprotIdMapper:
    """
    Lightweight batch mapper around the UniProt REST API. Results are cached
    on disk (JSON) so repeated annotations of the same accessions are free.
    """

    API_URL = "https://rest.uniprot.org/uniprotkb/search"

    def __init__(
        self,
        cache_path: Path,
        *,
        fetch_missing: bool = True,
        chunk_size: int = 100,
        throttle_seconds: float = 0.2,
    ) -> None:
        self.cache_path = cache_path
        self.fetch_missing = fetch_missing
        self.chunk_size = chunk_size
        self.throttle_seconds = throttle_seconds
        self._cache: MutableMapping[str, Dict[str, object]] = self._load_cache()

        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.headers.update(
            {
                "User-Agent": "membership-splits/0.1 (+https://github.com/)",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> MutableMapping[str, Dict[str, object]]:
        if not self.cache_path.exists():
            return {}
        with open(self.cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, indent=2)
        tmp_path.replace(self.cache_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_accessions(
        self,
        accessions: Sequence[str],
        *,
        fetch_missing: Optional[bool] = None,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, object]]:
        """
        Return a mapping from accession -> metadata dictionary. Missing entries
        are retrieved from UniProt (unless disabled) and persisted to disk.
        """
        result: Dict[str, Dict[str, object]] = {}
        missing: list[str] = []
        for acc in accessions:
            if acc in self._cache:
                result[acc] = self._cache[acc]
            else:
                missing.append(acc)

        should_fetch = self.fetch_missing if fetch_missing is None else fetch_missing
        if missing and should_fetch:
            iterator = range(0, len(missing), self.chunk_size)
            if show_progress:
                iterator = tqdm(iterator, desc="UniProt fetch", unit="batch")
            for start in iterator:
                batch = missing[start : start + self.chunk_size]
                fetched = self._fetch_batch(batch)
                self._cache.update(fetched)
                result.update(fetched)
                if self.throttle_seconds:
                    time.sleep(self.throttle_seconds)
            self._save_cache()

        return {acc: result.get(acc, {}) for acc in accessions}

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def _fetch_batch(self, batch: Sequence[str]) -> Dict[str, Dict[str, object]]:
        query = " OR ".join(f"(accession:{acc})" for acc in batch)
        params = {
            "query": query,
            "fields": "accession,organism_name,organism_id,lineage,length",
            "format": "json",
            "size": len(batch),
        }
        resp = self._session.get(self.API_URL, params=params, timeout=120)
        resp.raise_for_status()
        payload = resp.json()

        mapping: Dict[str, Dict[str, object]] = {}
        for entry in payload.get("results", []):
            accession = entry["primaryAccession"]
            org = entry.get("organism", {})
            lineage = org.get("lineage", [])
            mapping[accession] = {
                "accession": accession,
                "organism_name": org.get("scientificName"),
                "taxon_id": org.get("taxonId"),
                "lineage": lineage,
                "length": entry.get("sequence", {}).get("length"),
            }
        return mapping
