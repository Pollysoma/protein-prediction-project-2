#!/usr/bin/env python3
from typing import List, Optional

import requests
import argparse
import logging
import time
import sys
import json

from tqdm import tqdm

# --- Constants ---
UNIPROTKB_API_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIREF_API_URL = "https://rest.uniprot.org/uniref/search"
PAGE_SIZE = 500


# --- Setup (Identical to before) ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        stream=sys.stdout,
    )


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fetch protein sequences from UniProtKB with advanced filters."
    )
    # ... (all previous arguments are the same)
    parser.add_argument(
        "-o", "--outfile", default="uniprot_results.fasta",
        help="Path to the output file. (Default: uniprot_results.fasta)"
    )
    parser.add_argument(
        "--format", choices=["fasta", "json"], default="fasta",
        help="Format for the output file. (Default: fasta)"
    )
    parser.add_argument(
        "--created_before", default="2021-01-01",
        help="Filter for entries created before this date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--existence_level", type=int, default=1,
        help="Filter by protein existence level (PE). (Default: 1)"
    )
    parser.add_argument(
        "--reviewed", action="store_true", default=True,
        help="If specified, only fetch reviewed (Swiss-Prot) entries."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the total number of records to fetch."
    )
    return parser


def get_next_link(headers):
    if "Link" in headers:
        link_header = headers["Link"]
        links = link_header.split(", ")
        for link in links:
            if 'rel="next"' in link:
                return link.split(";")[0].strip("<>")
    return None

def get_paged_results(url, params, result_limit: Optional[int]):
    all_results = []
    while url:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None
        data = response.json()
        results = data.get("results", [])
        all_results.extend(results)
        if result_limit and len(all_results) >= result_limit:
            return all_results[:result_limit]
        url = get_next_link(response.headers)
        params = None
        if url: time.sleep(1)
    return all_results


def save_results(results, outfile, output_format):
    logging.info(f"Saving {len(results)} records to '{outfile}' in '{output_format}' format.")

    try:
        with open(outfile, "w") as f:
            if output_format == "json":
                json.dump(results, f, indent=2)
            elif output_format == "fasta":
                for entry in results:
                    f.write(f">{entry['primaryAccession']}\n{entry['sequence']['value']}\n")
    except IOError as e:
        logging.error(f"Failed to write to file '{outfile}': {e}")
        return False
    logging.info("Successfully saved results.")
    return True


def fetch_uniprot_data(fields: List[str], result_limit=None, modified_before = "2021-01-01", existence_level: int = 1, reviewed: bool = True):
    query_parts = [
        f"(date_modified:[* TO {modified_before}])",
        f"(existence:{existence_level})",
    ]
    if reviewed:
        query_parts.append("(reviewed:true)")
    query = " AND ".join(query_parts)

    logging.info(f"Constructed UniProt query: {query}")

    params = {"query": query, "format": "json", "fields": ",".join(fields), "size": PAGE_SIZE}
    url = UNIPROTKB_API_URL

    results = get_paged_results(url, params, result_limit)
    return results


def fetch_uniref50_data(fields: List[str], result_limit=None, modified_before = "2021-01-01"):
    query_parts = [
        f"(date_modified:[* TO {modified_before}])",
        f"(identity:0.5)",
    ]
    query = " AND ".join(query_parts)

    logging.info(f"Constructed UniRef query: {query}")

    params = {"query": query, "format": "json", "fields": ",".join(fields), "size": PAGE_SIZE}
    url = UNIREF_API_URL

    results = get_paged_results(url, params, result_limit)
    return results



def main():
    setup_logging()
    parser = setup_arg_parser()
    args = parser.parse_args()

    # TODO


if __name__ == "__main__":
    main()