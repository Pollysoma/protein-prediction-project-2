#!/usr/bin/env python3
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
    parser.add_argument(
        "--uniref50_filter", action="store_true",
        help="Filter results to keep one representative per UniRef50 cluster. This is slower as it requires extra API calls."
    )
    return parser


# --- Core Logic ---
def build_uniprot_query(args):
    # ... (this function is unchanged)
    query_parts = [
        f"(date_modified:[* TO {args.created_before}])",
        f"(existence:{args.existence_level})",
    ]
    if args.reviewed:
        query_parts.append("(reviewed:true)")
    query = " AND ".join(query_parts)
    logging.info(f"Constructed UniProt query: {query}")
    return query


def get_next_link(headers):
    # ... (this function is unchanged)
    if "Link" in headers:
        link_header = headers["Link"]
        links = link_header.split(", ")
        for link in links:
            if 'rel="next"' in link:
                return link.split(";")[0].strip("<>")
    return None


def fetch_uniprot_data(query, result_limit=None):
    # ... (this function is unchanged)
    all_results = []
    params = {"query": query, "format": "json", "fields": "accession,protein_name,sequence", "size": PAGE_SIZE}
    url = UNIPROTKB_API_URL
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


def get_uniref50_id_for_accession(accession):
    """
    Queries the UniRef API for a single accession to find its UniRef50 cluster ID.
    Returns the cluster ID string or None if not found or an error occurs.
    """
    query = f"(uniprot_id:{accession}) AND (identity:0.5)"
    params = {"query": query, "format": "json", "fields": "id"}

    try:
        # Respect API rate limits to avoid being blocked
        time.sleep(0.5)
        response = requests.get(UNIREF_API_URL, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get("results")

        if results:
            # As per your example, the ID is in the first result object
            return results[0].get("id")

    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not fetch UniRef ID for {accession}: {e}")

    return None


def filter_by_uniref50(results):
    """
    Filters a list of UniProtKB results to keep only one entry per UniRef50 ID.
    This version processes proteins one by one.
    """
    logging.info(f"Starting simple UniRef50 filtering for {len(results)} proteins...")
    seen_uniref50_ids = set()
    filtered_results = []

    for i, entry in tqdm(enumerate(results), total=len(results)):
        accession = entry['primaryAccession']

        uniref_id = get_uniref50_id_for_accession(accession)
        logging.debug(f"Processing ({i + 1}/{len(results)}): {accession} is in cluster {uniref_id}")

        if not uniref_id:
            logging.warning(f"No UniRef50 cluster found for {accession}.")
            filtered_results.append(entry)
            continue

        if uniref_id not in seen_uniref50_ids:
            filtered_results.append(entry)
            seen_uniref50_ids.add(uniref_id)
        else:
            logging.info(f"Cluster {uniref_id} already seen. Skipping {accession}.")

    logging.info(f"Filtered {len(results)} entries down to {len(filtered_results)} non-redundant entries.")
    return filtered_results


def save_results(results, outfile, output_format):
    # ... (this function is unchanged)
    logging.info(f"Saving {len(results)} records to '{outfile}' in '{output_format}' format.")
    # ... (implementation is identical)
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


# --- Main Execution (Unchanged) ---
def main():
    setup_logging()
    parser = setup_arg_parser()
    args = parser.parse_args()

    logging.info("Starting UniProt data fetching process...")
    query = build_uniprot_query(args)
    results = fetch_uniprot_data(query, result_limit=args.limit)

    if results is None:
        logging.error("Could not fetch data from UniProt. Exiting.")
        sys.exit(1)

    if args.uniref50_filter:
        results = filter_by_uniref50(results)

    if not results:
        logging.warning("Query returned no results after filtering.")
        sys.exit(0)

    save_results(results, args.outfile, args.format)


if __name__ == "__main__":
    main()