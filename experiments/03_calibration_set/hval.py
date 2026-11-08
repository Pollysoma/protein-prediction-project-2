"""HSSP value (HVAL) of MMseqs2 hits, and per-query summaries.

HVAL = PIDE - pI(L) with the curve from Rost 1999 (the criterion the ProtTucker
paper used to make test219 strictly non-redundant to lookup69k):

    pI(L) = 100                                 for L < 11
    pI(L) = 480 * L ** (-0.32 * (1 + e ** (-L / 1000)))   for 11 <= L <= 450
    pI(L) = 19.5                                for L > 450

L is the number of aligned residues and PIDE the percentage of identical
residues in the alignment. HVAL > 0 means detectable sequence similarity;
every query of test219 has HVAL <= 0 to every protein in lookup69k.
"""

import argparse
import csv
import math
from pathlib import Path


def pide_threshold(aln_len):
    if aln_len < 11:
        return 100.0
    if aln_len > 450:
        return 19.5
    return 480 * aln_len ** (-0.32 * (1 + math.exp(-aln_len / 1000)))


def hval(pide_pct, aln_len):
    return pide_pct - pide_threshold(aln_len)


def best_per_query(m8_path):
    """Query -> (highest HVAL, its target, its PIDE, its alignment length)."""
    best = {}
    with open(m8_path) as f:
        for line in f:
            query, target, fident, aln_len = line.split()[:4]
            if query == target:
                continue
            pide = float(fident) * 100
            value = hval(pide, int(aln_len))
            if query not in best or value > best[query][0]:
                best[query] = (value, target, pide, int(aln_len))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m8", type=Path, help="MMseqs2 hits, columns query,target,fident,alnlen,...")
    parser.add_argument("--queries", type=Path, required=True, help="FASTA of all queries, including those without a hit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    queries = [l[1:].split()[0] for l in open(args.queries) if l.startswith(">")]
    best = best_per_query(args.m8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "best_hval", "best_target", "pide_pct", "aln_len"])
        for query in queries:
            if query in best:
                value, target, pide, aln_len = best[query]
                writer.writerow([query, f"{value:.2f}", target, f"{pide:.1f}", aln_len])
            else:
                writer.writerow([query, "", "", "", ""])

    values = [best[q][0] for q in queries if q in best]
    no_hit = len(queries) - len(values)
    bins = [(-999, -5), (-5, 0), (0, 5), (5, 20), (20, 999)]
    print(f"{args.m8.name}: {len(queries)} queries, {no_hit} without any hit")
    print(f"  HVAL <= 0 (as remote as test219): {no_hit + sum(v <= 0 for v in values)}")
    for low, high in bins:
        print(f"  {low:>4} < HVAL <= {high:<4}: {sum(low < v <= high for v in values)}")


if __name__ == "__main__":
    main()
