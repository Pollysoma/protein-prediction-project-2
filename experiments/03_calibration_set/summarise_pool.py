"""How many calibration candidates survive which rule, and how remote they are.

Reads the two HVAL tables from survey_pool.sh and writes:
  pool_survey.csv       one row per candidate: HVAL to lookup69k and to test219
  pool_summary.csv      counts per remoteness band, before and after dropping
                        candidates that are similar to a test219 query, with the
                        superfamily diversity that limits how many independent
                        queries a band can contribute
"""

import argparse
import csv
import collections
from pathlib import Path

# HVAL bands. test219 queries all sit at <= 0 by construction.
BANDS = [("midnight (<= -5)", -1e9, -5), ("remote (-5, 0]", -5, 0), ("twilight (0, 5]", 0, 5),
         ("clear (5, 20]", 5, 20), ("close (> 20)", 20, 1e9)]


def read_hval(path):
    out = {}
    for row in csv.DictReader(open(path)):
        out[row["query"]] = float(row["best_hval"]) if row["best_hval"] else None
    return out


def band(value):
    if value is None:
        return "no hit"
    for name, low, high in BANDS:
        if low < value <= high:
            return name
    return "no hit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hval-lookup", type=Path, required=True)
    parser.add_argument("--hval-test219", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--lookup", type=Path, required=True, help="lookup69k.fasta")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--leak-hval", type=float, default=0.0,
                        help="drop a candidate when its HVAL to any test219 query exceeds this")
    parser.add_argument("--caps", type=int, nargs="+", default=[3, 5, 8],
                        help="per-superfamily caps to report")
    args = parser.parse_args()

    to_lookup, to_test = read_hval(args.hval_lookup), read_hval(args.hval_test219)
    labels = dict(line.strip().split(",") for line in open(args.labels) if "," in line)
    lookup_ids = {l[1:].split()[0] for l in open(args.lookup) if l.startswith(">")}
    # Superfamilies a transfer can actually hit: present in the lookup set.
    lookup_families = {labels[i] for i in lookup_ids if i in labels}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for query, lookup_hval in to_lookup.items():
        test_hval = to_test.get(query)
        leaks = test_hval is not None and test_hval > args.leak_hval
        rows.append([query, labels.get(query, ""), lookup_hval, test_hval, int(leaks)])
    with open(args.out_dir / "pool_survey.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "cath_h", "hval_to_lookup69k", "hval_to_test219", "drop_for_test_similarity"])
        for query, label, lookup_hval, test_hval, leaks in rows:
            writer.writerow([query, label,
                             "" if lookup_hval is None else f"{lookup_hval:.2f}",
                             "" if test_hval is None else f"{test_hval:.2f}", leaks])

    counts = collections.Counter()
    families = collections.defaultdict(set)
    answerable = collections.Counter()
    answerable_families = collections.defaultdict(collections.Counter)
    for query, label, lookup_hval, _, leaks in rows:
        name = band(lookup_hval)
        counts[(name, "all")] += 1
        if not leaks:
            counts[(name, "kept")] += 1
            families[name].add(label)
            if label in lookup_families:
                answerable[name] += 1
                answerable_families[name][label] += 1
    order = ["no hit"] + [name for name, _, _ in BANDS]
    def capped(name, cap):
        return sum(min(n, cap) for n in answerable_families[name].values())
    with open(args.out_dir / "pool_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["hval_band_to_lookup69k", "candidates", "kept_after_test_filter", "superfamilies_kept",
                  "answerable", "answerable_superfamilies"]
        header += [f"answerable_cap{cap}" for cap in args.caps]
        writer.writerow(header)
        for name in order:
            writer.writerow([name, counts[(name, "all")], counts[(name, "kept")], len(families[name]),
                             answerable[name], len(answerable_families[name])]
                            + [capped(name, cap) for cap in args.caps])
        writer.writerow(["total", len(rows), sum(counts[(n, "kept")] for n in order),
                         len(set().union(*families.values())) if families else 0,
                         sum(answerable.values()),
                         len(set().union(*(set(c) for c in answerable_families.values())) if answerable_families else set())]
                        + [sum(capped(n, cap) for n in order) for cap in args.caps])

    print(f"{'band':<18} {'candidates':>11} {'kept':>7} {'families':>9} {'answerable':>11} {'cap5':>6}")
    for name in order:
        print(f"{name:<18} {counts[(name, 'all')]:>11} {counts[(name, 'kept')]:>7} {len(families[name]):>9} "
              f"{answerable[name]:>11} {capped(name, 5):>6}")
    print(f"{'total':<18} {len(rows):>11} {sum(counts[(n, 'kept')] for n in order):>7}")


if __name__ == "__main__":
    main()
