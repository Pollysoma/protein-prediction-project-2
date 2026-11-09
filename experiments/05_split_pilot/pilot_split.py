#!/usr/bin/env python3
"""Build and measure the provisional CATH v4.4 three-role split pilot."""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import math
import random
import shlex
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


SIZE_BANDS = [
    ("2", lambda n: n == 2),
    ("3-10", lambda n: 3 <= n <= 10),
    ("11-100", lambda n: 11 <= n <= 100),
    (">100", lambda n: n > 100),
]

HVAL_BANDS = [
    ("no_hit", None, None),
    ("<= -5", -math.inf, -5.0),
    ("(-5, 0]", -5.0, 0.0),
    ("(0, 5]", 0.0, 5.0),
    ("(5, 20]", 5.0, 20.0),
    ("> 20", 20.0, math.inf),
]


def size_band(n):
    for name, predicate in SIZE_BANDS:
        if predicate(n):
            return name
    raise AssertionError(f"no size band for {n}")


def family_class(family):
    return family.split(".")[0]


def lookup_size_band(n):
    return "1" if n == 1 else size_band(n)


def fasta_domain(header):
    return header[1:].split("|")[2].split("/")[0]


def cath_domain(identifier):
    text = identifier[1:] if identifier.startswith(">") else identifier
    fields = text.split("|")
    return fields[2].split("/")[0] if len(fields) >= 3 else text.split("/")[0]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, log_path=None):
    rendered = shlex.join(map(str, command))
    print(f"+ {rendered}", flush=True)
    started = time.monotonic()
    with (
        open(log_path, "w")
        if log_path
        else contextlib.nullcontext(subprocess.DEVNULL)
    ) as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return time.monotonic() - started, rendered


def read_classification(path):
    labels = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 10:
                continue
            domain, cath_class, architecture, topology, homologous = fields[:5]
            labels[domain] = {
                "class": cath_class,
                "superfamily": ".".join(
                    [cath_class, architecture, topology, homologous]
                ),
                "length": int(fields[10]) if len(fields) > 10 else 0,
            }
    return labels


def sequence_ids(path):
    ids = []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(fasta_domain(line))
    return ids


def write_fasta(source, ids, output):
    wanted = set(ids)
    written = 0
    with open(source) as src, open(output, "w") as dst:
        keep = False
        for line in src:
            if line.startswith(">"):
                keep = fasta_domain(line) in wanted
                if keep:
                    written += 1
                    dst.write(line)
            elif keep:
                dst.write(line)
    if written != len(wanted):
        raise RuntimeError(
            f"FASTA subset mismatch: wanted {len(wanted)}, wrote {written}"
        )


def allocate_counts(weights, total, caps):
    """Largest-remainder allocation with deterministic tie-breaking."""
    if total > sum(caps):
        raise ValueError("allocation exceeds capacity")
    weight_sum = sum(weights)
    raw = [weight * total / weight_sum for weight in weights]
    counts = [min(cap, math.floor(value)) for value, cap in zip(raw, caps)]
    remaining = total - sum(counts)
    while remaining:
        candidates = [
            index
            for index, count in enumerate(counts)
            if count < caps[index]
        ]
        if not candidates:
            raise AssertionError("allocation failed")
        candidates.sort(
            key=lambda index: (
                -(raw[index] - math.floor(raw[index])),
                -(caps[index] - counts[index]),
                index,
            )
        )
        counts[candidates[0]] += 1
        remaining -= 1
    return counts


def choose_families(families, family_sizes, seed, calib_target, test_target):
    strata = defaultdict(list)
    for family in families:
        stratum = (
            family_class(family),
            size_band(family_sizes[family]),
        )
        strata[stratum].append(family)

    total_available = len(families)
    total_requested = calib_target + test_target
    total_selected = min(total_requested, total_available)
    if total_selected < total_requested:
        calib_target = total_selected // 2
        test_target = total_selected - calib_target

    ordered_strata = sorted(strata)
    stratum_sizes = [len(strata[key]) for key in ordered_strata]
    selected_counts = allocate_counts(
        stratum_sizes,
        total_selected,
        stratum_sizes,
    )

    rng = random.Random(seed)
    selected = {}
    for key, count in zip(ordered_strata, selected_counts):
        selected[key] = rng.sample(sorted(strata[key]), count)

    calib_quota = [count // 2 for count in selected_counts]
    test_quota = [count // 2 for count in selected_counts]
    remaining_calib = calib_target - sum(calib_quota)
    remaining_test = test_target - sum(test_quota)
    for index, count in enumerate(selected_counts):
        if count % 2 == 0:
            continue
        if remaining_calib > remaining_test or (
            remaining_calib == remaining_test and index % 2 == 0
        ):
            calib_quota[index] += 1
            remaining_calib -= 1
        else:
            test_quota[index] += 1
            remaining_test -= 1
    if remaining_calib or remaining_test:
        raise AssertionError("role quota allocation failed")

    calibration = []
    test = []
    for key, calib_count, test_count in zip(
        ordered_strata, calib_quota, test_quota
    ):
        choices = selected[key]
        calibration.extend(choices[:calib_count])
        test.extend(choices[calib_count : calib_count + test_count])
    return calibration, test


def choose_domains(families, members, rng):
    return {family: rng.choice(sorted(members[family])) for family in families}


def iter_hits(path):
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            query = cath_domain(fields[0])
            target = cath_domain(fields[1])
            fident, alnlen = fields[2:4]
            yield {
                "query": query,
                "target": target,
                "fident": float(fident),
                "alnlen": int(alnlen),
                "qcov": float(fields[4]) if len(fields) > 4 else None,
                "tcov": float(fields[5]) if len(fields) > 5 else None,
            }


def remove_mmseqs_result(mmseqs, result, temporary):
    if any(result.parent.glob(result.name + "*")):
        run([mmseqs, "rmdb", result])
    if temporary.exists():
        shutil.rmtree(temporary)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def append_count(rows, step, metric, value):
    rows.append([step, metric, value])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--hval-script", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-target", type=int, default=1000)
    parser.add_argument("--test-target", type=int, default=1000)
    parser.add_argument("--identity", type=float, default=0.20)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--evalue", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--max-seqs", type=int, default=20000)
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    for stale in args.results.iterdir():
        if stale.is_file():
            stale.unlink()

    mmseqs_version = subprocess.check_output(
        [args.mmseqs, "version"], text=True
    ).strip()

    labels = read_classification(args.classification)
    sequence_id_list = sequence_ids(args.fasta)
    missing_sequences = sorted(set(labels) - set(sequence_id_list))
    missing_labels = sorted(set(sequence_id_list) - set(labels))
    if missing_sequences or missing_labels:
        raise RuntimeError(
            "CATH files disagree: "
            f"{len(missing_sequences)} missing sequences, "
            f"{len(missing_labels)} missing labels"
        )

    family_members = defaultdict(list)
    for domain in sequence_id_list:
        family_members[labels[domain]["superfamily"]].append(domain)
    family_sizes = {
        family: len(members) for family, members in family_members.items()
    }
    eligible_families = {
        family for family, members in family_members.items() if len(members) >= 2
    }
    eligible_domains = sum(
        len(family_members[family]) for family in eligible_families
    )

    calibration_families, test_families = choose_families(
        eligible_families,
        family_sizes,
        args.seed,
        args.calibration_target,
        args.test_target,
    )
    rng = random.Random(args.seed)
    calibration_domains = choose_domains(calibration_families, family_members, rng)
    test_domains = choose_domains(test_families, family_members, rng)
    query_domains = {**calibration_domains, **test_domains}
    query_domain_ids = set(query_domains.values())
    candidate_lookup = [
        domain for domain in sequence_id_list if domain not in query_domain_ids
    ]

    query_fasta = args.work / "queries.fasta"
    calibration_fasta = args.work / "calibration.fasta"
    test_fasta = args.work / "test.fasta"
    candidate_fasta = args.work / "candidate_lookup.fasta"
    write_fasta(args.fasta, query_domains.values(), query_fasta)
    write_fasta(args.fasta, calibration_domains.values(), calibration_fasta)
    write_fasta(args.fasta, test_domains.values(), test_fasta)
    write_fasta(args.fasta, candidate_lookup, candidate_fasta)

    query_db = args.work / "queries_db"
    calibration_db = args.work / "calibration_db"
    test_db = args.work / "test_db"
    candidate_db = args.work / "candidate_db"
    for fasta, database in [
        (query_fasta, query_db),
        (calibration_fasta, calibration_db),
        (test_fasta, test_db),
        (candidate_fasta, candidate_db),
    ]:
        run([args.mmseqs, "createdb", fasta, database])

    search_flags = [
        "--num-iterations", "3",
        "-s", "7.5",
        "--cov-mode", "0",
        "-c", f"{args.coverage:g}",
        "--alignment-mode", "3",
        "-e", f"{args.evalue:g}",
        "--max-seqs", str(args.max_seqs),
        "--threads", str(args.threads),
    ]
    main_result = args.work / "main_result"
    main_tmp = args.work / "main_tmp"
    main_hits = args.work / "main_hits.m8"
    remove_mmseqs_result(args.mmseqs, main_result, main_tmp)
    search_command = [
        args.mmseqs, "search", query_db, candidate_db, main_result, main_tmp,
        *search_flags,
    ]
    search_seconds, search_rendered = run(search_command, args.work / "main_search.log")
    convert_command = [
        args.mmseqs, "convertalis", query_db, candidate_db, main_result, main_hits,
        "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
        "--threads", str(args.threads),
    ]
    convert_seconds, convert_rendered = run(convert_command)

    hits = iter_hits(main_hits)
    removed_lookup = {
        hit["target"]
        for hit in hits
        if hit["fident"] >= args.identity
    }
    final_lookup = [domain for domain in candidate_lookup if domain not in removed_lookup]

    surviving_calibration = []
    surviving_test = []
    lost_calibration = 0
    lost_test = 0
    final_lookup_set = set(final_lookup)
    for family, domain in calibration_domains.items():
        remaining = [
            member
            for member in family_members[family]
            if member != domain and member in final_lookup_set
        ]
        if remaining:
            surviving_calibration.append((family, domain))
        else:
            lost_calibration += 1
    for family, domain in test_domains.items():
        remaining = [
            member
            for member in family_members[family]
            if member != domain and member in final_lookup_set
        ]
        if remaining:
            surviving_test.append((family, domain))
        else:
            lost_test += 1

    surviving_lookup_families = {
        labels[domain]["superfamily"] for domain in final_lookup
    }

    # Cross-role leakage measurement, without dropping either query.
    cross_seconds = 0.0
    violating_pairs = set()
    directional_hits = Counter()
    role_query_hits = defaultdict(set)
    for query_db_name, target_db_name, source_role, target_role in [
        (calibration_db, test_db, "calibration", "test"),
        (test_db, calibration_db, "test", "calibration"),
    ]:
        result = args.work / f"cross_{source_role}_{target_role}_result"
        tmp = args.work / f"cross_{source_role}_{target_role}_tmp"
        hits_path = args.work / f"cross_{source_role}_{target_role}.m8"
        remove_mmseqs_result(args.mmseqs, result, tmp)
        command = [
            args.mmseqs, "search", query_db_name, target_db_name, result, tmp,
            *search_flags,
        ]
        elapsed, _ = run(command, args.work / f"cross_{source_role}_{target_role}.log")
        cross_seconds += elapsed
        run([
            args.mmseqs, "convertalis", query_db_name, target_db_name, result, hits_path,
            "--format-output", "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "--threads", str(args.threads),
        ])
        for hit in iter_hits(hits_path):
            if hit["fident"] >= args.identity:
                directional_hits[f"{source_role}_to_{target_role}"] += 1
                pair = frozenset((hit["query"], hit["target"]))
                violating_pairs.add(pair)
                role_query_hits[source_role].add(hit["query"])
                role_query_hits[target_role].add(hit["target"])

    # Reuse the repository's HVAL implementation.
    spec = importlib.util.spec_from_file_location("hval", args.hval_script)
    hval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hval_module)
    surviving_hits = args.work / "surviving_hits.m8"
    with open(main_hits) as src, open(surviving_hits, "w") as dst:
        for line in src:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                query = cath_domain(fields[0])
                target = cath_domain(fields[1])
                if target in final_lookup_set:
                    dst.write("\t".join([query, target, *fields[2:]]) + "\n")
    best_hits = hval_module.best_per_query(surviving_hits)

    query_rows = []
    hval_values = {"calibration": [], "test": []}
    for role, pairs in [
        ("calibration", surviving_calibration),
        ("test", surviving_test),
    ]:
        for family, domain in pairs:
            hit = best_hits.get(domain)
            if hit is None:
                value, target, pide, alnlen = "", "", "", ""
            else:
                value, target, pide, alnlen = hit
                hval_values[role].append(value)
            query_rows.append([
                role,
                domain,
                family,
                labels[domain]["class"],
                size_band(len(family_members[family])),
                value,
                target,
                pide,
                alnlen,
            ])

    count_rows = []
    append_count(count_rows, "input", "total_domains", len(sequence_id_list))
    append_count(count_rows, "input", "total_superfamilies", len(family_members))
    append_count(count_rows, "eligibility", "eligible_superfamilies", len(eligible_families))
    append_count(count_rows, "eligibility", "eligible_domains", eligible_domains)
    append_count(count_rows, "assignment", "calibration_query_families", len(calibration_families))
    append_count(count_rows, "assignment", "test_query_families", len(test_families))
    append_count(count_rows, "assignment", "candidate_lookup_domains", len(candidate_lookup))
    append_count(count_rows, "filter", "domains_removed", len(removed_lookup))
    append_count(count_rows, "filter", "lookup_domains_after_filter", len(final_lookup))
    append_count(count_rows, "query_loss", "calibration_families_lost", lost_calibration)
    append_count(count_rows, "query_loss", "test_families_lost", lost_test)
    append_count(count_rows, "query_loss", "surviving_calibration_queries", len(surviving_calibration))
    append_count(count_rows, "query_loss", "surviving_test_queries", len(surviving_test))
    append_count(count_rows, "final", "lookup_superfamilies", len(surviving_lookup_families))
    append_count(
        count_rows,
        "cross_role_filter",
        "unique_pairs_at_or_above_20pct",
        len(violating_pairs),
    )
    append_count(
        count_rows,
        "cross_role_filter",
        "calibration_to_test_hits",
        directional_hits["calibration_to_test"],
    )
    append_count(
        count_rows,
        "cross_role_filter",
        "test_to_calibration_hits",
        directional_hits["test_to_calibration"],
    )
    append_count(
        count_rows,
        "cross_role_filter",
        "calibration_queries_with_violation",
        len(role_query_hits["calibration"]),
    )
    append_count(
        count_rows,
        "cross_role_filter",
        "test_queries_with_violation",
        len(role_query_hits["test"]),
    )

    write_csv(args.results / "step_counts.csv", ["step", "metric", "value"], count_rows)

    distribution_rows = []
    for band_name, low, high in HVAL_BANDS:
        row = [band_name]
        for role in ("calibration", "test"):
            denominator = len(surviving_calibration if role == "calibration" else surviving_test)
            if band_name == "no_hit":
                count = denominator - len(hval_values[role])
            else:
                count = sum(low < value <= high for value in hval_values[role])
            row.extend([count, 100 * count / denominator if denominator else 0])
        distribution_rows.append(row)
    write_csv(
        args.results / "hval_distribution.csv",
        ["hval_band", "calibration_count", "calibration_pct", "test_count", "test_pct"],
        distribution_rows,
    )
    write_csv(
        args.results / "hval_query_details.csv",
        ["role", "query", "superfamily", "class", "size_band", "best_hval", "best_target", "pide_pct", "aln_len"],
        query_rows,
    )

    balance_rows = []
    surviving_family_names = {
        "calibration": [family for family, _ in surviving_calibration],
        "test": [family for family, _ in surviving_test],
    }
    for role_name, families in [
        ("calibration", calibration_families),
        ("test", test_families),
    ]:
        selected_class = Counter(family_class(family) for family in families)
        selected_band = Counter(
            size_band(family_sizes[family]) for family in families
        )
        surviving_families = surviving_family_names[role_name]
        surviving_class = Counter(family_class(family) for family in surviving_families)
        surviving_band = Counter(
            size_band(family_sizes[family]) for family in surviving_families
        )
        all_classes = sorted({family_class(family) for family in eligible_families})
        for class_name in all_classes:
            balance_rows.append([
                "class",
                class_name,
                role_name,
                selected_class[class_name],
                surviving_class[class_name],
            ])
        for band_name, _ in SIZE_BANDS:
            balance_rows.append([
                "size_band",
                band_name,
                role_name,
                selected_band[band_name],
                surviving_band[band_name],
            ])
    write_csv(
        args.results / "query_balance.csv",
        ["dimension", "category", "role", "selected_families", "surviving_families"],
        balance_rows,
    )

    family_size_rows = []
    final_lookup_members = defaultdict(list)
    for domain in final_lookup:
        final_lookup_members[labels[domain]["superfamily"]].append(domain)
    for band_name in ["1", *[name for name, _ in SIZE_BANDS]]:
        families = [
            family for family in surviving_lookup_families
            if lookup_size_band(len(family_members[family])) == band_name
        ]
        family_size_rows.append([
            band_name,
            len(families),
            sum(len(final_lookup_members[family]) for family in families),
        ])
    write_csv(
        args.results / "lookup_family_sizes.csv",
        ["size_band", "superfamilies", "domains"],
        family_size_rows,
    )

    calibration_manifest = args.work / "calibration_ids.txt"
    test_manifest = args.work / "test_ids.txt"
    lookup_manifest = args.work / "lookup_ids.txt"
    for path, values in [
        (calibration_manifest, [domain for _, domain in surviving_calibration]),
        (test_manifest, [domain for _, domain in surviving_test]),
        (lookup_manifest, final_lookup),
    ]:
        with open(path, "w") as handle:
            handle.write("\n".join(values) + "\n")

    checksum_rows = []
    for name, path in [
        ("cath_fasta", args.fasta),
        ("cath_classification", args.classification),
        ("calibration_manifest", calibration_manifest),
        ("test_manifest", test_manifest),
        ("lookup_manifest", lookup_manifest),
    ]:
        checksum_rows.append([name, sha256(path)])
    write_csv(args.results / "checksums.csv", ["item", "sha256"], checksum_rows)

    with open(args.results / "reproducibility.txt", "w") as handle:
        handle.write("CATH v4.4 pilot split\n")
        handle.write(f"seed={args.seed}\n")
        handle.write(f"identity_field=fident\n")
        handle.write(f"identity_cutoff={args.identity:.2f} (fraction, inclusive)\n")
        handle.write(f"coverage_mode=0 (bidirectional)\n")
        handle.write(f"coverage_cutoff={args.coverage:.2f}\n")
        handle.write(f"alignment_mode=3 (identical aligned residues / aligned columns)\n")
        handle.write(f"mmseqs_version={mmseqs_version}\n")
        handle.write(f"mmseqs_search_seconds={search_seconds:.3f}\n")
        handle.write(f"mmseqs_convert_seconds={convert_seconds:.3f}\n")
        handle.write(f"cross_role_search_seconds={cross_seconds:.3f}\n")
        handle.write(f"search_command={search_rendered}\n")
        handle.write(f"convert_command={convert_rendered}\n")

    summary_lines = [
        "CATH v4.4 split pilot",
        f"Seed: {args.seed}",
        f"Eligible superfamilies: {len(eligible_families)}",
        f"Eligible domains: {eligible_domains}",
        f"Calibration query families: {len(calibration_families)}",
        f"Test query families: {len(test_families)}",
        f"Candidate lookup before filter: {len(candidate_lookup)}",
        f"Domains removed: {len(removed_lookup)}",
        f"Lookup after filter: {len(final_lookup)}",
        f"Lookup superfamilies after filter: {len(surviving_lookup_families)}",
        f"Calibration families lost: {lost_calibration}",
        f"Test families lost: {lost_test}",
        f"Surviving calibration queries: {len(surviving_calibration)}",
        f"Surviving test queries: {len(surviving_test)}",
        f"Cross-role pairs at or above 20% identity: {len(violating_pairs)}",
        f"Calibration queries with no hit: {len(surviving_calibration) - len(hval_values['calibration'])}",
        f"Test queries with no hit: {len(surviving_test) - len(hval_values['test'])}",
        f"Calibration queries above HVAL 0: {sum(value > 0 for value in hval_values['calibration'])}",
        f"Test queries above HVAL 0: {sum(value > 0 for value in hval_values['test'])}",
        f"Calibration queries involved in a cross-role violation: {len(role_query_hits['calibration'])}",
        f"Test queries involved in a cross-role violation: {len(role_query_hits['test'])}",
        f"Main MMseqs2 search wall-clock: {search_seconds:.1f}s",
        f"Cross-role search wall-clock: {cross_seconds:.1f}s",
    ]
    with open(args.results / "summary.txt", "w") as handle:
        handle.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines), flush=True)


if __name__ == "__main__":
    main()
