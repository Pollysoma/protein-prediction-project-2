"""Supporting checks for experiments 01 and 02 (test219 queries vs lookup69k).

Per pLM: spread of the embedding norms, and for how many queries the
Euclidean and the cosine nearest neighbour are the same protein.
With --reference: how close a second embedding file of the same pLM is
(per-protein cosine similarity, same nearest neighbour).
With --mmseqs-m8: percent identity of the best MMseqs2 hit per query.
Also: per CATH level, how many queries the evaluation masks, and how many
have no lookup protein with the same label (no correct transfer possible).
"""

import argparse
import copy
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_prott5_cath_reproduction"))
import evaluate  # noqa: E402


def load(eat, eat_dir, labels, h5_path):
    """Lookup and query matrices with the same IDs and order as the evaluation."""
    splitter = evaluate.make_splitter(eat, Path(h5_path), labels, "euclidean")
    sets = eat_dir / "data" / "ProtTucker"
    lookup = splitter.get_embeddings(sets / "lookup69k.fasta")
    test = splitter.get_embeddings(sets / "test219.fasta")
    to_np = lambda d: (list(d), np.vstack([v.cpu().numpy() for v in d.values()]).astype(np.float64))
    return to_np(lookup), to_np(test)


def nearest(lookup, queries, metric):
    if metric == "cosine":
        lookup = lookup / np.linalg.norm(lookup, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    d2 = (queries**2).sum(1)[:, None] + (lookup**2).sum(1)[None, :] - 2 * queries @ lookup.T
    return d2.argmin(axis=1)


def masking_counts(eat, eat_dir, labels, h5_path, lookup_ids, test_ids):
    """Per level: queries masked by mask_singletons vs queries without any lookup member of their label."""
    splitter = evaluate.make_splitter(eat, Path(h5_path), labels, "euclidean")
    id2label, label2id = splitter.parse_label_mapping_cath(set(lookup_ids) | set(test_ids))
    evaluator = SimpleNamespace(label2id=label2id)  # all that Eval.mask_singletons uses
    lookup_prefixes = [{tuple(id2label[i][: lvl + 1]) for i in lookup_ids} for lvl in range(4)]
    rows = []
    for lvl, level in enumerate(evaluate.LEVELS):
        masked = sum(
            np.isnan(eat.Eval.mask_singletons(evaluator, copy.deepcopy(id2label[q]))[lvl]) for q in test_ids
        )
        no_member = sum(tuple(id2label[q][: lvl + 1]) not in lookup_prefixes[lvl] for q in test_ids)
        rows.append([level, len(test_ids), masked, no_member])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eat-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--embeddings", nargs="+", required=True, metavar="NAME=H5")
    parser.add_argument("--reference", metavar="NAME=H5", help="second file to compare with the first --embeddings entry")
    parser.add_argument("--mmseqs-m8", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    eat = evaluate.load_eat(args.eat_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, first = [], None
    for spec in args.embeddings:
        name, path = spec.split("=", 1)
        (lookup_ids, lookup), (test_ids, test) = load(eat, args.eat_dir, args.labels, path)
        first = first or (name, lookup_ids, lookup, test_ids, test)
        norms = np.linalg.norm(lookup, axis=1)
        same = (nearest(lookup, test, "euclidean") == nearest(lookup, test, "cosine")).sum()
        rows.append([name, lookup.shape[1], f"{norms.mean():.3f}", f"{norms.std() / norms.mean():.3f}", same, len(test_ids)])
        print(rows[-1])
    with open(args.out_dir / "euclidean_vs_cosine.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plm", "dim", "norm_mean", "norm_cv", "queries_same_nn", "n_queries"])
        w.writerows(rows)

    if args.reference:
        name, lookup_ids, lookup, test_ids, test = first
        ref_name, ref_path = args.reference.split("=", 1)
        (r_lookup_ids, r_lookup), (r_test_ids, r_test) = load(eat, args.eat_dir, args.labels, ref_path)
        assert lookup_ids == r_lookup_ids and test_ids == r_test_ids
        a, b = np.vstack([lookup, test]), np.vstack([r_lookup, r_test])
        cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        same = (nearest(lookup, test, "euclidean") == nearest(r_lookup, r_test, "euclidean")).sum()
        row = [name, ref_name, len(a), f"{cos.mean():.6f}", f"{cos.min():.6f}", f"{np.abs(a - b).max():.4f}", same, len(test_ids)]
        print(row)
        with open(args.out_dir / "embeddings_ours_vs_authors.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "n_proteins", "cos_sim_mean", "cos_sim_min", "max_abs_diff", "queries_same_nn", "n_queries"])
            w.writerow(row)

    if args.mmseqs_m8:
        _, lookup_ids, _, test_ids, _ = first
        best = evaluate.read_best_hits(args.mmseqs_m8, set(lookup_ids))
        pident = {}
        with open(args.mmseqs_m8) as f:
            for line in f:
                q, t, _, _, pid = line.split()[:5]
                if best.get(q) == t and q not in pident:
                    pident[q] = float(pid)
        with open(args.out_dir / "mmseqs_best_hit_identity.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query", "best_hit", "pident"])
            for q in test_ids:
                w.writerow([q, best.get(q, ""), pident.get(q, "")])
        p = np.array(list(pident.values()))
        print(f"mmseqs best hits: n={len(p)}, median pident {np.median(p):.1f}, max {p.max():.1f}")

    name, lookup_ids, _, test_ids, _ = first
    path = next(s.split("=", 1)[1] for s in args.embeddings if s.startswith(name + "="))
    rows = masking_counts(eat, args.eat_dir, args.labels, path, lookup_ids, test_ids)
    with open(args.out_dir / "masking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "n_queries", "masked", "no_lookup_member"])
        w.writerows(rows)
    for row in rows:
        print("masking", row)


if __name__ == "__main__":
    main()
