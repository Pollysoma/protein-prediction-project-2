"""Score EAT (nearest neighbour in embedding space) and MMseqs2 on CATH test219.

All scoring comes from the original Rostlab/EAT code (train_prottucker.py):
DataSplitter reads the sets and labels, Eval does the k=1 Euclidean search,
singleton masking, hierarchical correctness and the bootstrap error. This
script only wires it up and reports accuracy with a 95 % CI (1.96 x bootstrap
standard deviation), as in Heinzinger et al. 2022, Table 1.
"""

import argparse
import copy
import csv
import random
import sys
from pathlib import Path

import h5py
import numpy as np

LEVELS = ["C", "A", "T", "H"]


def load_eat(eat_dir):
    sys.path.insert(0, str(eat_dir))
    import train_prottucker

    return train_prottucker


def make_splitter(eat, embedding_p, label_p):
    class Splitter(eat.DataSplitter):
        """DataSplitter with an explicit label file and support for plain h5 keys."""

        def __init__(self):
            self.verbose = False
            self.id2embedding = self.get_precomputed_embeddings(embedding_p)
            self.cath_label_path = label_p
            self.id2label, self.label2id = self.parse_label_mapping_cath(set(self.id2embedding))

        def get_precomputed_embeddings(self, path):
            with h5py.File(path, "r") as f:
                if "|" in next(iter(f.keys())):
                    return super().get_precomputed_embeddings(path)
                return {key: np.expand_dims(np.array(f[key]), axis=0) for key in f.keys()}

    return Splitter()


def counts_per_level(evaluator):
    """Number of test queries that count at each level after singleton masking."""
    n = np.zeros(len(LEVELS), dtype=int)
    for test_id in evaluator.testIdx2label.values():
        y = evaluator.mask_singletons(copy.deepcopy(evaluator.id2label[test_id]))
        n += ~np.isnan(np.array(y, dtype=float))
    return n


def eval_embeddings(evaluator):
    random.seed(42)
    acc, std = evaluator.evaluate(evaluator.get_lookup_set(), evaluator.get_test_set(), n_nearest=1, update=False)
    return acc, std


def read_best_hits(m8_path, lookup_ids):
    """Query -> target of the hit with the lowest E-value (ties: higher bit score).

    Only targets in the evaluated lookup set count. EAT's read_cath_ids drops
    lookup entries whose sequence occurs twice; the kept copy gets the same hit.
    """
    best = {}
    with open(m8_path) as f:
        for line in f:
            query, target, evalue, bits = line.split()[:4]
            if target not in lookup_ids:
                continue
            key = (float(evalue), -float(bits))
            if query not in best or key < best[query][0]:
                best[query] = (key, target)
    return {query: target for query, (_, target) in best.items()}


def eval_mmseqs(evaluator, m8_path):
    """Same scoring as Eval.evaluate, but the predicted label is the best MMseqs2 hit.

    Queries without any hit are wrong at every level that is not masked.
    """
    best_hit = read_best_hits(m8_path, set(evaluator.lookupIdx2label.values()))
    confmats = evaluator.init_confmats()
    for test_id in evaluator.testIdx2label.values():
        y = evaluator.mask_singletons(copy.deepcopy(evaluator.id2label[test_id]))
        hit = best_hit.get(test_id)
        yhat = evaluator.id2label[hit] if hit is not None else [np.nan] * len(LEVELS)
        confmats = evaluator.add_sample(y, yhat, confmats)
    random.seed(42)
    acc = [confmats[i, 1, 1] / confmats[i].sum() for i in range(len(LEVELS))]
    std = [evaluator.compute_err(confmats[i]) for i in range(len(LEVELS))]
    n_with_hit = sum(test_id in best_hit for test_id in evaluator.testIdx2label.values())
    return acc, std, n_with_hit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eat-dir", type=Path, required=True, help="clone of github.com/Rostlab/EAT")
    parser.add_argument("--labels", type=Path, required=True, help="cath-domain-list.txt")
    parser.add_argument("--embeddings", nargs="+", required=True, metavar="NAME=H5",
                        help="one or more embedding files, e.g. prott5_authors=prott5_cath_S100.h5")
    parser.add_argument("--mmseqs-m8", type=Path, help="MMseqs2 hits of test219 against lookup69k")
    parser.add_argument("--out", type=Path, required=True, help="output CSV")
    args = parser.parse_args()

    eat = load_eat(args.eat_dir)
    sets = args.eat_dir / "data" / "ProtTucker"
    rows = []
    evaluator = None
    for spec in args.embeddings:
        name, h5_path = spec.split("=", 1)
        splitter = make_splitter(eat, Path(h5_path), args.labels)
        lookup = splitter.get_embeddings(sets / "lookup69k.fasta")
        test = splitter.get_embeddings(sets / "test219.fasta")
        evaluator = eat.Eval(lookup, test, splitter, n_classes=len(LEVELS))
        acc, std = eval_embeddings(evaluator)
        n = counts_per_level(evaluator)
        print(f"{name}: lookup {len(lookup)}, queries {len(test)}")
        for i, level in enumerate(LEVELS):
            rows.append([f"EAT {name}", level, 100 * acc[i], 196 * std[i], n[i], len(lookup)])

    if args.mmseqs_m8:
        acc, std, n_with_hit = eval_mmseqs(evaluator, args.mmseqs_m8)
        n = counts_per_level(evaluator)
        print(f"mmseqs2: {n_with_hit} of {len(evaluator.testIdx2label)} queries have a hit")
        for i, level in enumerate(LEVELS):
            rows.append(["MMseqs2 best hit", level, 100 * acc[i], 196 * std[i], n[i], len(evaluator.lookupIdx2label)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "level", "accuracy_pct", "ci95_pct", "n_queries", "n_lookup"])
        for method, level, acc, ci, n_q, n_l in rows:
            writer.writerow([method, level, f"{acc:.1f}", f"{ci:.1f}", n_q, n_l])
    for row in rows:
        print(f"{row[0]:<28} {row[1]}  {row[2]:5.1f} +/- {row[3]:4.1f}  (n={row[4]})")


if __name__ == "__main__":
    main()
