# 02: All Table 1 pLMs, Euclidean vs cosine distance

- **Date:** 05.11.2026
- **Issue / PR:** #18, #19 (checks added in #21)
- **Code:** [`experiments/02_table1_plms/`](../experiments/02_table1_plms/), commit `9bff2a4`
- **Status:** done

## Question

1. Does the reproduction from [01](01_prott5_reproduction.md) hold for the other raw pLMs in Table 1 (ESM-1b, ProtBERT, ProSE-DLM, ProSE-MT)?
2. Does cosine distance transfer CATH labels better or worse than Euclidean? This is part of research question 1. One pLM is not enough to answer it, so we use all five.

test219 is only used for reporting here, not to choose a distance.

## Setup

As in [`methods.md`](methods.md), with the authors' embeddings for all five pLMs. ProSE-MT is trained with structural supervision (SCOPe), so it is not a raw pLM in our sense; it is listed because it is in Table 1. Each pLM is scored twice: Euclidean, and cosine (unit-length embeddings, then Euclidean).

Commands: `WORK=/scratch/eat bash run.sh` for the accuracies (CPU, about 3 minutes for all ten runs after the 5.9 GB download). The supporting checks come from `checks.py`, run on the server because they also need our ProtT5 embeddings and the MMseqs2 hits from 01:

```sh
python checks.py --eat-dir EAT --labels cath-domain-list.txt \
    --embeddings ProtT5=embeddings/prott5_cath_S100.h5 ProSE-MT=embeddings/proseMT_cathS100.h5 \
                 ESM-1b=embeddings/ESM1b_cath_S100.h5 ProSE-DLM=embeddings/proseDLM_cathS100.h5 \
                 ProtBERT=embeddings/protBERT_bfd_half_cath_S100.h5 \
    --reference ProtT5-ours=prott5_ours.h5 --mmseqs-m8 mm/hits.m8 --out-dir results/checks
```

## Results

Accuracy in %, ± 95 % CI; queries counted: C 219, A 219, T 210, H 150. Data: [`euclidean.csv`](../experiments/02_table1_plms/results/euclidean.csv), [`cosine.csv`](../experiments/02_table1_plms/results/cosine.csv).

| pLM | Distance | C | A | T | H |
| --- | --- | --- | --- | --- | --- |
| ProtT5 | paper | 84 ± 5 | 67 ± 6 | 57 ± 6 | 64 ± 8 |
| | Euclidean | 83.6 ± 4.9 | 66.7 ± 6.2 | 56.7 ± 6.7 | 64.0 ± 7.7 |
| | cosine | 81.7 ± 5.1 | 67.1 ± 6.2 | 57.6 ± 6.7 | 63.3 ± 7.8 |
| ProSE-MT | paper | 82 ± 5 | 65 ± 6 | 52 ± 7 | 56 ± 8 |
| | Euclidean | 81.7 ± 5.1 | 65.3 ± 6.3 | 52.4 ± 6.7 | 56.7 ± 8.0 |
| | cosine | 82.6 ± 5.0 | 64.8 ± 6.3 | 53.3 ± 6.7 | 56.7 ± 8.0 |
| ESM-1b | paper | 79 ± 5 | 61 ± 6 | 50 ± 7 | 57 ± 8 |
| | Euclidean | 78.1 ± 5.5 | 60.7 ± 6.5 | 50.5 ± 6.7 | 56.7 ± 8.0 |
| | cosine | 76.7 ± 5.6 | 59.8 ± 6.5 | 50.5 ± 6.7 | 56.7 ± 8.0 |
| ProSE-DLM | paper | 74 ± 6 | 48 ± 7 | 28 ± 6 | 25 ± 7 |
| | Euclidean | 73.5 ± 5.8 | 47.9 ± 6.6 | 27.6 ± 6.0 | 25.3 ± 7.0 |
| | cosine | 73.5 ± 5.8 | 49.3 ± 6.6 | 30.0 ± 6.2 | 26.7 ± 7.1 |
| ProtBERT | paper | 67 ± 6 | 38 ± 6 | 22 ± 6 | 18 ± 6 |
| | Euclidean | 67.6 ± 6.2 | 38.8 ± 6.5 | 21.9 ± 5.6 | 18.0 ± 6.1 |
| | cosine | 69.9 ± 6.1 | 40.6 ± 6.6 | 24.3 ± 5.8 | 21.3 ± 6.5 |

**Euclidean vs cosine neighbours** ([`euclidean_vs_cosine.csv`](../experiments/02_table1_plms/results/checks/euclidean_vs_cosine.csv)). Norm statistics are over the 69,605 lookup embeddings; CV is the standard deviation of the norm divided by its mean.

| pLM | Dim | Mean norm | Norm CV | Same nearest neighbour |
| --- | --- | --- | --- | --- |
| ProtT5 | 1024 | 1.43 | 0.16 | 153 / 219 |
| ProSE-MT | 6165 | 5.50 | 0.21 | 155 / 219 |
| ESM-1b | 1280 | 17.98 | 0.03 | 210 / 219 |
| ProSE-DLM | 6165 | 4.33 | 0.26 | 166 / 219 |
| ProtBERT | 1024 | 1.70 | 0.22 | 119 / 219 |

## Interpretation

- **The reproduction holds for all five pLMs.** With Euclidean distance, 16 of 20 values match Table 1 after rounding. The other four (ESM-1b C, ProtBERT C and A, ProSE-MT H) are each one query away from the published value, far inside the CI. We did not track down these single queries; possible causes are near-ties in distance or small differences in floating-point arithmetic. As in the paper, ProtT5 is best and ProtBERT worst at every level.
- **Cosine and Euclidean give the same accuracy on this benchmark.** The largest difference is 3.3 points (ProtBERT, H), about half a CI, and the sign varies: cosine is slightly better for ProtBERT and ProSE-DLM and slightly worse for ProtT5 and ESM-1b at C. The paper states that earlier studies found Euclidean better than cosine; test219 neither supports nor refutes this.
- **The distance changes the neighbour, not the hit rate.** For ProtT5, 66 of 219 queries get a different nearest neighbour under cosine, for ProtBERT 100, yet accuracy barely moves: the new neighbour is about as often in the right class as the old one. ESM-1b is the exception with only 9 changes, because its embedding norms hardly vary (CV 0.03), and if all norms were equal the two distances would rank neighbours identically. For the other pLMs the norm spread does not predict the number of changes well (ProSE-DLM has the largest spread but fewer changes than ProtBERT).
- **Raw Euclidean distances are on different scales for different pLMs.** Mean norms range from 1.4 (ProtT5) to 18 (ESM-1b). A distance threshold or a reliability score such as goPredSim's `RI = 0.5 / (0.5 + d)` therefore cannot be shared between pLMs as it is. This matters for research question 2. Cosine distance is bounded, which may make it the easier basis for a common reliability score even though it does not change accuracy.
- **test219 is too small to choose a distance.** With 150–219 queries per level the CI is ±5–8 points, so only differences of that size could be detected. A proper comparison needs a larger benchmark and a paired test on per-query results (see next steps).

## Deviations and problems

- **ProtBERT misses two lookup proteins** (`2hr0B03`, `3pifB08`) in the authors' file; its lookup has 69,603 entries instead of 69,605. All other files are complete.
- The four one-query differences from Table 1 above.

## Open questions and next steps

- Save per-query predictions (neighbour, distance, correct per level) so that two settings can be compared with a paired test (e.g. McNemar) instead of overlapping CIs, and so that later experiments can reuse them for reliability curves.
- Repeat the distance comparison on a larger set (CATH v4.4 non-redundant sets) and with our own embeddings of the project pLMs (ProstT5, ESM-2, ESM C, Ankh).
- Check whether accuracy depends on the distance to the nearest neighbour in the same way for all pLMs once distances are put on a common scale; this is the basis for the reliability score.
