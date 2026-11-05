# Raw EAT for all Table 1 pLMs, Euclidean and cosine

The authors' precomputed embeddings (Zenodo 14675997) for the five pLMs in Table 1 of Heinzinger et al. 2022, scored on test219 against lookup69k with the evaluation from [`01_prott5_cath_reproduction`](../01_prott5_cath_reproduction/). Issue #18.

## Results

Accuracy in %, ± 95 % CI. Paper values are Table 1 (Euclidean). Queries counted per level after singleton masking: C 219, A 219, T 210, H 150.

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

Raw numbers: [`results/euclidean.csv`](results/euclidean.csv), [`results/cosine.csv`](results/cosine.csv). Supporting checks (`checks.py`: embedding norms, Euclidean vs cosine neighbours, our vs the authors' ProtT5 embeddings, MMseqs2 hit identities, masking) are in [`results/checks/`](results/checks/). Full write-up with interpretation: [`protocol/02_table1_plms_distance.md`](../../protocol/02_table1_plms_distance.md).

## Findings

- **Euclidean reproduces Table 1.** 16 of the 20 values match after rounding. The other four differ by one query or less: ESM-1b C (78.1 vs 79), ProtBERT C and A (67.6 vs 67, 38.8 vs 38) and ProSE-MT H (56.7 vs 56). The ranking of the pLMs is the same as in the paper.
- **Cosine and Euclidean are indistinguishable on test219.** The largest difference is 3.3 points (ProtBERT, H), about half a confidence interval. Neither distance wins consistently: cosine is slightly higher for ProtBERT and ProSE-DLM and slightly lower for ProtT5 and ESM-1b at C. With 150 to 219 queries per level, test219 is too small to separate the two.
- **ProtBERT misses two lookup proteins** (`2hr0B03`, `3pifB08`) in the authors' file, so its lookup has 69,603 entries instead of 69,605.

## Setup

- Cosine is computed as the Euclidean nearest neighbour after scaling every embedding to unit length; on unit vectors both give the same neighbour.
- No GPU needed: `WORK=/scratch/eat bash run.sh` downloads 5.9 GB from Zenodo, and scoring all five pLMs with both distances takes about 3 minutes on a laptop CPU. Python environment as in `01` (torch, h5py, numpy, matplotlib and seaborn; the last two are only imported by the EAT code).
