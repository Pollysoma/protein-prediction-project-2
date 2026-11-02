# Raw ProtT5 EAT and MMseqs2 on CATH test219

Reproduction of Heinzinger et al. 2022 (NAR Genom Bioinform, lqac043), Table 1: queries from test219, lookup set lookup69k, labels from CATH v4.3 (`cath-domain-list.txt`). Issue #16.

## Results

Accuracy in %, ± 95 % CI (1.96 × std over 10,000 bootstrap resamples). `n` is the number of queries that count at that level after singleton masking.

| Method | C | A | T | H |
| --- | --- | --- | --- | --- |
| EAT, authors' ProtT5 embeddings | 83.6 ± 4.9 | 66.7 ± 6.2 | 56.7 ± 6.7 | 64.0 ± 7.7 |
| EAT, our ProtT5 embeddings | 83.6 ± 4.9 | 66.7 ± 6.2 | 56.7 ± 6.7 | 64.0 ± 7.7 |
| MMseqs2, best hit | 51.6 ± 6.6 | 36.1 ± 6.4 | 29.0 ± 6.1 | 34.7 ± 7.6 |
| *Paper: EAT ProtT5* | *84 ± 5* | *67 ± 6* | *57 ± 6* | *64 ± 8* |
| *Paper: MMseqs2* | *52 ± 7* | *36 ± 6* | *29 ± 6* | *35 ± 6* |
| n | 219 | 219 | 210 | 150 |

Both rows match Table 1 after rounding. Our own embeddings give exactly the same accuracies as the authors' precomputed ones. MMseqs2 finds a hit for 172 of the 219 queries; the other 47 count as wrong.

Raw numbers: [`results/cath_test219.csv`](results/cath_test219.csv).

## Setup

- **Scoring** reuses the original code from [Rostlab/EAT](https://github.com/Rostlab/EAT) at commit `b82ea03` (`DataSplitter` and `Eval` in `train_prottucker.py`). Nearest neighbour by Euclidean distance (k = 1). A level is masked when the query is the only member of its class at that level within lookup ∪ test. Once a level is wrong, all finer levels count as wrong.
- **Embeddings**: `Rostlab/prot_t5_xl_half_uniref50-enc` (revision `94a6abc`), fp16, mean over the last-layer residue embeddings without the end token, U/Z/O/B mapped to X. The 69,605 lookup proteins take about 3.5 min on one H100.
- **MMseqs2** (release `21fbeafa`): single-sequence search, `-s 7.5 -e 10`. Each query takes the label of the hit with the lowest E-value and is scored with the same masking and hierarchy rules as EAT.
- **Data**: sets from the EAT repo, labels and the authors' embeddings from [Zenodo 14675997](https://zenodo.org/records/14675997). The authors' file covers all of CATH-S100, including the test219 domains; only lookup69k IDs are used as lookup.

## Reproduce

```sh
python -m venv venv && venv/bin/pip install -r requirements.txt   # torch from the CUDA 12.1 index
PATH=$PWD/venv/bin:$PATH WORK=/scratch/eat bash run.sh
```

`run.sh` downloads the data (about 6 GB, mostly the authors' embeddings), computes our embeddings, runs MMseqs2 and writes `results/cath_test219.csv`. It needs one CUDA GPU.

## Notes

- `eat.py` in the EAT repo has a batching bug: the condition `seq_idx == len(id2seq)` is never true, so a final partial batch of the shortest sequences is silently skipped. `embed_prott5.py` does not reuse that function.
- Not covered here: the HMMER baseline and the 73 % at H that Ankh reports for the same setup.
