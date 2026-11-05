# 01: Reproduce raw ProtT5 EAT and the MMseqs2 baseline

- **Date:** 02.11.2026
- **Issue / PR:** #16, #17 (checks added in #21)
- **Code:** [`experiments/01_prott5_cath_reproduction/`](../experiments/01_prott5_cath_reproduction/), commit `a6735aa`
- **Status:** done

## Question

Before we compare pLMs, our pipeline has to give the published numbers. Can we reproduce raw ProtT5 EAT and the MMseqs2 baseline on test219 vs lookup69k (Heinzinger et al. 2022, Table 1), first with the authors' embeddings and then with embeddings we compute ourselves?

Separating the two isolates the possible error sources: a gap with the authors' embeddings would point to the evaluation, a gap that only appears with our embeddings to the embedding pipeline.

## Setup

As in [`methods.md`](methods.md): Euclidean distance, k = 1. Three runs on the same queries and lookup set:

1. EAT with the authors' ProtT5 embeddings (Zenodo).
2. EAT with our ProtT5 embeddings (`embed_prott5.py`, fp16, mean pooling).
3. MMseqs2 best hit (`-s 7.5 -e 10`).

Command: `WORK=/tmp/ra62ler_pp2 bash run.sh` on the H100 server.

Run time: our embeddings take 200 s for the 69,605 lookup proteins and 1 s for the 219 queries (plus about 25 s to load the model). The whole script needs about 15 minutes from scratch, most of it to download and unpack the 5.9 GB of authors' embeddings.

## Results

Accuracy in %, ± 95 % CI. Data: [`results/cath_test219.csv`](../experiments/01_prott5_cath_reproduction/results/cath_test219.csv).

| Method | C | A | T | H |
| --- | --- | --- | --- | --- |
| EAT, authors' ProtT5 | 83.6 ± 4.9 | 66.7 ± 6.2 | 56.7 ± 6.7 | 64.0 ± 7.7 |
| EAT, our ProtT5 | 83.6 ± 4.9 | 66.7 ± 6.2 | 56.7 ± 6.7 | 64.0 ± 7.7 |
| MMseqs2, best hit | 51.6 ± 6.6 | 36.1 ± 6.4 | 29.0 ± 6.1 | 34.7 ± 7.6 |
| Paper, EAT ProtT5 | 84 ± 5 | 67 ± 6 | 57 ± 6 | 64 ± 8 |
| Paper, MMseqs2 | 52 ± 7 | 36 ± 6 | 29 ± 6 | 35 ± 6 |
| Queries counted | 219 | 219 | 210 | 150 |

Both EAT runs and the MMseqs2 run match Table 1 after rounding.

**Our vs the authors' embeddings** ([`embeddings_ours_vs_authors.csv`](../experiments/02_table1_plms/results/checks/embeddings_ours_vs_authors.csv)), over all 69,824 proteins:

- Mean cosine similarity 0.999995; 69,821 proteins above 0.999.
- The nearest neighbour is the same lookup protein for all 219 queries, which is why the accuracies are identical rather than just close.
- One clear outlier, `9pcyA00` (cosine 0.68), see Deviations.

**MMseqs2 hits** ([`mmseqs_best_hit_identity.csv`](../experiments/02_table1_plms/results/checks/mmseqs_best_hit_identity.csv)):

- 172 of 219 queries have at least one hit with E-value ≤ 10; the other 47 count as wrong.
- Median sequence identity of the best hit: 33 %. 65 of the 172 are at or below 30 %. The three best hits with 90 % identity or more are alignments of 11–13 residues with E-values between 0.3 and 2.1, i.e. chance matches, not real homologs.

## Interpretation

- **The evaluation is correct.** It is the authors' own code, and with their embeddings it gives their numbers. Every later experiment uses the same evaluation.
- **Our embedding pipeline is equivalent to theirs.** Near-identical vectors and the same neighbour for every query show that pooling, precision and tokenisation match. We can therefore compute embeddings for new pLMs with the same recipe and compare them fairly with the published ProtT5 numbers. The remaining tiny differences (cosine 0.9999 instead of 1) are what we would expect from fp16 arithmetic on different hardware and library versions; we did not test this further.
- **EAT beats sequence search by about 30 points at every level** (H: 64 vs 35 %). This is expected by construction: the lookup set was filtered so that no lookup protein has more than 20 % identity to a query by MMseqs2, so MMseqs2 can mostly only find chance hits (median 33 % identity over short alignments). The benchmark measures remote homology, where embeddings have an advantage. It does not show that EAT beats sequence search in general; that needs a benchmark where close homologs are present, binned by sequence identity.
- **H is higher than T** (64 vs 57 %) because different queries count at each level: 69 queries have no other member of their superfamily and are excluded at H, but only 9 are excluded at T. H is measured on a smaller subset (150 queries) whose superfamilies have at least one lookup member, and that subset is apparently easier. Accuracies at different levels are therefore not directly comparable.

## Deviations and problems

- **Corrupt sequence in lookup69k.** The FASTA entry for `9pcyA00` (in `lookup69k.fasta` and `train66k.fasta`) contains two NUL bytes after residue 99, followed by 115 more residues. CATH lists this domain as residues 1–99. We embedded the whole 216-character string; the authors' embedding is probably of the correct 99 residues. It never was a nearest neighbour, so it does not change any result here. To fix for later experiments: cut the sequence at the NUL bytes, or take sequences from CATH directly.
- **`eat.py` batching bug.** In the EAT repo, `get_embeddings_batch` closes the last batch only when `seq_idx == len(id2seq)`, which never happens, so the last partial batch of short sequences can be skipped. We did not use that function.
- **Masking does not catch every impossible query.** The paper says a query is excluded at a level if no lookup protein shares its label there. The code instead excludes it only if it is the only protein with that label in lookup ∪ test. At T, 11 queries have no lookup protein with their topology, but 2 of them share it with another test protein and stay in the count, so they are always wrong ([`masking.csv`](../experiments/02_table1_plms/results/checks/masking.csv)). Excluding them would raise T from 56.7 % (119/210) to 57.2 % (119/208). At H the two definitions agree (69 queries each). We keep the code's definition so our numbers stay comparable with the paper.
- **Bootstrap size.** The paper text says 1,000 resamples, the code uses 10,000. We follow the code.
- **MMseqs2 search mode.** The paper's Methods mention a profile search for the redundancy reduction, but the Table 1 legend says MMseqs2 "used single sequence search". We ran a single-sequence search, which reproduces their numbers.

## Open questions and next steps

- HMMER baseline (paper: 77 % at H, higher than every raw pLM) not run.
- The Ankh paper reports 73.3 % at H for raw ProtT5 on what should be the same setup. Our pipeline gives 64.0 %, so the difference must come from their setup; find out which setting reproduces 73 %.
- Bin queries by sequence identity of their closest lookup protein to show where EAT gains over MMseqs2.
