# Methods shared by all experiments

## Benchmark: CATH test219 vs lookup69k

From Heinzinger et al. 2022 (ProtTucker, *NAR Genom Bioinform* 4:lqac043), built from CATH v4.3.0 (CATH-S100, about 123k domains).

- **test300**: 300 domains split off at random from CATH-S100, with every homologous superfamily at most once across test300 and val200 and every domain annotated with a CATH structural sub-group. **test219**: the 219 of them with HVAL ≤ 0 to every lookup protein, i.e. no detectable sequence similarity. This is the query set.
- **lookup69k**: 69,605 CATH-S100 domains. Every domain with more than 20 % pairwise sequence identity to a test or validation protein was removed (MMseqs2 iterative profile search, `--num-iterations 3 -s 7.5 --cov-mode 0`); families with more than 100 members were clustered at 95 % identity. This is the set labels are transferred from.
- Both sets are FASTA files in [Rostlab/EAT](https://github.com/Rostlab/EAT) (`data/ProtTucker/`). Neither file contains identical sequences, so the duplicate removal in the evaluation code removes nothing.
- **Labels**: `cath-domain-list.txt` (CATH v4.3), columns 2–5 give the class (C), architecture (A), topology (T) and homologous superfamily (H). MD5 `6df99c76b576d7fda8c47f0d43187ce1`.

Because the lookup set was redundancy-reduced against the queries with MMseqs2, the benchmark measures remote homology and is the worst case for MMseqs2 by construction.

## Evaluation

We reuse `DataSplitter` and `Eval` from `train_prottucker.py` in Rostlab/EAT (commit `b82ea03`) unchanged and only call them ([`evaluate.py`](../experiments/01_prott5_cath_reproduction/evaluate.py)).

- **Transfer**: each query takes the CATH label of its nearest lookup protein (k = 1). Distance is Euclidean unless stated otherwise. For cosine distance we scale every embedding to unit length and keep the Euclidean search, which picks the same neighbour.
- **Hierarchical correctness**: a level counts as correct only if this level and all coarser levels match. Once C is wrong, A, T and H are wrong.
- **Singleton masking**: if the query is the only protein with its label at a level within lookup ∪ test, that level is excluded for this query (and the coarser levels, if they are singletons too). As a result, the number of queries differs per level: C 219, A 219, T 210, H 150. This is why accuracy at H can be higher than at T.
- **Accuracy** per level = correct / counted queries.
- **Confidence interval**: 10,000 bootstrap resamples of the per-query correct/wrong list, CI = 1.96 × their standard deviation, seed 42. The paper text says 1,000 resamples; the code uses 10,000.

## Embeddings

- **Authors' embeddings**: Zenodo record [14675997](https://zenodo.org/records/14675997), `prottucker_training_embeddings.tar.gz` (MD5 `170a4ca81e258371c9afabbdfaa438e4`), with one 1-D vector per CATH-S100 domain for ProtT5, ESM-1b, ProtBERT, ProSE-DLM and ProSE-MT. The files cover all CATH-S100 domains, including the test219 queries; only lookup69k IDs are used as lookup.
- **Our ProtT5 embeddings** ([`embed_prott5.py`](../experiments/01_prott5_cath_reproduction/embed_prott5.py)): encoder of `Rostlab/prot_t5_xl_half_uniref50-enc` (Hugging Face revision `94a6abc`), fp16. Rare amino acids U, Z, O, B become X; residues are separated by spaces; the tokenizer appends `</s>`. One vector per protein is the mean over the last-layer states of the residues, without `</s>` and padding. Batches of up to 8,000 residues, sorted by length.

## MMseqs2 baseline

MMseqs2 release `21fbeafa`, single-sequence search of test219 against lookup69k with `-s 7.5 -e 10` (highest sensitivity, hits up to E-value 10, as in the paper). Each query takes the label of the hit with the lowest E-value (ties broken by bit score). Queries without a hit count as wrong at every level that is not masked. Scoring uses the same masking and hierarchy rules as EAT.

## Compute and software

- GPU server `aranuka` (LMU), one NVIDIA H100 PCIe 80 GB; scoring of precomputed embeddings also runs on a laptop CPU.
- Python 3.10 (server) / 3.11 (laptop), torch 2.5.1, transformers 4.46.3, sentencepiece 0.2.0, h5py 3.12.1, numpy 1.26.4. Pinned in [`requirements.txt`](../experiments/01_prott5_cath_reproduction/requirements.txt).
