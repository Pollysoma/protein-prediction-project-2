# Project overview

**EAT: a benchmark for annotation transfer with raw pLM embeddings.** Project 4, Protein Prediction II, WS 2026/27, Chair of Bioinformatics (RostLab), TUM.

Sources: the project description `4_eat_baseline.pdf`, the course slides `Projects PP2 - WS26_27.pdf`, and the kickoff meeting on 28.10.2026 (protocol in #3). The course PDFs live on Moodle and are not committed here.

## Idea

To annotate an uncharacterised protein, you usually copy the label of a similar, characterised one. Embedding-based annotation transfer (EAT) does this in the embedding space of a protein language model (pLM): a frozen pLM turns each protein into one vector, and a query inherits the label of its nearest neighbour in a labelled lookup set. goPredSim turned the Euclidean distance `d` into a reliability index `RI = 0.5 / (0.5 + d)`.

Results so far:

- Without training, EAT transferred GO terms better than BLAST (Littmann 2021).
- On CATH remote homologues, raw ProtT5 reached 64 % at superfamily level, against 35 % for MMseqs2 and 77 % for profile HMMs (Heinzinger 2022).

## Why a benchmark is needed

- Raw EAT is reported as a baseline in many papers, but each one sets it up differently (pLM, pooling, distance, k, lookup set). Raw ProtT5 scores 64 % in one paper and 73 % in another on the same CATH test219 and lookup sets.
- The choice of pLM matters: ProtT5 → ProstT5 raises CATH-H from 69.6 to 78.5 % on a strict benchmark (ContrasTED 2026).
- The RI cutoffs were set for SeqVec, and distances from different pLMs live on different scales.
- HFSP gives a safe-transfer curve for sequence alignments (96 % precision for EC 1–3, 44 % for all four levels). Nothing like it exists for embeddings.

## Research questions

1. Which pLM and which distance (Euclidean or cosine) transfer structure (CATH) and function (EC) best, and where does EAT beat plain sequence similarity?
2. Up to which distance is a transfer safe (e.g. ≥ 95 % precision), for structure and for function? Can distance become a reliability score that means the same for every pLM (0.9 = 90 % of such transfers are correct)?

## Scope: what "raw" means

- The pLM stays frozen. One vector per protein is the mean of the last-layer residue embeddings; layer and pooling are fixed.
- Nothing is trained on top: no contrastive or metric learning, no probes, no fine-tuning. Label-free steps such as centring are allowed.
- A query takes the label of its nearest neighbour(s).
- Labels may be used on a calibration set to choose settings, set thresholds and fit the reliability score, **never on the test set**.
- pLMs pre-trained with annotations (e.g. ESM3) form a separate group. Learned methods (ProtTucker, ContrasTED, CLEAN) are reference points: compare their numbers only with the raw baseline from the same paper.

## Minimum result

For ProtT5, ProstT5, ESM-2 (8M, 35M, 150M, 650M, 3B), ESM C (300M, 600M), Ankh-base, Ankh-large and Ankh3-large:

- accuracy per level on CATH and EC, with confidence intervals, next to transfer from the best MMseqs2 hit, binned by sequence identity to the lookup set;
- distance thresholds per pLM, task and level at the stated precision(s), checked on test data;
- an updated reliability score;
- documented code and frozen splits that score any new pLM or method the same way.

| Family | Models | Embedding dim |
| --- | --- | --- |
| ProtT5 | ProtT5-XL-U50, ProstT5 | 1024 |
| ESM-2 | 8M, 35M, 150M, 650M, 3B | 320, 480, 640, 1280, 2560 |
| ESM C | 300M, 600M | 960, 1152 |
| Ankh | base, large, Ankh3-large | 768, 1536, 1536 |

## Data

**CATH (structure; levels C/A/T/H).** Start with the ProtTucker benchmark (lookup69k, test219) and the [EAT code](https://github.com/Rostlab/EAT); its data is mirrored on [Zenodo](https://zenodo.org/records/14675997). The pre-computed CATH-S100 lookup in that repo contains all test219 domains, so never use it as the lookup for them. Then move to the larger [CATH v4.4 non-redundant sets](http://download.cathdb.info/cath/releases/all-releases/v4_4_0/non-redundant-data-sets/).

**EC (function; four levels, e.g. 3.4.21.4).** There is no ready-made split. Two starting points:

- UniProtKB entries with experimental evidence for their catalytic activity: `(ec:*) AND (cc_catalytic_activity_exp:*)` finds 48,401 entries in release 2026_03, all in Swiss-Prot (which has 279,671 entries with an EC number). Evidence is given per reaction, so one entry can mix measured and inferred EC numbers.
- The HFSP set: 7,022 Swiss-Prot enzymes (release 2017_09) with one complete EC number, experimental evidence of existence and a BRENDA annotation (Supplementary Data 1 of the paper). It allows a direct comparison with HFSP. Older UniProt releases for time-based splits are on the [UniProt FTP](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/).

**Embeddings.**

- The ProtSpace embedding notebook covers every pLM above except ESM-2 3B. For that one, load `facebook/esm2_t36_3B_UR50D` with `transformers`.
- The [biocentral API](https://github.com/biocentral/biocentral_api) is an alternative, but it currently returns wrong ESM C embeddings, so do not use it for ESM C.
- Compute all embeddings of one pLM with the same tool.
- Time a small batch first: the EC lookup set is the compute bottleneck.
- Fast nearest-neighbour search at scale is a known challenge (exact search with FAISS, vector databases, hashing).

**Sharing.** Code goes on GitHub. Data (a few GB) goes on Zenodo (preferred), Figshare or Hugging Face.

## Steps

1. **Get started.** Read the EAT, ProtTucker and HFSP papers. Reproduce raw ProtT5 on test219: target 84 / 67 / 57 / 64 % at C/A/T/H. If our numbers differ, find out why.
2. **Design a fair benchmark.** Draft answers to the open questions below, agree on splits and protocol with the supervisors, then freeze them.
3. **Answer the research questions** with the frozen protocol, then go beyond the minimum (e.g. normalising embeddings, or where and why EAT fails).

### Open design questions

- How do we split lookup and query proteins so that results reflect real use? How similar are queries to the lookup set?
- Which EC labels can we trust as test labels?
- ESM-2 8M → 3B: is a larger model better because it knows more, or because its embeddings have more dimensions?
- When can Euclidean and cosine distance pick different neighbours at all? (On unit-length vectors they rank identically, so differences come from vector norms.) How do we compare pLMs whose distances live on different scales?
- What happens with long or multi-domain proteins, and with enzymes that have several or incomplete EC numbers?
- Besides the best MMseqs2 hit, which baselines must EAT beat to be useful?

## Course logistics

- Lecture/project slot: Monday 10:15–11:45, room 5608.EG.038. Contact: pp-org@rostlab.org.
- Presentations every third week: **02.11** (pitch), **23.11**, then every three weeks. Each is a 10-minute talk plus 5 minutes of questions, all seven groups from 10:00 sharp. Guidelines: [`presenting.md`](presenting.md).
- Each talk answers: what we worked on and which part of the project it serves, our biggest success, the current roadblock, the next steps and why, and how we evaluated progress.
- Supervisor meetings are on Wednesdays, with none in a week that has a Monday presentation unless there are results to discuss.
- Exam (provisional): in project groups; a 15-minute presentation plus 10 minutes of questions per student; a 0.3 grade bonus.
- LLM use is encouraged, but every one of us must understand and be able to explain everything we present.
- A computational biology group works on the same project, and we are encouraged to exchange with them.

## References

1. M. Littmann … B. Rost. Embeddings from deep learning transfer GO annotations beyond homology. *Sci Rep* 11, 1160 (2021). https://doi.org/10.1038/s41598-020-80786-0
2. M. Heinzinger … B. Rost. Contrastive learning on protein embeddings enlightens midnight zone. *NAR Genom Bioinform* 4(2), lqac043 (2022). https://doi.org/10.1093/nargab/lqac043
3. D. Miller … C. Orengo. ContrasTED: Contrastive domain embeddings for scalable remote homology classification. *bioRxiv* (2026). https://doi.org/10.64898/2026.08.28.747864
4. A. Elnaggar … B. Rost. Ankh: Optimized protein language model unlocks general-purpose modelling. *arXiv* 2301.06568 (2023). https://doi.org/10.48550/arXiv.2301.06568
5. Y. Mahlich … Y. Bromberg. HFSP: High speed homology-driven function annotation of proteins. *Bioinformatics* 34(13), i304–i312 (2018). https://doi.org/10.1093/bioinformatics/bty262
6. A. Elnaggar … B. Rost. ProtTrans: Toward understanding the language of life through self-supervised learning. *IEEE TPAMI* (2022).
