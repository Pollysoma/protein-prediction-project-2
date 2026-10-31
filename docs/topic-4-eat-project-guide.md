# Topic 4: Embedding-Based Annotation Transfer (EAT)

This document explains the project described in [the full Topic 4 brief](./4_eat_baseline.pdf). It starts with the necessary biology and machine-learning ideas, then turns the brief into a practical project plan.

## The project in one paragraph

Many proteins have a known amino-acid sequence but no experimentally confirmed structure or function. A common way to annotate such a protein is to find a similar, already annotated protein and copy its label. Embedding-based annotation transfer (EAT) measures similarity using vectors made by a protein language model instead of using only sequence alignment. Our task is to compare several current protein language models and distance measures, determine when their transferred labels can be trusted, and build a reproducible benchmark that future models can be tested against. We will study two kinds of labels: protein structure through CATH and enzyme function through EC numbers.

The kickoff protocol also records a practical downstream use: the resulting model-specific distance thresholds should be suitable for integration into the supervisor's ProtSpace visualization tool.

## 1. Essential biology

### Proteins and amino-acid sequences

A protein is a chain made from amino acids. Its **sequence** is the ordered list of those amino acids, usually written as letters such as `MKT...`. The sequence influences how the chain folds into a three-dimensional structure, and structure is strongly connected to what the protein does.

Two proteins can be evolutionarily related and perform similar jobs even when many positions in their sequences have changed. This makes annotation difficult: simple sequence matching works well for close relatives, but can miss remote relationships.

### How proteins and enzymes are related

**Protein** describes a type of molecule, while **enzyme** describes a job: speeding up a chemical reaction. Most enzymes are proteins, but many proteins have other roles, such as providing structure, transporting molecules, sending signals, or supporting the immune system. A useful shortcut is therefore: most enzymes are proteins, but not all proteins are enzymes. There are also exceptions called ribozymes, which are RNA molecules that act as enzymes. In this project, the EC task concerns proteins that function as enzymes.

### Annotation transfer

An **annotation** is a label describing a known property of a protein. Annotation transfer works like this:

1. Start with an unlabelled query protein.
2. Search a collection of labelled proteins, called the **lookup set**.
3. Find the lookup protein most similar to the query.
4. Copy that protein's label to the query.

The copied label is a prediction, not new experimental evidence. Its quality depends on the similarity method and on whether the lookup set contains a genuinely related protein.

### The two label systems in this project

#### CATH: structural classification

CATH organizes protein domains by their three-dimensional structure. A **domain** is a part of a protein that can often fold and function as a unit. CATH has four levels, from broad to specific:

- **C — Class:** broad secondary-structure composition.
- **A — Architecture:** overall arrangement of secondary structures.
- **T — Topology:** how structural elements connect and fold.
- **H — Homologous superfamily:** domains believed to share a common ancestor.

A prediction can therefore be correct at a broad level and wrong at a more specific one. Accuracy must be reported separately for C, A, T, and H.

#### EC numbers: enzyme-function classification

Enzymes speed up chemical reactions. An **EC number** describes the reaction catalysed by an enzyme and contains four increasingly specific levels, for example `EC 3.4.21.4`. A partial label such as `EC 3.4.-.-` specifies only the first two levels. However, the degree of functional specificity at the same EC level can vary between branches, so two fourth-level labels do not always represent the same degree of biological difference.

Function labels require special care. A protein can catalyse multiple reactions, labels can be incomplete, and many database annotations are inferred rather than measured. The benchmark needs clear rules about which EC labels count as trustworthy ground truth and how multi-label or incomplete cases are evaluated.

## 2. Essential machine-learning ideas

### Protein language models

A protein language model (pLM) learns statistical patterns from large collections of amino-acid sequences. Its training resembles language-model training: for example, it may learn to reconstruct hidden amino acids from their context. In doing so, it can learn information related to evolution, structure, and function without receiving CATH or EC labels for this project.

When a sequence is passed through a pLM, the model produces a numerical vector for every amino acid. We average the vectors across the sequence to obtain one fixed-length vector for the protein. This vector is its **embedding**. Proteins placed near one another in embedding space may share biological properties even if their raw sequences are not obviously similar.

### What “raw” and “frozen” mean here

This project evaluates **raw embeddings** under a strict definition:

- the pretrained pLM is frozen; its parameters are not changed;
- embeddings come from the model's last layer;
- residue embeddings are averaged to produce one vector per protein;
- no classifier, prediction head, probe, fine-tuning, contrastive learning, or metric learning is trained on top;
- a query receives the label of its nearest lookup neighbor or neighbors.

Label-free processing such as centering embeddings is allowed, but it must be evaluated fairly. Models pretrained using annotations, such as ESM3, must be reported separately because they had access to different information during pretraining.

This restriction makes the comparison easier to interpret: differences mainly show how usefully each pLM arranges proteins in its embedding space, rather than how good an additional trained classifier is.

### Nearest neighbors and distance

For each query embedding, EAT finds the closest embedding in the lookup set. The project compares at least:

- **Euclidean distance:** straight-line distance between two vectors;
- **cosine distance:** difference in vector direction, mostly ignoring vector length.

The two measures can select different neighbors because Euclidean distance depends on both direction and vector magnitude, while cosine distance depends mainly on direction. Embedding dimensions and scales also differ between models, so the same raw distance does not have the same meaning for every pLM.

### Why a reliability score is needed

For transfer from a single neighbor (`k = 1`), the original EAT/goPredSim reliability index reduces to:

```text
RI = 0.5 / (0.5 + d)
```

This gives 1 for identical embeddings and decreases as distance grows. However, the formula was designed around an older model, and distances from different pLMs have different scales. An RI of 0.9 therefore does not automatically mean the same thing for ProtT5, ESM-2, or Ankh.

We need to calibrate an updated score. Ideally, a reported reliability of 0.9 should mean that about 90% of comparable transfers are correct. The mapping may need to differ by pLM, task, and label level even if the final score has a common interpretation.

## 3. Research questions

The project has two main questions:

1. **Which pLM and distance measure transfer annotations best?** Compare structure transfer for CATH and function transfer for EC. Also determine where embedding-based transfer performs better than ordinary sequence-similarity search.
2. **When is a transfer safe?** Find distance thresholds that reach stated precision targets, such as at least 95%, and turn distance into a well-calibrated reliability score.

These questions are related but not identical. A model can have high overall accuracy yet lack a useful high-confidence region, or it can make fewer predictions while being very reliable on those it accepts.

## 4. What the benchmark must contain

### Models

The minimum comparison listed in the project brief is:

- ProtT5;
- ProstT5;
- ESM-2 at 8M, 35M, 150M, 650M, and 3B parameters;
- ESM C at 300M and 600M parameters;
- Ankh-base, Ankh-large, and Ankh3-large.

Each checkpoint and revision must be recorded exactly. All embeddings for one pLM must be produced with the same software path. In particular, the Biocentral API must not be used for ESM C because the project brief reports that it currently returns incorrect ESM C embeddings.

### Data

For **CATH**, begin with the established ProtTucker benchmark containing the `lookup69k` lookup set and the `test219` query set. The first reproducibility check is raw ProtT5 on `test219`; the expected C/A/T/H accuracies from the brief are 84/67/57/64%. In the ProtTucker paper, a query was excluded at a given CATH level if its correct label was absent from `lookup69k`; this level-specific reachability rule must be reproduced for those numbers, and any later benchmark must state whether unreachable queries are excluded or counted as incorrect. A mismatch should be investigated before scaling up. The precomputed CATH-S100 lookup in the EAT repository must not be used for this test because it contains the test domains and would leak answers into the lookup set. After the small benchmark works, use the larger non-redundant CATH v4.4 domain sets.

For **EC**, no finished split is provided. The brief suggests experimentally supported catalytic-activity entries from UniProtKB/Swiss-Prot and the older HFSP enzyme set as starting points. The team must define and agree with the supervisors on:

- which evidence qualifies an EC label as ground truth;
- how lookup, calibration, and test proteins are separated;
- whether a time-based split using older UniProt releases is appropriate;
- how incomplete EC numbers and proteins with multiple EC numbers are handled;
- how sequence redundancy or close homologs across splits are controlled.

Once agreed, the splits must be versioned and frozen rather than changed after seeing test performance.

### Baselines

At minimum, EAT must be compared with transfer from the best MMseqs2 sequence-search hit. Results should be grouped by sequence identity to the lookup set, because EAT is especially interesting when no close sequence match exists. The team must agree on any additional useful baselines.

Methods trained on embeddings, such as ProtTucker, ContrasTED, or CLEAN, answer a different question from raw EAT. They can be shown as reference points, but their published results should only be compared with the raw baseline from the same paper and experimental setup.

### Metrics and uncertainty

The minimum outputs are:

- accuracy at every CATH and EC level;
- confidence intervals for every reported number, using a reproducible method such as bootstrap resampling with a fixed seed;
- results binned by sequence identity to the lookup set;
- distance thresholds for each pLM, task, and level at explicitly stated precision targets;
- test-set checks of thresholds chosen on calibration data;
- an updated, calibrated reliability score.

It is useful to report **coverage** with each precision threshold: coverage is the fraction of queries close enough to receive an accepted prediction. Otherwise, a system could obtain very high precision simply by making almost no predictions.

## 5. The most important experimental rule

The data must have three distinct roles:

```text
lookup set     labelled reference proteins searched by EAT
calibration set used to choose distances, settings, thresholds, and reliability mapping
test set       used once for the final unbiased evaluation
```

Do **not** use the test set to choose distance measures, preprocessing, number of neighbors, thresholds, precision targets, or the reliability formula. Repeatedly checking test results while changing the method would indirectly train the benchmark on its test set and make the final numbers too optimistic.

The lookup set is also different from a training set: raw EAT does not optimize model parameters from it. It is the labelled library from which annotations are copied.

## 6. Recommended project workflow

### Phase 1: Understand and reproduce the existing baseline

1. Read the EAT, ProtTucker, and HFSP papers listed in the project brief.
2. Obtain the EAT code and the correct `lookup69k`/`test219` data.
3. Reproduce raw ProtT5 on CATH `test219`.
4. Check the expected 84/67/57/64% C/A/T/H values. If they differ, inspect sequence preprocessing, model checkpoint, pooling, lookup contents, distance calculation, and tie handling.

This small experiment validates the pipeline before expensive embeddings are computed.

### Phase 2: Write the protocol before running the full benchmark

Decide with the supervisors and document:

- exact data sources, versions, filters, and checksums;
- lookup/calibration/test split construction;
- trusted EC evidence and multi-label evaluation rules;
- exact pLM checkpoints, revisions, and embedding software;
- sequence-length and multi-domain handling;
- Euclidean/cosine preprocessing, including any centering or normalization;
- one-neighbor versus multiple-neighbor transfer and tie handling;
- metrics, confidence intervals, sequence-identity bins, and random seeds;
- calibration method, target precision levels, and minimum sample requirements.

Freeze the protocol and splits after approval. If a later correction is unavoidable, version and explain it rather than silently replacing it.

### Phase 3: Build and validate the datasets

1. Build the larger CATH benchmark without overlap or leakage.
2. Build the EC dataset using the agreed experimental-evidence rules.
3. Verify that every sequence has the intended labels and split membership.
4. Measure sequence similarity from every query to the lookup set.
5. Produce summary tables for label frequencies, sequence lengths, identity ranges, and exclusions.

### Phase 4: Compute embeddings consistently

1. Pin every model by exact checkpoint and revision.
2. Test a small batch first, especially for the large EC lookup set.
3. Save the mean last-layer embedding and enough metadata to reproduce it.
4. Check vector shapes, missing values, duplicates, and sequence-to-embedding IDs.
5. Record runtime, hardware, batch size, software versions, and failures.

Embeddings and datasets are large and should not be committed to Git. Store them locally under ignored paths or publish them through an appropriate data repository. Commit the scripts, metadata, and checksums needed to rebuild or download them.

Nearest-neighbor search over a large lookup set can also become a computational bottleneck. Measure exact-search runtime first; if an approximate index, hashing method, or vector database is later needed, verify its results against exact neighbors on a manageable subset and document the trade-off.

### Phase 5: Select settings using calibration data only

For each pLM and task:

1. Search the lookup set using Euclidean and cosine distance.
2. Evaluate candidate settings on the calibration queries.
3. Choose the setting according to a rule written in advance.
4. Estimate distance cutoffs for each label level and precision target.
5. Fit or construct the distance-to-reliability mapping.
6. Inspect whether calibration remains valid across distance ranges, sequence-identity ranges, and important data subgroups.

### Phase 6: Run the final test once settings are frozen

Apply the frozen choices to the untouched test set. Report:

- accuracy and confidence intervals by task and hierarchy level;
- precision and coverage at each calibrated distance cutoff;
- reliability calibration, for example predicted reliability versus observed correctness;
- performance by sequence-identity bin;
- the MMseqs2 baseline under the same split;
- failures, limitations, and cases where sequence search remains better.

### Phase 7: Package a reusable benchmark

The final repository should let another researcher score a new pLM in exactly the same way. It therefore needs documented commands, frozen split identifiers, pinned dependencies and model revisions, deterministic evaluation scripts, checksums, and a clear result format. Every published number should be reproducible from committed code and a recorded seed.

## 7. Common mistakes to avoid

- **Data leakage:** a test protein, duplicate, or forbidden close copy appears in the lookup or calibration data.
- **Tuning on the test set:** settings improve after repeatedly looking at final test results.
- **Mixing embedding pipelines:** embeddings from the same pLM are generated by different tools or settings.
- **Comparing raw distances across pLMs:** a distance of 0.2 need not have the same meaning for two models.
- **Ignoring hierarchy:** only the most specific CATH or EC level is reported.
- **Ignoring coverage:** a high-precision threshold is reported without saying how many queries it accepts.
- **Trusting every EC database label equally:** inferred and experimentally supported reactions are mixed without a rule.
- **Changing frozen splits silently:** results from different split versions become impossible to compare.
- **Reporting point estimates only:** accuracy is given without confidence intervals or sample counts.
- **Overstating annotation transfer:** a predicted label is described as if it were experimental proof.

## 8. What success looks like

At the end of the project, we should be able to answer questions such as:

- Which raw pLM is best for CATH and which is best for EC?
- Does the answer change between broad and specific label levels?
- When does cosine distance outperform Euclidean distance?
- How does EAT compare with MMseqs2 for close and remote sequence relationships?
- For a chosen model and task, how close must a neighbor be for at least 95% precision?
- If a transfer has reliability 0.9, is it actually correct about 90% of the time on unseen data?
- Can another team add a new pLM and obtain a fair, directly comparable result?

The main deliverable is therefore not just a leaderboard. It is a transparent yardstick: fixed data, fixed rules, calibrated confidence, honest uncertainty, and code that applies the same evaluation to every model.

## 9. Short glossary

- **Accuracy:** fraction of all evaluated predictions that are correct.
- **Calibration:** agreement between stated confidence and observed correctness.
- **Confidence interval:** a range expressing uncertainty around an estimated metric.
- **Coverage:** fraction of queries for which the method accepts a prediction.
- **Embedding:** numerical vector representing a protein sequence.
- **Ground truth:** label treated as correct for evaluation, ideally backed by strong evidence.
- **Lookup set:** labelled proteins searched for nearest neighbors.
- **Precision:** among accepted predictions, the fraction that are correct.
- **pLM:** protein language model.
- **Sequence identity:** fraction of aligned amino acids that are the same.

## References and supplied material

- [Topic 4 full project description](./4_eat_baseline.pdf).
- [PP2 project overview](./Projects%20PP2%20-%20WS26_27-2.pdf), especially slide 24.
- [PP2 presentation guidance](./PP2%20Intro%20How%20to%20present%20-%20WS26_27.pdf).
- Littmann et al. (2021), “Embeddings from Deep Learning Transfer GO Annotations beyond Homology.” [Scientific Reports](https://doi.org/10.1038/s41598-020-80786-0).
- Heinzinger et al. (2022), “Contrastive Learning on Protein Embeddings Enlightens Midnight Zone.” [NAR Genomics and Bioinformatics](https://doi.org/10.1093/nargab/lqac043).
- Mahlich et al. (2018), “HFSP: High Speed Homology-Driven Function Annotation of Proteins.” [Bioinformatics](https://doi.org/10.1093/bioinformatics/bty262).
