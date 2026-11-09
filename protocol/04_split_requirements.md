# 04: Requirements for the data split

- **Date:** 08.11.2026
- **Issue / PR:** #24, #25
- **Sources:** the five papers in `papers/`, the EAT reference code, the CATH and MMseqs2 documentation, and external literature on homology-aware splitting. Raw extraction notes with page and section pointers: [`experiments/04_split_requirements/notes/`](../experiments/04_split_requirements/notes/).
- **Status:** requirements agreed in writing, pending supervisor confirmation of the three open decisions

## Why this document exists

Every number this project reports depends on how the data is split. A split that leaks makes a weak pLM look strong; a split that is too strict measures something nobody cares about. The published work we build on does not describe its splits completely enough to copy, and none of it separates calibration from test, which is precisely what our research question 2 needs. So the split is not a preliminary step for us, it is a result in itself.

## The three roles

| Role | What it is | What it may be used for |
| --- | --- | --- |
| Lookup set | The labelled library. Annotations are copied from it. | Nothing is fitted on it; it is searched. |
| Calibration set | Queries with known labels, held out from the lookup. | Choosing distance measure, k, pooling and normalisation; fitting distance thresholds and the reliability score. |
| Test set | Queries with known labels, held out from everything. | The final numbers, computed once. |

## Requirements

Each requirement states the rule, the reason, the evidence, and how we check that we met it. "Query sets" means calibration and test together.

### A. Separation and leakage

**R1. Separate by sequence similarity, never by identifier alone.** Two domains with different IDs can be near-identical. *Evidence:* ProtTucker removes every lookup candidate above 20 % identity to a query; ContrasTED does the same against both its query sets. *Check:* no query has a lookup hit above the cut after the filter.

**R2. Filter the queries against the complete lookup set, not against cluster representatives.** Sequence similarity is not transitive: A can be far from the representative of a cluster and close to another member of it. *Evidence:* ProteinNet methodology; Teufel et al. 2023 on representative-only criteria. *Check:* run the search against every lookup sequence, before any clustering.

**R3. Write down the identity definition and the coverage mode.** "20 % identity" is ambiguous. MMseqs2's default output is a similarity score, not the fraction of identical aligned columns, and without an explicit `--cov-mode` a short domain matching inside a long chain can pass the filter. *Evidence:* MMseqs2 documentation; CATH uses BLAST identity with 60 % overlap of the longer sequence. *Check:* the exact command, including `--alignment-mode`, `--cov-mode` and `-c`, is in the committed script.

**R4. Use a length-aware similarity criterion in addition to a flat identity cut.** 20 % identity over 300 residues is a real relationship; over 15 residues it is noise. HVAL subtracts a length-dependent curve from the identity, `HVAL = PIDE - 480 * L^(-0.32 * (1 + e^(-L/1000)))` for alignment lengths 11 to 450, 19.5 above that. *Evidence:* ProtTucker filters test300 at HVAL > 0; HFSP gives the curve and its refit (770, 0.33, 28.4). *Check:* implemented and unit-tested in [`hval.py`](../experiments/03_calibration_set/hval.py); the three branches join continuously.

**R5. Filter in both directions, in a fixed order.** First remove lookup candidates similar to any query, then drop queries that still have a hit in the reduced lookup. *Evidence:* ProtTucker: 20 % filter on the lookup, then every test300 domain with HVAL > 0 removed, which cost 27 % of the queries (300 to 219). *Check:* record the count after each step.

**R6. Do not claim more separation than the method delivers.** A 20 % sequence filter does not prove that structural homologs are gone: CATH itself assigns homology below that level, using structure and HMM comparison. A topology-level claim would need profile-profile search near 10 %, as ProteinNet does. *Evidence:* CATH FAQ; ProteinNet methodology. *Check:* the protocol states what the filter guarantees and what it does not.

**R7. Make the calibration set as remote from the lookup as the test set.** A threshold fitted on easier queries is too permissive when applied to harder ones. Build both query sets by the same procedure from the same pool. *Evidence:* ProtTucker built lookup69k so that validation had "similar sequence properties" to the final evaluation; exchangeability of calibration and test is also the condition for any conformal guarantee. *Check:* compare the HVAL distributions of calibration and test against the lookup; they should look the same.

**R8. Calibration and test must also be non-redundant to each other.** Otherwise the threshold is partly fitted on the test material. None of the three Rost-lab papers states this rule, because none of them has a separate calibration set. *Check:* same filter, applied between the two query sets.

### B. Query selection and balance

**R9. At most one query per fine-grained label.** Otherwise large families dominate the score and the queries are not independent observations. *Evidence:* ProtTucker allows each superfamily at most once across test300 and val200; ContrasTED samples one domain per superfamily into disjoint validation and test sets. *Check:* count distinct labels equals count of queries.

**R10. Keep lookup families from being dominated by near-duplicates.** ProtTucker clusters families above 100 members at 95 % identity and 95 % coverage. *Check:* report the member-count distribution of the lookup set.

**R11. Report results by family size and by sequence identity to the lookup.** Accuracy rises with family size, and the interesting claim is what happens when no close relative exists. *Evidence:* ProtTucker's family-size strata; the project brief asks explicitly for identity bins. *Check:* both breakdowns appear in the result tables.

**R12. Decide the evaluable population from the lookup set alone.** A query can only be answered at a level if some lookup protein carries its label at that level. The EAT code instead marks a query unanswerable only when it is the single carrier of its label in lookup ∪ queries, which is not the same thing: at the T level, 210 queries pass that test while only 208 are actually reachable. *Evidence:* `Eval.mask_singletons` in `train_prottucker.py`, verified independently on our data. *Check:* the masks are computed from the lookup, frozen before any model is run, and the excluded IDs are published.

### C. Evaluation rules that are really split decisions

**R13. Compare full label prefixes, not single numbers.** Level T means the triple C.A.T, not the third integer on its own. A count that compares only the local integer inflates accuracy, and it is one of the few explanations we have for the 73 % that Ankh reports where ProtTucker reports 64 %. *Check:* unit test with a toy label set.

**R14. Once a coarse level is wrong, the finer levels are wrong.** *Evidence:* `Eval.add_sample`; ProtTucker's definition of accuracy per level. *Check:* covered by the same unit test.

**R15. Give every baseline the same split and the same filter.** The ProtTucker benchmark was redundancy-reduced with MMseqs2, so MMseqs2 is handicapped on it by construction; that has to be stated rather than presented as a win for embeddings. *Evidence:* the paper says the data set "had been optimized for minimal performance of MMseqs2". *Check:* the limitation is in our results text, and the MMseqs2 baseline runs on the same lookup.

**R16. Exclude self-hits and define tie handling.** *Evidence:* HFSP excludes self-matches; EAT takes a single top hit. *Check:* in the evaluation code.

### D. Calibration and thresholds

**R17. Fit every setting and threshold on the calibration set only, then freeze it and touch the test set once.** This is the rule the existing work breaks: ProtTucker reads its thresholds off test219, and goPredSim tuned on the CAFA3 targets and says so. *Evidence:* ProtTucker Figure 5; goPredSim's own statement that its F-max "arguably over-estimates performance". *Check:* the frozen thresholds are committed before the test run.

**R18. Fit one mapping per pLM, task and level.** Distances are on different scales between models, between tasks, and between levels. *Evidence:* goPredSim on reliability indices not being comparable across embeddings; our own measurement of mean embedding norms from 1.43 to 17.98. *Check:* no threshold is reused across configurations.

**R19. Report "no usable threshold" when that is the answer.** Frozen ProstT5 embeddings reached no cutoff at ContrasTED's 99 % precision target. Lowering the target afterwards to produce a number would be fitting on the test set. *Evidence:* ContrasTED results. *Check:* targets 90 %, 95 % and 99 % are stated in advance.

**R20. Report coverage together with precision.** A method can reach any precision by answering almost nothing. *Evidence:* ContrasTED reports 64.0 % coverage at 98.5 % precision; CAFA's coverage definition exists for the same reason. *Check:* every threshold row carries its coverage and its denominator.

### E. Label quality

**R21. State the evidence level of the labels in the lookup set.** Labels that were themselves transferred by homology inflate the score. *Evidence:* goPredSim's experimental-only variants score differently from the full sets. *Check:* recorded per data set. For CATH this is uncritical; for EC it is the central problem, see the EC section.

**R22. Record a residual-redundancy curve.** Re-run the evaluation while progressively removing lookup proteins above a decreasing similarity to each query. A flat curve is evidence that the split does not leak. *Evidence:* ProtTucker Figure 6; goPredSim's nine identity steps from 100 % to 20 %. *Check:* one figure per task, produced from the committed script.

### F. EC only

The EC task has no ready-made split, and its labels are the weakest point of the whole project. Full notes: [`ec-split-requirements.md`](../experiments/04_split_requirements/notes/ec-split-requirements.md).

**R23. Judge evidence per reaction, not per entry.** UniProt attaches evidence codes to each catalytic activity, so one entry can carry a measured EC number and an inferred one. Being in Swiss-Prot, being reviewed, or having experimental evidence of protein existence says nothing about whether a given EC number was measured. HFSP's stricter rule, one complete four-level EC with experimental evidence of existence plus a BRENDA annotation, is the conservative alternative. *Check:* the builder reads ECO codes per catalytic activity and records which rule produced each label.

**R24. Keep multiple EC numbers, and treat a dash as unknown rather than wrong.** A query may legitimately have several EC numbers, and `3.4.-.-` is an incomplete label, not a wrong one. Scoring an unknown level as an error measures annotation completeness instead of transfer quality. *Check:* the metric is defined on label prefixes at known levels, and the multi-label rule is written down before any run.

**R25. Split EC by time as well as by sequence.** Entries annotated after a chosen release are an untuned confirmation set and reflect the real use case, predicting something new. Sequence filtering alone is weak here because sequence identity and EC agreement are only loosely coupled. *Evidence:* HFSP's post-2002 hold-out; goPredSim's GOA2017 to GOA2020 comparison; the CARE benchmark's split design. *Check:* the release and cut-off date are pinned with the ID manifests.

**R26. Report macro as well as micro numbers, with support per class.** EC is extremely imbalanced; a micro average is dominated by a few huge classes. HFSP caps class sizes when fitting for this reason. *Check:* both averages and the per-class support appear in the tables.

### G. Reproducibility

**R27. Freeze three explicit ID manifests plus the seed, the database release and a checksum.** Never let a file name or a prose description define membership. *Check:* committed lists; counts after every filter step recorded in this protocol.

**R28. Verify that embeddings exist for every ID in every split, for every pLM.** A silently missing embedding redefines the split per model. We already hit this: the authors' ProtBERT file is missing two lookup proteins. *Check:* the evaluation asserts ID coverage and fails loudly.

## Where the published work falls short

Worth saying out loud in the presentation, because it is the gap our benchmark fills.

- **No separate calibration set anywhere.** ProtTucker, goPredSim and HFSP all choose thresholds on the data they then report. ContrasTED is the exception and does it correctly, but only for its own trained model.
- **Incomplete method descriptions.** ProtTucker never says which tool computed its HVAL values, and gives no coverage parameter for the 20 % filter. Ankh does not describe its EAT evaluation well enough to explain its 73 % against ProtTucker's 64 % on nominally the same data.
- **Unstated identity definitions.** None of the papers says whether "identity" means identical columns over the alignment, over the shorter sequence, or MMseqs2's similarity score.
- **A masking rule in the code that does not match the text.** See R12.
- **Thresholds presented as portable.** ProtTucker's distance 0.9 is quoted for a 5 % error rate, with the paper itself noting it could be up to 15 %.

## Open decisions

1. **CATH classes.** ContrasTED uses classes 1 to 3; the ProtTucker benchmark uses everything. Proposal: keep all classes and report class 4 and 6 separately if they behave differently.
2. **How strict to make the separation.** Profile search at 20 % identity, as both reference papers use, or profile-profile search near 10 %, which is the only recipe we found that genuinely supports a topology-level claim. Proposal: 20 % profile search for the headline split, because it keeps us comparable with both papers, plus an honest statement of what it does not guarantee. Revisit if the supervisors want a topology claim.
3. **Threshold method.** A fitted precision-versus-distance curve, or conformal risk control, which gives a finite-sample bound on the precision loss under the same exchangeability assumption we need anyway. Proposal: start with the simple empirical rule ContrasTED uses, the largest distance that still meets the target precision on calibration, and add conformal risk control afterwards if time allows.
4. **EC label rule.** Experimental evidence required for every reaction label, which is strict and small, or HFSP's rule of one complete EC with experimental protein existence plus BRENDA, which is larger and comparable with HFSP. Proposal: build the strict set and keep the HFSP set as a comparison, but this needs the supervisors.

## Recipe that follows from the requirements

1. CATH v4.4, S100 sequences and the S100 classification file. 123,251 domains in 6,630 superfamilies; 4,339 superfamilies have at least two members and can therefore contribute a query (R9, R12).
2. Sample one domain per eligible superfamily into a calibration set and a test set, disjoint, with a fixed seed (R9, R23).
3. Remove from the remaining domains everything with at least 20 % identity to any calibration or test domain, with MMseqs2 iterative profile search (`--num-iterations 3 -s 7.5 --cov-mode 0`), identity mode and coverage written into the command, run against the full set (R1, R2, R3, R5).
4. Drop queries that still have a hit at HVAL > 0 in the reduced lookup, and queries whose superfamily lost all its lookup representatives (R4, R5, R12).
5. Apply the same filter between calibration and test (R8).
6. Freeze the three ID lists, the seed, the CATH release and the checksums; record the counts after every step (R27).
7. Verify: HVAL distributions of calibration and test against the lookup are comparable (R7); one query per superfamily (R9); no embedding is missing (R28).

Keep `lookup69k` and `test219` unchanged as the reproduction benchmark. The two never mix.
