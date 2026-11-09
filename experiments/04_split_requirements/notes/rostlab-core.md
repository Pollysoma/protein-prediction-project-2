# Split requirements: RostLab core papers (ProtTucker/EAT, goPredSim, HFSP)

Source texts: `pdftotext -layout` of the three PDFs in `papers/`.
Notation: **[stated]** = in the paper; **[inference]** = my reading, not stated. Quotes are at most 15 words.
Supplementary tables (S1-S7 of ProtTucker; S1-S11 of goPredSim; S1-S2 of HFSP) are NOT in the PDFs, so their contents are unknown. Only the main-text mentions are reported.

---

## 1. Heinzinger et al. 2022, ProtTucker / EAT (`lqac043.pdf`, NAR Genom. Bioinform. 4(2):lqac043)

### 1.1 Source data
- [stated] Source is CATH v4.3, the "sequence-unique datasets provided by CATH", CATH-S100 = 123k proteins (Methods > Data set). CATH v4.3 is also the release named in "CATH hierarchy". Four levels C/A/T/H. "5481 different superfamilies" (CATH v4.3.0), from Performance measures.
- [stated] Domains are single CATH domains from PDB. The text says "developed exclusively on single domain proteins" (Discussion). Multi-domain handling is not part of the benchmark.
- [stated] pLM embedding = per-residue embeddings from the last layers, averaged over protein length (Fig. 1 caption: "averaging over protein length"). The pLMs are static, with no fine-tuning. Only the ProtT5 encoder is used, in half precision.
- [stated] The distance used is Euclidean. "The Euclidean distance performed better than the cosine distance" (Evaluation and prediction). In ProtTucker the raw 1024-d ProtT5 vectors are mapped by a 2-layer FNN to 128-d. For a raw-embedding benchmark (our case) the 1024-d ProtT5 mean vector is what Table 1 "EAT unsupervised ProtT5" uses.

### 1.2 Construction of the sets (in order)
1. [stated] From CATH-S100 (123k), **test300** (300 proteins, final evaluation) and **val200** (200 proteins, early stopping) were "randomly split off". Two constraints: (1) "every homologous superfamily appeared maximally once in test300 ∩ val200", and (2) every protein in test300/val200 has a Structural Sub-group (SSG) annotation (clusters of domain-structure relatives superposing within 5 Å).
   - Ambiguity [inference]: "(1)" can be read as "each H-superfamily at most once within test300 and val200 combined". It can also be read as "each H at most once per set". Not clarified in the text. The effect is that test queries are drawn one per superfamily, so the test set is spread over many superfamilies instead of being dominated by large ones.
2. [stated] **train66k**: from CATH-S100, remove every protein with "more than 20% pairwise sequence identity (PIDE)" to any val or test protein. Tool: MMseqs2 iterative profile search, `--num-iterations 3`, highest sensitivity `-s 7.5`, bidirectional coverage `--cov-mode 0`. (Same 20% also written as PIDE(Ptrain, Ptest300|val200) ≤ 20% for the Gene3D set.) Coverage value for this step is not given. The 80% value is stated only for the Gene3D clustering.
3. [stated] Large families (>100 members) within CATH-S100 were clustered at **95% PIDE and 95% length coverage of both proteins** (MMseqs2, bidirectional coverage, `--cov-mode 0`). Cluster representatives were used for training (66k, train66k) and "as lookup set during early stopping on set val200".
4. [stated] **lookup69k** (69k proteins) = a second lookup set built the same way "but ignored val200 proteins during redundancy reduction". Purpose: "similar sequence properties to those during the final evaluation while 'hiding' them during training".
   - [inference] So lookup69k was filtered (≤20% PIDE) against test300 only, not against val200. Val200-like proteins may therefore be in lookup69k. This is safe because val200 is not used for evaluation.
5. [stated] **test219**: "To ensure strict non-redundancy between lookup69k and test300, we further removed any protein from test300 with HVAL > 0 to any protein in lookup69k (219 proteins)". "All performance measures were computed using test219."
   - [inference] The 300 → 219 drop (81 queries, 27%) shows the PIDE ≤ 20% lookup-side filter alone did not guarantee HVAL ≤ 0. The HSSP-value criterion (see 1.3) is stricter than plain PIDE for short and for long alignments. The filtering therefore ran in two passes: lookup-vs-queries at 20% PIDE (MMseqs2), then queries-vs-final-lookup at HVAL > 0. The second pass removes **queries**, not lookup entries.
6. [stated] Extra training set train11M: Gene3D (v21.0.1) 61M sequences clustered at 50% PIDE and 80% bidirectional coverage, then MMseqs2 profile search (same parameters) to remove representatives with ≥ 20% PIDE to any test300/val200 protein → 11M sequences. Not a lookup set; mentioned only for the training-data comparison.
7. [stated] Side set **CATH-S30** (3186 proteins): CATH v4.3 clustered at 30% using HMM profiles from HMMER, with proteins lacking an equivalent SCOPe entry discarded (domain boundaries and superfamily assignment "nearly identical"). Used only for the SSAP correlation (Fig. 4), not for accuracy. Description of the rest of Fig. 4 is in the caption.
8. [stated] Subcellular-location side test (not CATH): EAT from DeepLocTrain to setHard (490 proteins), "strictly non-redundant to DeepLocTrain". Details in Supplementary Table S5 (unavailable).

### 1.3 Similarity criteria beyond %identity
- [stated] HVAL (HSSP value), Introduction: "HVAL > 5 (>25% PIDE ... over >250 aligned residues)" is the daylight zone, with >90% of pairs structurally similar. "HVAL<-5 (<15% PIDE for > 250 aligned residues)" begins the midnight zone, with >90% of pairs structurally different. So HVAL > 0 means roughly above the twilight/midnight boundary. Citations (10, 12) are Rost 1999 / Sander & Schneider.
- [stated] The formula is NOT given in ProtTucker. It is given in HFSP (see §3.3). HVAL = PIDE minus an alignment-length-dependent threshold curve, so 25% PIDE at >250 residues ≈ HVAL 5.
- [stated] Rule applied: HVAL > 0 to any lookup protein removes the query. Which alignment tool produced the HVAL values for test300 vs lookup69k is not stated. [inference] Probably the same MMseqs2 profile search, but this is unverified.
- [stated] Sequence-distance sensitivity: Fig. 6 removes, from lookup69k, every protein with E-value ≤ threshold to a query (thresholds on a log scale, including 1e-3 as "default threshold for most sequence searches"). Hits come from HMMER with CATH-Gene3D HMM profiles and from MMseqs2 `-s 7.5 --cov 0`.

### 1.4 Leakage control and fairness statements
- [stated] Train/lookup vs val/test: PIDE ≤ 20% (MMseqs2, 3 iterations, `-s 7.5`, cov-mode 0). Query vs lookup: HVAL ≤ 0 (test219).
- [stated] Val200 is kept "hidden" from training and from "any other optimization" except early stopping (accuracy of H-level prediction on val200).
- [stated] Fairness caveat about the baseline: "MMseqs2 had been used for redundancy-reduction, i.e. the data set had been optimized for minimal performance of MMseqs2" (Results, EAT with raw embeddings). The authors state that the split is biased against MMseqs2 HBI. HMMER HBI (Gene3D profiles) is not subject to this bias in the same way, and it wins at H (77 vs 64 for raw ProtT5, Table 1).
- [stated] Redundancy limits the lookup: "highly non-redundant lookup set, presumably removing many easy hits" (Discussion). In contrast, for proteome annotation "we transferred annotations for all CATH-S100 proteins, leveraging 'redundant annotation transfers' to increase coverage". So lookup redundancy changes coverage and accuracy figures. The benchmark and the deployment setting are deliberately different.
- [stated] Family size bias: "when splitting test219 into three bins of varying family sizes, we still observed a trend towards higher accuracy" for larger families at H (small ≤10, medium 11-70, large ≥70 members; Fig./Supp. Fig. S1). Explanations given: more random hits in large families, and large families are sampled more in pLM pre-training. 
- [stated] pLM pre-training leakage is not addressed as a split issue beyond that remark. [inference] There is no statement that ProtT5/BFD/UniRef50 training data were checked against test300.

### 1.5 Queries with no possible correct answer
- [stated] "If no protein in the lookup set shared the annotation of the query protein at a certain CATH-level (more likely for H than for C), the sample was excluded from the evaluation of this CATH-level as no correct prediction was possible (Supplementary Table S1)." (Evaluation and prediction). So the evaluation population differs per level (Supp. Tables S2, S3 give counts and numbers of classes; unavailable).
- [stated] A level-n prediction is correct "if and only if all previous levels were correctly predicted" (hierarchical correctness). So C, A, T, H accuracies are nested, but they are computed on level-specific subsets.
- [stated] "Predictions were counted as incorrect if a query did not have a hit in the lookup set but a lookup protein of the same CATH annotation existed" (this applies to HBI, which can return no hit; EAT always has a nearest neighbour). Table 1 note: no hit with E-value < 10 for MMseqs2/HMMER counts as wrong.
- [stated] "performance measures not normalized by background numbers could be higher for lower levels", so the random baseline (random lookup protein): C 29, A 9, T 1, H 0 (Table 1).
- [inference] Absolute accuracies are not comparable across levels or across splits unless the per-level query counts and class counts are fixed. The reported 95% CIs are ±4 to ±8 points on n ≈ 219 or fewer.

### 1.6 Calibration, thresholds, reliability, coverage vs precision
- [stated] Reliability = Euclidean distance of the nearest neighbour. Accuracy rises and coverage falls as the distance threshold falls (Fig. 5, Supp. Tables S3/S4). Equation 4 defines coverage as the fraction of test219 with a prediction below the threshold. Accuracy corresponds to precision, coverage to recall (micro-averaged).
- [stated] Example: at Euclidean distance ≤ 1.1, "predictions were made for 57%, 57%, 59% or 75%" of the test set for C, A, T, H, and "96%, 93%, 91% or 90% were correct". Note the H coverage is higher than C, A, T. Reason not given. (Fig. 5 caption phrases the same point: Cov(H) = 75%, Acc(H) = 90% at 1.1.)
- [stated] 0.9 is used as the "5% expected error rate" threshold (Table 4, from Supp. Table S3). The Discussion says the real error at ≤0.9 may be "up to 15%" instead of 5%, because Gene3D agreement was lower than expected (75-84% vs the >90% expected).
- [critical, stated] **These curves/thresholds were computed on test219 itself** (Fig. 5 caption: "transferring annotations ... from lookup69k to the queries in set test219"). There is no separate calibration set. val200 was used only for early stopping, not for threshold selection. The thresholds 1.1 and 0.9 are therefore fitted on the test data, and the paper itself shows the 5% target miscalibrated when applied to proteomes. This is the gap our calibration set must close.
- [stated] Distances are in the ProtTucker 128-d space. Raw ProtT5 distances are not on the same scale. (goPredSim states the same about embeddings, see §2.4.)
- [stated] Fig. 2 dashed line at 1.1 "marked the threshold at which EAT performances started to decrease".

### 1.7 Metrics and statistics
- [stated] Accuracy (Eq. 3), coverage (Eq. 4), multi-class MCC (Eq. 5, used in Fig. 6). 95% CI from n = 1000 bootstrap resamples of test-set predictions; CI = 1.96 × bootstrap standard error (Eq. 6).
- [stated] Random baseline = label of a randomly picked protein of lookup69k.
- [stated] Ablation Table 2, EAT from lookup69k to test219: ProtTucker(ProtT5) baseline 89/75/64/76 (C/A/T/H).
- [stated] Table 1 key rows (lookup69k → test219): random 29/9/1/0; MMseqs2 52/36/29/35; HMMER 70/60/59/77; raw ProtT5 84/67/57/64; raw ESM-1b 79/61/50/57; ProtTucker(ProtT5) 89/75/64/76; with train11M 88/77/68/79.

### 1.8 Stated limitations / self-criticism about the splits
- Baseline is penalised by the split (MMseqs2 used for redundancy reduction; 1.4).
- Residual redundancy matters: HBI "performed much better with than without residual redundancy" (Fig. 6). EAT "much less pronounced". The size of EAT's drop is the quantity that justifies the HVAL/PIDE filter. Fig. 6 handles no-hit as false positive; "pairs without CATH class matches were ignored".
- Family-size trend (1.4).
- Multi-domain proteins and domain number mismatch are a stated limit of per-protein averages (Discussion).
- Test set is small (219). CI ±4-8 points. "values in last row not always higher" (train11M effect "neither statistically significant nor consistent").
- Expected error at distance ≤0.9 possibly underestimated (5% vs up to 15%).

---

## 2. Littmann et al. 2021, goPredSim (`s41598-020-80786-0.pdf`, Sci. Rep. 11:1160)

### 2.1 Datasets
- [stated] Lookup source: GOA (29,904,266 UniProtKB proteins in Feb 2020), restricted to **Swiss-Prot** records. Proteins annotated only at the ontology roots (GO:0003674, GO:0008150, GO:0005575) "were considered meaningless and were excluded". GOA2020 = 295,558 proteins, 31,485 GO terms. Each protein carries all annotated terms plus parents (Methods > Data set).
- [stated] GOA2017 (version at the CAFA3 submission deadline, Jan 17, 2017): 307,287 proteins, 30,124 GO terms. Fewer proteins in the newer set, "could not find a definite explanation".
- [stated] GOA2017X: only experimental evidence codes (EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC, "following the CAFA3 definition"), after excluding PIDE = 100% to any CAFA3 target → 303,984 proteins, 28,677 terms. Standard GOA2017/2020 sets are NOT restricted to experimental evidence.
- [stated] Evaluation queries: CAFA3 targets. 130,827 released; 3328 with experimental annotations at benchmark collection (Nov 2017): BPO 2145, MFO 1101, CCO 1097. Modes: NK (no prior experimental annotation) vs LK (some annotations in another ontology), full vs partial evaluation. Only the "full" evaluation is used because the method always predicts.
- [stated] GOA2020-new: 298 proteins with experimental GO annotations added after Feb 2020 and "without any identical hits (i.e. 100% PIDE) in the lookup set GOA2020".
- [stated] Time-based logic: with the 2017 lookup, "our predictions were directly comparable to CAFA3" because the lookup existed before the CAFA3 deadline (Feb 2017 as written in Results; Methods says Jan 17, 2017 [minor inconsistency in the paper]). The queries are proteins whose experimental annotations appeared later.

### 2.2 Redundancy reduction
- [stated] Base rule: remove from the lookup all entries identical to a query: "removed all entries identical to CAFA3 targets (PIDE = 100%; set: GOA2017-100)". GOA2020-100 likewise. The "-100" suffix means PIDE = 100% to any query was removed.
- [stated] Additional filtered lookups: "applying CD-HIT and PSI-CD-HIT to the GOA2020 and GOA2017 sets against the evaluation set at thresholds θ of PIDE = 100, 90, 80, 70, 60, 50, 40, 30 and 20%" (Methods, Performance evaluation). Table S6 holds details (unavailable). CD-HIT parameters (word size, coverage flags, alignment mode) are NOT given in the main text.
- [stated] Direction: lookup filtered against the evaluation set (lookup entries removed). Queries are not dropped.
- [stated] Main-text results: GOA2017-20 or GOA2020-20 (PIDE < 20% between query and every lookup protein) gives Fmax 33/43/53 (abstract, BPO/MFO/CCO) vs 37/50/57 with only exact duplicates removed.
- [stated] BLAST baseline comparison fairness: comparing BLAST on the full GOA2020-100 with embedding transfer on GOA2020-40 "under-estimated the differences ... because the two approaches transferred annotations from different data sets". For fairness they "re-ran BLAST only considering hits below certain PIDE thresholds". Consequence: BLAST Fmax at PIDE < 20% = 8% BPO, 10% MFO, 11% CCO, "largely due to low coverage". Same lookup and same cap must be applied to both methods.

### 2.3 Similarity measures
- [stated] Only PIDE is used (plus the alignment-based CD-HIT thresholds). No HVAL, no E-value cut in the main text. The introductory rule is "IF PIDE(Q,T) > threshold θ, transfer annotation".
- [stated] Weak correlation between embedding distance and PIDE (Spearman ρ = 0.29, "unexpectedly low") and between embedding distance and GO similarity (ρ = 0.28; Jaccard index, Eq. 7). Pairs close in embedding and distant in sequence exist.

### 2.4 Reliability and calibration
- [stated] Reliability index RI(p) = (1/k) Σ_{i=1..l} 0.5/(0.5 + d(q, n_i)) (Eq. 5), d = Euclidean distance on 1024-d SeqVec/ProtBert vectors. k = hits, l = hits carrying term p. RI = 1 when d = 0 (a self-hit). Minimal RI on GOA2020-100 was 0.24.
- [stated] "the RI ... not directly comparable for predictions based on different values for k" and not comparable across embeddings, "because embeddings might be on different scales".
- [stated] **Thresholds fitted on the evaluation set**: "Our method and especially the threshold to transfer a GO term were 'optimized' using the CAFA3 targets." Fmax-optimal RI thresholds: 0.35 BPO, 0.28 MFO, 0.29 CCO (Fig. 3). Fmax "implies that the assessment fixes the optimal value rather than the method providing this value. Although this arguably over-estimates performance" (Methods).
- [stated] Held-out confirmation: without any changes, thresholds from CAFA3 were applied to GOA2020-new (298 proteins). F1 = 50 ± 11 (BPO), 54 ± 5 (MFO), 66 ± 8 (CCO). MFO is "slightly below but within the 95% CI". So a second, later, never-tuned set is used as the honest check. 
- [stated] Precision/recall vs RI (Fig. 3): both almost constant for RI up to ~0.3; above that, precision rises (up to 82% BPO, 91% MFO, 70% CCO) and recall falls, with fluctuation at RI > 0.8 "caused by the low number of terms predicted". For CCO, RI "essentially did not correlate with precision" (naive method gets Fmax ~55%, so the metric is weak there). RI = 0.5 ≈ 50% precision and recall (Fig. 4 text).
- [stated] k was chosen as 1: for k = 2..10, Fmax stayed within the 95% CI of k = 1 (Table S2); more terms predicted as k grows.

### 2.5 No-possible-answer handling
- [stated] No explicit exclusion rule. The method "predicted for every protein" (no reliability cut inside Fmax beyond sweeping thresholds), so only the CAFA "full" evaluation is used, in which an unpredicted protein would be penalised. Terms are GO sets, so partial overlap is possible and an "impossible" query is not a single yes/no case as in CATH.
- [inference] GO's multi-label, hierarchical setting avoids the empty-label case that CATH H-level has. EC/CATH with single labels needs the explicit rule from ProtTucker (1.5).

### 2.6 Other stated points
- [stated] More recent annotations raise Fmax (GOA2017 → GOA2020: SeqVec BPO 37 → 51, MFO 50 → 61, CCO 57 → 65) but at 20% PIDE they fall back to the same level (33/44/54, same as GOA2017-20). Interpretation in the paper: "many helpful new experiments simply refined previous computational predictions", i.e. most of the new annotations are on proteins similar to already-annotated ones. Therefore a lookup that grows over time inflates scores unless redundancy is controlled.
- [stated] Non-experimental annotations in the lookup help (BPO 31 → 37 with GOA2017X → GOA2017). Probable cause: many non-experimental annotations are themselves sequence-transferred, a "merger of sequence and embedding transfer". Lookup evidence quality is a split parameter.
- [stated] Protein length: no correlation between performance and length (Fig. S2, Table S1).
- [stated] Ties: "When multiple pairs reached the same distance, all were considered" (k may be exceeded).
- [stated] Bootstrap: 10,000 samples, symmetric 95% CI (Methods) [ProtTucker uses 1000; differing choices].
- [stated, limitation] "Ultimately, we failed to really explain why the abstracted level of sequences captured in embeddings outperformed raw sequences."

---

## 3. Mahlich et al. 2018, HFSP (`bioinformatics_34_13_i304.pdf`, Bioinformatics 34(13):i304-i312)

### 3.1 EC enzyme dataset construction
- [stated] Starting set: reviewed Swiss-Prot proteins "with only one EC annotation per protein (complete at all four levels; 214 000 proteins; Swiss-Prot set)" (2.1).
- [stated] **Trusted label** for the reference sets: "experimental evidence of protein existence (e.g. crystal structure, protein detection by antibodies, etc.) and an EC annotation in BRENDA" (2.1). Swiss-Prot 2017 set = 7022 proteins (UniProt release 2017_09, extracted Oct 2017).
- [stated] Time slice: entries present before January 2002 → Swiss-Prot 2002 (3908 proteins). Used to fit the curve because the 2002 HSSP formula was developed on data of that time. Same extraction date; redundancy reduced with CD-HIT to **98% sequence similarity and 98% target sequence coverage** → Swiss-Prot 2002 = 3801 proteins, 1481 unique ECs; Swiss-Prot 2017 = 6835 proteins, 2552 unique ECs.
- [stated] Sub-splits: Swiss-Prot_pro 2017 = 2572 prokaryotic; Swiss-Prot_euk 2017 = 4263 eukaryotic. Further sets: 293,058 proteins without EC; 48,536 proteins with incomplete or multiple EC annotation.
- [stated] **Incomplete or multiple EC numbers**: excluded from the reference/training sets ("only one EC annotation per protein, complete at all four levels"). They are handled as a separate query-only group ("incomplete & multiple EC", Fig. 7B). Predictions were possible for 57% of them (27,717 of 48,536); 53% scored at HFSP ≥ 0, 13% above HFSP ≥ 14.
- [stated] Pair-level label: functional identity = same EC at the chosen level (third or fourth). Example: 1.1.1.27 vs 1.1.1.28 are different at level 4, same at level 3 (1.1.1) (2.3). Dashes ("-") denote lack of specificity.
- [stated, caveat] EC level-4 numbers differ in specificity: e.g. 5.4.99.- is a "temporary house" for unclassified reactions; 2.7.11.1 collects non-specific kinases. So "EC level 4 identity" is not a uniform notion of function (Introduction, 3.2).
- [stated] Class balancing: "limited the size of EC categories to no more than 50 proteins (randomly chosen for the 19 larger categories)" at EC level 3 (Fig. 2C; 2.4), then pairs balanced to equal numbers of same-EC vs different-EC. In test folds, the removed proteins were reintroduced.
- [stated] Alignments (2.2): each dataset's proteins aligned against the **full non-reduced Swiss-Prot** (555,594 proteins), then only alignments where both proteins are in the dataset were kept. PSI-BLAST 2.2.29+, `-num_iterations 3`, top 500 hits, E-value 1e-10 inclusion (`-inclusion_ethresh 1e-10`), final `-evalue 1e-3`. MMseqs2: `--alignment-mode 3` (identity over the full alignment length), `--num-iterations 3`, `--e-profile 1e-10`, `-e 1e-3`, prefilter sensitivity `-s 5.6` (default). Profile uses the 1000 most diverse sequences (vs PSI-BLAST "n best").

### 3.2 Redundancy reduction and leakage
- [stated] Cross-validation: 10 folds "such that no sequence in one set shared more than 40% identity with a sequence in another set (CD-HIT clusters)" (2.3). Redundancy reduction within sets only at 98%.
- [stated] Final curve retrained on the complete balanced Swiss-Prot 2002 pair set "for all further use" (no held-out data for that final curve).
- [stated] Temporal hold-out: Swiss-Prot 2017 proteins NOT in Swiss-Prot 2002 (added after Jan 2002) used as test (Fig. 4). Level 3 performance similar to expectation; at level 4, newly added proteins perform worse. Authors' reading: the measure "remains applicable for newly added proteins AND enzyme classes".
- [caveat, stated] Reference-vs-query overlap: "this performance is the upper limit of actual HFSP performance, as Swiss-Prot 2002, on which our method was developed, is a subset of Swiss-Prot 2017." So predicting Swiss-Prot 2017 contains training proteins.
- [inference] Within the "highest hit" evaluation, the query's own other redundant copies up to 98% identity remain in the reference, unless excluded by the self-match rule (below). Splits stricter than 98% are not used for the prediction experiments.
- [stated] Self-hits are excluded: "highest HFSP scoring protein match (≥ 0; excluding self-matches)".

### 3.3 HSSP/HFSP formula (exact definition)
- [stated] Eq. (1), HSSP (Rost 2002): HSSP = PIDE − 100 for L ≤ 11; HSSP = PIDE − 480·L^(−0.32·(1+e^(−L/1000))) for 11 < L ≤ 450; HSSP = PIDE − 19.5 for L > 450. PIDE is the percent identity of the alignment; L is the **ungapped** alignment length. [The PDF text extraction garbles the exponent; the reading above is a reconstruction. Check against the PDF page i307 and i308 before reuse. I verified consistency: 480·450^(−0.32·(1+e^(−0.45))) = 19.5 and 770·450^(−0.33·1.638) = 28.4, so the three branches join continuously.]
- [stated] Eq. (4), HFSP (refit): same form with factor 770 and exponent 0.33; constant 28.4 for L > 450 (original factor 480, exponent 0.32). Parameters fitted by Nelder-Mead on F1 over factor 300-1500 in steps of 50 and exponent 0.3-0.9 in steps of 0.05, on EC level 3 identity (2.4). Pairs "above the curve" are predicted functionally identical; "HFSP ≥ 0" is the default cut. 
- [stated] HFSP is computed from MMseqs2 profile alignments. Pearson correlation of HFSP-from-MMseqs2 vs HSSP-from-PSI-BLAST = 0.95 (Fig. 3).
- [consistency with ProtTucker] ProtTucker's "HVAL > 0" is the same quantity as HSSP above (Rost 1999/2002 refs 10, 12), but ProtTucker does not say whether its HVAL came from PSI-BLAST, MMseqs2 or the HFSP-refit constants. [Not stated; open question.]

### 3.4 Precision/coverage logic of the curve
- [stated] Pair-level metrics: TP = same-EC pairs on or above the threshold, FP = different-EC pairs above, FN = same-EC pairs below (2.4). Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 (Eqs. 2-3). Thresholds are distances from the curve along the PIDE axis.
- [stated] EC level 3 at HFSP ≥ 0: precision 96 ± 1.2%, recall 64 ± 1.6% (cross-validation). EC level 4 at HFSP ≥ 0: precision 44.1 ± 3.6%, recall 71.5 ± 1.6% ("disappointing"). Raising the cut improves precision at a recall cost: HFSP ≥ 20 → 93% at level 4; HFSP ≥ 64.6 → >99% (example pair 93% PIDE, L = 758).
- [stated] Fig. 1B vs 1C: "prediction was done using the highest HFSP scoring alignment per protein" vs "all alignments were used, resulting in significantly worse performance". The nearest-neighbour (top hit) protocol matches EAT.
- [stated] Protein-level function prediction: Swiss-Prot 2017 set, top hit at cut ≥ 0, no self-hit: fourth-level EC correct for 4668 of 5647 (83%); third-level 5425 (96%). Of 6835: 625 did not reach the cutoff, 563 had no alignment; 645 of these had unique ECs (no other protein with the same level-4 EC, 291 + 354). Hence 1188 proteins (17%) could not be assigned: 8% HFSP limit, 9% absence of homologs.
- [stated] **Unique-label queries**: "unknown" = level-4 EC appearing once in the set (19%: 1317 of 6835). Their top-hit HFSP score peaks at ~2, while correct "known" predictions have peaks at ~20 (distant) and ~65 (close); incorrect "known" predictions follow the "unknown" distribution (Fig. 7A). At HFSP ≥ 14, known-incorrect + unknown make up <10% (FDR 9.6%); between 0 and 14 nearly 40%. The authors say they cannot separate wrong predictions from missing-reference ones, and that the 5% FDR point (HFSP ≥ 28) cannot be estimated reliably for unannotated proteins.
- [stated] Prokaryote/eukaryote difference at level 3 (precision/recall 96 ± 1.5/62 vs 91 ± 1.5/47) and reversed at level 4 (62/55 vs 42/79). Authors suggest the eukaryote majority may have biased the curve. Class bias is the same problem as the EC-size balancing.
- [stated] Swiss-Prot-wide use: estimated 142,831 of 214,000 (67%) enzymes correctly annotated at level 4; 32,201 (15%) carry a level-4 EC with no member in the reference ("false positives by default"); 4,937 are "wrongly annotated" by top hit; 34,031 (19%) could not be annotated vs 8% expected. Authors infer "at least a sixth" of annotations are incorrect; they also say part of this may be HFSP error (17% false positives at that cutoff).

### 3.5 Stated limitations
- EC level 4 is not a consistent unit of function (3.1); "moonlighting" proteins and multi-domain proteins confuse similarity (Introduction, 3.3).
- Fit on Swiss-Prot 2002 with class cap and balanced pairs; eukaryote-heavy.
- The evaluation set is partly the training set (upper limit statement).
- FDR cannot be estimated for unannotated data because enzymes among them may not follow the Swiss-Prot distribution.
- Non-enzymes: 91% no match; of the matching ones 21% scored ≥ 0 (3% of predictions; 1% at ≥ 14).

---

## 4. Conflicts, gaps and surprises across the papers

1. **Different redundancy tools and cuts**: ProtTucker lookup-side 20% PIDE (MMseqs2, 3 iterations, `-s 7.5`, cov-mode 0) plus query-side HVAL > 0; goPredSim CD-HIT / PSI-CD-HIT at 20-100% PIDE (parameters in unavailable Table S6); HFSP CD-HIT 98%/98% coverage for the sets and 40% for CV folds. None states a single coverage rule for the 20% cut in ProtTucker.
2. **HVAL source unspecified** in ProtTucker (§3.3).
3. **Calibration on test**: ProtTucker thresholds 1.1 and 0.9 were read off test219; goPredSim RI thresholds were tuned on the CAFA3 evaluation targets (admitted), then checked on a never-tuned set; HFSP fitted its curve on Swiss-Prot 2002 and checked on post-2002 proteins. Only goPredSim and HFSP have a truly separate confirmation set; ProtTucker has none for the thresholds.
4. **Empty-label handling differs**: ProtTucker excludes queries per level; HFSP counts unique-EC queries as "unknown" and reports them separately; goPredSim does not exclude.
5. **ProtTucker "test300 → test219"** loses 27% of queries to the HVAL rule, so split size must anticipate a loss.
6. **Surprise**: coverage at H (75%) is above coverage at C/A/T (57-59%) at the same distance 1.1 (§1.6). Not explained in the text.
7. **Surprise**: goPredSim states the CAFA3 deadline as Feb 2017 in Results and Jan 17, 2017 in Methods.
8. The BLAST/MMseqs2 baselines are penalised by the redundancy reduction in both ProtTucker and goPredSim (the latter fixed by re-running BLAST under the same PIDE cap).

---

## 5. Requirements implied (imperative form, with evidence)

1. **Keep three roles disjoint by sequence.** Lookup, calibration and test sets must be mutually non-redundant by sequence similarity, not only by identical ID. [ProtTucker Methods > Data set: 20% PIDE train/lookup vs val/test; goPredSim: PIDE = 100% removal as the minimum (Results; Methods).]
2. **Filter the lookup against the queries, then filter the queries against the final lookup.** First remove lookup entries above the identity cut to any calibration/test protein, then drop any test query that still has HVAL > 0 to the final lookup. Expect to lose about a quarter of the queries (300 → 219). [ProtTucker, Data set: "we further removed any protein from test300 with HVAL > 0".]
3. **Use an alignment-length-aware criterion, not plain PIDE alone.** Define the cut with HVAL = PIDE − f(L) (HFSP Eq. 1/4 form, ungapped L), at HVAL > 0 as the leak criterion. Use PIDE ≤ 20% from an iterative profile search as the first-pass cut: MMseqs2 `--num-iterations 3 -s 7.5 --cov-mode 0`. [ProtTucker Data set + Intro; HFSP Eq. 1, 4.] State and record which alignment tool computes HVAL (open point, §3.3).
4. **Apply the same cut to every baseline.** Run sequence baselines (MMseqs2, BLAST) on the same lookup under the same identity cap, so they are not handicapped nor advantaged by the split. Also report how the filter tool relates to the baseline tool. [ProtTucker Results: "dataset had been optimized for minimal performance of MMseqs2"; goPredSim Fig. S4 re-run of BLAST with PIDE caps.]
5. **Compress large families in the lookup.** Cluster very large families (>100 members) at 95% PIDE and 95% bidirectional coverage so one family does not dominate. [ProtTucker, Data set.]
6. **Spread queries over labels.** Select test and calibration queries so each homologous superfamily (fine label) occurs at most once, and every query has a full label (SSG-style quality filter). [ProtTucker, Data set condition (1)-(2).] For EC, the analogue is one query per 4-digit EC, with complete four-level EC.
7. **Define the evaluable query population per level.** Exclude from a level's accuracy any query for which no lookup protein carries its label at that level; report the per-level n and the number of classes; report separately how many queries were unanswerable. [ProtTucker, Evaluation: "no correct prediction was possible"; HFSP 3.4 and Fig. 7: report "unknown" separately.] Count a hit at a deeper level correct only if all shallower levels are correct.
8. **Report accuracy together with coverage as a function of distance.** Use micro-averaged accuracy (precision) and coverage (recall) across a distance threshold sweep; add bootstrap 95% CI (1000+ resamples). [ProtTucker Eq. 3, 4, 6, Fig. 5; goPredSim Fig. 3.]
9. **Fit distance thresholds and settings on the calibration set only; touch the test set once.** ProtTucker and goPredSim both tuned thresholds on their evaluation targets and say so; avoid repeating this. [ProtTucker Fig. 5 on test219; goPredSim: "optimized using the CAFA3 targets"; Fmax "arguably over-estimates performance".] Apply the frozen thresholds unchanged to the test set.
10. **Make the calibration set a faithful proxy of the test set.** Build it with the same procedure and the same redundancy relation to the lookup as the test set. [ProtTucker: lookup69k built "for validation which had similar sequence properties to those during the final evaluation".] Calibration must additionally be non-redundant to test (inference; not stated in any of the three papers, which have no separate calibration-vs-test rule). 
11. **Thresholds are not portable.** Do not reuse distance thresholds across embeddings, lookup sets, k, or pooling; recompute per configuration. [goPredSim Methods on RI: "embeddings might be on different scales"; ProtTucker 0.9 → possibly 15% error instead of 5%.]
12. **Prefer a temporal or later-annotation hold-out as an extra check where feasible.** Entries added after the lookup freeze serve as an untuned confirmation set (HFSP: post-2002 proteins; goPredSim: GOA2020-new, CAFA3 post-deadline annotations). Control redundancy within it, because growth of annotations is mostly near existing families. [goPredSim Results: GOA2017 vs GOA2020; HFSP Fig. 4.]
13. **Fix the nearest-neighbour protocol.** Use the single top hit (k = 1), exclude self-hits, and treat distance ties explicitly. [ProtTucker EAT; goPredSim k = 1 and tie rule; HFSP 2.5 "excluding self-matches"; HFSP Fig. 1B vs 1C shows all-alignment use is worse.]
14. **For EC: trust only complete, single, experimentally supported labels in lookup and test.** Keep proteins with exactly one complete four-digit EC and evidence of existence (experimental), exclude incomplete or multiple EC from lookup, calibration and test, and handle them only as an optional separate query-only stress set. [HFSP 2.1, Fig. 7B.] Pin the Swiss-Prot release and the EC source (BRENDA).
15. **For EC: cap and report class imbalance.** Cap EC class sizes (HFSP used ≤50 per level-3 class) in anything used to fit or calibrate, and report results per EC level; level 4 is not uniformly specific, so report levels 1-4 separately. [HFSP 2.4, 3.2.]
16. **For EC: report the unique-label ("unknown") fraction.** Queries whose 4-digit EC occurs nowhere else in the lookup cannot be answered by transfer; give their count and how they are scored. [HFSP 3.4 / 3.6: 19% of Swiss-Prot 2017 set.]
17. **Report family-size strata.** Break test results down by family size (≤10, 11-70, ≥70 members) because accuracy rises with family size. [ProtTucker, Family size mattered, Supp. Fig. S1.]
18. **Report residual-redundancy sensitivity.** Show accuracy as lookup entries above increasing similarity (E-value or PIDE) to the query are removed. This is the check that shows a split is not leaking. [ProtTucker Fig. 6; goPredSim Fig. 2, nine PIDE steps 100 → 20%.]
19. **Record the lookup evidence level.** The lookup should be experimentally verified (e.g. goPredSim's X sets) or the evidence mix must be stated, because homology-transferred labels inflate scores. [goPredSim Results: GOA2017X vs GOA2017; HFSP "experimental evidence".]
20. **Use a CD-HIT-style identity-clustered fold design if cross-validation is used.** Folds must not share >40% identity across folds (HFSP); this is weaker than the 20%/HVAL cut and is acceptable only for fitting, not for the final test. [HFSP 2.3.]
