# EC split requirements

Research notes for the EC side of the EAT benchmark. Checked 8 November 2026. **Unfinished checks are marked explicitly.** URLs below are the cited sources.

## 1. Which EC annotations can be trusted as ground truth?

**Established practice**

- UniProt evidence belongs to an annotation, not to the entry as a whole. A reviewed Swiss-Prot record can contain experimentally supported activity alongside activity inferred from similarity or a sequence rule. In ECO, `ECO:0000269` denotes experimental evidence used in a manual assertion; `ECO:0000250` is sequence-similarity evidence used in a manual assertion; `ECO:0000255` is sequence-model evidence used in a manual assertion; `ECO:0000256` is sequence-model evidence used in an automatic assertion; and `ECO:0000305` is curator inference. Manual curation or Swiss-Prot status alone therefore does not make an EC experimentally established. [UniProt evidence guide](https://www.ebi.ac.uk/training/online/courses/uniprot-exploring-protein-sequence-and-functional-info/where-does-the-data-come-from/data-evidence/); [ECO](https://evidenceontology.org/).
- Catalytic-activity comments can describe more than one reaction and carry evidence at the reaction/comment level. An entry-level query such as `(ec:*) AND (cc_catalytic_activity_exp:*)` is a useful accession filter, but it does not prove that every EC number on a returned entry is experimentally supported. [UniProt advanced search](https://www.uniprot.org/help/advanced_search); [UniProtKB annotation manual](https://www.uniprot.org/help/catalytic_activity).
- Protein-existence evidence is not catalytic-activity evidence. In particular, HFSP selected proteins with experimental evidence of protein existence plus a BRENDA annotation; this is not the same inclusion rule as requiring experimental evidence for each UniProt catalytic reaction. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262).

**Project inference / recommended rule**

- Use the UniProt experimental catalytic-activity field to find candidate entries, then parse each catalytic reaction separately. Retain an EC label as ground truth only when that reaction has experimental ECO evidence and a supporting reference. Keep its ECO code and citation in the split manifest. Do not promote other EC numbers on the same accession to ground truth merely because one reaction passed the filter.
- Treat experimentally supported annotations as high-confidence, not infallible. Mahlich et al. report examples where highly similar proteins have different EC assignments and argue that even reviewed Swiss-Prot annotations can be wrong. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262).

**UniProt count check**

- The project brief's 2026_03 figures—48,401 entries for the experimental catalytic-activity/EC query and 279,671 Swiss-Prot entries with EC numbers—were **not independently verified against the archived 2026_03 query results**.
- The live UniProt REST API returned release **2026_04**, dated 4 November 2026. On 8 November 2026, `(ec:*) AND (cc_catalytic_activity_exp:*)` returned **48,484**; adding `reviewed:true AND ec:*` returned **279,887**. The unqualified experimental-activity query returned the same 48,484 as the reviewed-only query, so the current hits are all reviewed. These are current-release counts, not confirmation of the brief's 2026_03 counts. [UniProt REST API query: experimental catalytic activity + EC](https://rest.uniprot.org/uniprotkb/search?query=%28ec%3A%2A%29%20AND%20%28cc_catalytic_activity_exp%3A%2A%29&size=0); [reviewed EC query](https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20ec%3A%2A&size=0); [2026_03 archive](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2026_03/knowledgebase/).

## 2. Multiple and incomplete EC numbers; what counts as correct?

**Established practice**

- HFSP deliberately used proteins with one complete four-level EC number. In its 2017 Swiss-Prot set, the paper reports 7,022 candidates before filtering/redundancy reduction and 6,835 proteins in the reduced set; it also places proteins with incomplete or multiple EC annotations in a separate subset. Thus HFSP does not establish a scoring rule for multi-label or partial-label queries. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262).
- CLEAN evaluates enzyme prediction as multi-label prediction and reports precision/recall/F1; its external sets include Price-149 and New-392. CARE explicitly includes a promiscuous-enzyme test set. CARE's metric checks the best hierarchical match for each true label, which is useful for hierarchical partial credit but is not an exact-set metric: extra transferred labels need not be penalized as they would be under set precision or exact match. [Yu et al., 2023](https://doi.org/10.1126/science.adg6005); [CARE, NeurIPS Datasets & Benchmarks, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf).
- EC depth is uneven in biological meaning. HFSP reports substantially different pairwise performance at EC level 3 and level 4 and discusses cases where same-level labels represent very different functional distinctions. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262).

**Project inference / recommended rule**

- Keep every experimentally supported reaction label for a query and lookup protein. For a nearest neighbor carrying several trusted ECs, transfer its whole trusted label set—not an arbitrary single EC.
- At each level, compare the sets of known prefixes. For example, `3.4.-.-` establishes the level-1 prefix `3` and level-2 prefix `3.4`, but says nothing about levels 3 or 4. Score those deeper levels as **unknown/not evaluable**, not as false. Keep partial labels in a separate prefix-level analysis; do not silently treat a dash as a wildcard full label.
- For multi-label queries, report per-level set precision, recall and F1 plus exact full-label-set match. A hierarchical per-label “any match” score may be included for comparison with CARE, but report it separately and disclose whether false extra labels are penalized. Give per-level denominators because incomplete ground truth changes which levels can be scored.

## 3. How published EC splits are constructed

**Established practice**

- HFSP used CD-HIT at 98% sequence similarity and 98% target coverage. Its older Swiss-Prot 2002 subset contained 3,908 proteins before redundancy reduction and 3,801 after; its 2017 reduced set contained 6,835. It balanced same- versus different-EC-level-3 protein pairs for training, then restored held-out proteins removed for balancing when evaluating. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262).
- CLEAN (Yu et al., 2023) uses a large Swiss-Prot training collection (about 220k sequences in the released CLEAN dataset) and reports identity-controlled splits, including a stringent 10%-identity clustering evaluation. Its **New-392** set was assembled from enzymes added after April 2022; **Price-149** is a smaller experimentally characterized external set. The paper uses multilabel metrics. These sets are useful external tests, but they are not a three-role lookup/calibration/test design for nearest-neighbor transfer. [CLEAN paper](https://doi.org/10.1126/science.adg6005); [CLEAN data and code](https://github.com/tttianhao/CLEAN).
- CARE (2024) builds its main training set from 184,529 protein–EC pairs and uses multiple test groups: 432 sequences below 30% identity, 560 in a 30–50% identity band, 148 Price-149 sequences after its filtering, and 209 promiscuous sequences. Its design samples for broad EC coverage rather than preserving the natural class prevalence in every test subset. CARE also compares EC predictors including DeepEC, ECPred and ProteInfer. [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf); [ProteInfer, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10063232/).
- Recent follow-ons include CLEAN-Contact (structure-enhanced CLEAN), DeepECTransformer and EC-Bench (2026), which provides a more unified comparison and documents limitations in reproducing some earlier systems. These are trained predictors, not raw nearest-neighbor transfer; their scores should not be treated as directly comparable to EAT. [CLEAN-Contact](https://www.nature.com/articles/s42003-024-07359-z); [EC-Bench, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12889163/).
- Class imbalance is handled inconsistently. HFSP balances training pairs, CLEAN keeps a large skewed enzyme corpus, and CARE deliberately samples test sets for coverage, including a promiscuous subset. EC-Bench reports class-aware evaluation. A balanced test set answers a different question from performance at natural enzyme prevalence. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262); [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf); [EC-Bench, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12889163/).

**Research gap**

- The original supplementary details for every DeepEC and ECPred split were not re-extracted in this pass. CLEAN, CARE and EC-Bench provide the verified comparison points above; do not infer that every legacy method used the same release, identity algorithm or multi-label policy.

## 4. Leakage and the link between sequence identity and EC agreement

**Established practice**

- CARE documents split leakage caused by different similarity-search tools: its split construction and evaluation used different algorithms, and some test sequences therefore fell outside the intended identity ranges. It cautions that a nominal “<30% identity” test can depend on the algorithm and settings used to enforce that threshold. [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf).
- EC-Bench flags reproducibility and possible data-contamination problems for earlier predictors when training data or released model weights cannot be reconstructed against a later test set. CLEAN's temporal New-392 and Price-149 tests reduce some overlap risks, but do not automatically prove that every model's supervised data and pretraining corpus were free of test sequences. [EC-Bench, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12889163/); [CLEAN, 2023](https://doi.org/10.1126/science.adg6005).
- Sequence identity is not a reliable proxy for exact EC agreement. HFSP describes a pair with 93% identity but different fourth-level EC numbers; in its evaluation, level-3 functional identity was much easier to call than level 4. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262).

**Project inference**

- Remove same-accession and exact-sequence duplicates across lookup, calibration and test. For the main prospective transfer benchmark, do not remove every close homolog: close matches are part of the intended lookup-transfer use case. Instead publish nearest sequence-identity strata and add a separate cluster-cold/remote-homology challenge if desired. State the clustering tool, identity definition, coverage, and version; do not assume a cutoff from one tool is equivalent under another. [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf); [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262).

## 5. Dataset sizes and compute implications

**Established practice**

- HFSP's reduced 2017 reference set was 6,835 proteins. CLEAN's training collection is roughly 220k sequences; CARE uses 184,529 protein–EC pairs; EC-Bench reports 258,386 training samples in its processed setup and a 468-sequence later-release test, with Price-149 evaluated separately. These are different filtering and label units, so the counts are not directly interchangeable. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262); [CLEAN, 2023](https://doi.org/10.1126/science.adg6005); [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf); [EC-Bench, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12889163/).
- Current UniProt API counts imply a candidate set of 48,484 entries with EC and experimental catalytic-activity evidence, versus 279,887 reviewed entries with any EC. The strict evidence-filtered lookup is much smaller than all reviewed Swiss-Prot EC entries. [UniProt REST API](https://rest.uniprot.org/uniprotkb/search?query=%28ec%3A%2A%29%20AND%20%28cc_catalytic_activity_exp%3A%2A%29&size=0).

**Project inference**

- For 1,024-dimensional float32 embeddings, 48,484 vectors take about 0.20 GB (189 MiB); 279,887 take about 1.15 GB (1.07 GiB), before index and metadata. Exact distance work scales as `N_query × N_lookup × embedding_dim`; profile a small batch and chunk/vectorize the search before embedding the full corpus. This arithmetic is an estimate, not a measured runtime. [UniProt REST API counts above; EAT project guide](../../../docs/project.md).

## 6. Recommended EC split recipe

**Project inference / proposal for supervisor approval**

- Freeze **UniProt release 2026_04** as the current source snapshot (released 4 November 2026). Query `reviewed:true AND (ec:*) AND (cc_catalytic_activity_exp:*)`; save the query, release header, retrieval date, accession, canonical sequence, per-reaction EC, ECO evidence, references and Rhea identifiers. The query selects candidate entries; the per-reaction parse determines which labels are trusted. [UniProt REST API](https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20ec%3A%2A%20AND%20cc_catalytic_activity_exp%3A%2A&size=0); [UniProt previous releases](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/).
- Use reaction-level experimental evidence for the strict primary set. Exclude inferred-only ECs from ground truth even when the entry is reviewed. Keep an entry if it has at least one experimentally supported reaction, but include only that reaction's EC label(s). [UniProt evidence guide](https://www.ebi.ac.uk/training/online/courses/uniprot-exploring-protein-sequence-and-functional-info/where-does-the-data-come-from/data-evidence/); [UniProt annotation manual](https://www.uniprot.org/help/catalytic_activity).
- Preserve multiple experimental EC labels. For the four-level primary metric, evaluate complete ECs; retain incomplete ECs for separate prefix-only evaluation, with deeper levels marked unknown. Report exact-set and per-level precision/recall/F1, plus the simpler hierarchical hit score only as a secondary comparator. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262); [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf).
- Pilot a temporal three-role split using archived snapshots: lookup labels present by **2024_04**; calibration entries/experimental labels first appearing between **2024_04 and 2025_04**; test entries/labels first appearing between **2025_04 and 2026_04**. Compare release-to-release records, not just the current entry's last-modified date. Exclude same accessions and exact sequences across roles; measure close identity and report it rather than silently filtering it from the primary realistic-transfer test. Counts and feasibility need a pilot. [UniProt previous releases](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/); [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf).
- Keep lookup-set class frequencies natural; do not cap lookup examples per EC. Report micro and macro scores, per-class support, confidence intervals, coverage, and whether each test EC occurs in lookup. Count labels absent from lookup as unreachable/error in the primary strict result; optionally show reachable-only scores separately. Choose distance thresholds and reliability mapping on calibration only; reserve test for final evaluation. [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf); [EAT project guide](../../../docs/project.md).
- Keep HFSP's 2017 set as a historical comparison, not as the definition of experimental ground truth for the new split: it uses one complete EC per protein and its stated evidence-of-existence/BRENDA filter differs from the per-reaction UniProt rule. [Mahlich et al., 2018](https://doi.org/10.1093/bioinformatics/bty262).

## Requirements implied

- **Freeze** a UniProt release, exact query, retrieval date and checksum before building splits. [UniProt REST API; previous releases](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/).
- **Parse** ECO evidence per catalytic reaction; do not treat entry-level experimental evidence, Swiss-Prot review status or protein-existence evidence as proof for every EC label. [UniProt evidence guide; annotation manual](https://www.uniprot.org/help/catalytic_activity).
- **Retain** all experimentally supported reaction labels; **score** multi-label sets and incomplete prefixes without converting unknown levels into negatives. [Mahlich et al., 2018; CARE, 2024](https://doi.org/10.1093/bioinformatics/bty262).
- **Separate** lookup, calibration and test by an agreed temporal rule; **remove** exact duplicates across roles; **report** sequence identity strata and tool-specific cluster settings. [CARE, 2024](https://papers.nips.cc/paper_files/paper/2024/file/05a7ad45d75a3082d7a3a70de8743140-Paper-Datasets_and_Benchmarks_Track.pdf).
- **Preserve** natural lookup prevalence and report macro as well as micro metrics, support, coverage and unreachable-label behavior. [CARE, 2024; EAT project guide](../../../docs/project.md).

## Open questions for the supervisors

- Should the primary split use the proposed 2024_04 / 2025_04 / 2026_04 snapshots, or another cutoff after a size pilot?
- Is experimental evidence required for every reaction label, or may BRENDA-supported/curator-inferred labels enter a separate expanded set?
- Should incomplete EC labels be a separate prefix task, and which multi-label metric is primary?
- Is the main test intended to represent realistic transfer with close homologs, or should a cluster-cold identity cutoff be the primary result?
- Should test performance target natural EC prevalence or a balanced class-coverage sample? How many examples per EC are needed for calibration thresholds?
- Should HFSP be used as an external comparison set, and how will any accession/sequence overlap with the new split be handled?

## Verification gaps

- The 2026_03 UniProt figures in the project brief were not independently recomputed from that archived release. The live API check above is for 2026_04.
- Exact original supplementary split settings for each legacy DeepEC and ECPred dataset remain unchecked; use CLEAN, CARE and EC-Bench only for the cross-paper comparisons stated here.
