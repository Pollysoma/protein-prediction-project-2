# Recipe verification: what ContrasTED and ProtTucker actually say about their splits

- **Date:** 09.11.2026
- **Purpose:** settle seven points that `protocol/04_split_requirements.md` and its raw notes left ambiguous, before the split roadmap goes to the supervisors.
- **Sources read:** ContrasTED (Miller et al. 2026, bioRxiv 2026.08.28.747864v2), ProtTucker/EAT (Heinzinger et al. 2022, NAR Genomics Bioinf., lqac043), the EAT repository (`Rostlab/EAT`, cloned 09.11.2026) and the ContrasTED repository (`dmmiller597/contrasted`, cloned 09.11.2026, full git history inspected).
- **Citation conventions:**
  - `CT l.N` = line N of `pdftotext -layout` of the ContrasTED PDF. Section 4.4 sits at l.840-853; the margin numbers 292-301 printed in the PDF are given as "paper l.".
  - `PT raw l.N` = line N of plain `pdftotext` (no `-layout`) of the ProtTucker PDF, because the two-column layout interleaves. The "Data set" paragraph is raw l.198-226.
  - Numbers marked **[computed]** are ours, from the shipped files; they are not claims of the papers.
- **Verdict scale:** *stated* (the text says it), *implied* (follows from the text, code or numbers but is not said), *cannot tell*.

## The two source passages (for reference)

ContrasTED, Methods 4.4 (CT l.842-853, paper l.293-301):

> Training, validation, and test splits were derived from CATH v4.4 classes 1-3 ... we sampled one representative domain per superfamily into disjoint validation and test sets. To enforce strict sequence separation, CATH S100 domains sharing >= 20% sequence identity to any validation or test domain were removed using MMseqs2 profile search (sensitivity 7.5, three iterations, and 80% bidirectional coverage). Test superfamilies with no remaining training representatives after filtering were excluded. The resulting S20 benchmark comprises 116,301 training, 514 validation, and 1,028 test domains across 5,659 training superfamilies. In all Table 1 evaluations, the reference lookup set contains only the corresponding CATH training domains.

ProtTucker, Methods "Data set" (PT raw l.198-226), condensed: v4.3 CATH-S100 (123k); test300 and val200 "randomly split off" with two conditions (superfamily at most once; SSG annotation present); training set = S100 minus everything above 20% PIDE to val/test (MMseqs2, `--num-iterations 3`, `-s 7.5`, `--cov-mode 0`); then families above 100 members clustered at 95% PIDE / 95% coverage; representatives = train66k; lookup69k made the same way but ignoring val200; test300 then reduced to test219 by removing every protein with HVAL > 0 to any lookup69k protein.

---

## 1. ContrasTED: are validation and test drawn from disjoint superfamilies?

**Verdict: implied (strongly), not stated.**

**What the text says.** "one representative domain per superfamily into disjoint validation and test sets" (CT l.843-844). The word "disjoint" is attached to the *sets*, so on its own it could mean disjoint domains. Line 62 of the text adds "Each test set contains exactly one representative domain per superfamily", i.e. one-per-superfamily holds *within* each set; it does not say across the two.

**What the numbers say [computed from the shipped `data/cath-domain-sf-list.txt` and `data/cath-domain-seqs-S100-c123.fasta`].**

- c123 S100 has 120,799 domains in 5,659 superfamilies. 4,019 superfamilies have at least 2 domains, 3,133 at least 3, 1,640 exactly one.
- 514 + 1,028 = 1,542 = 3 x 514 exactly, a 1:2 ratio.
- Both readings fit numerically: 1,542 distinct superfamilies (disjoint reading) or fewer (overlap reading) are both at most 4,019. So the sizes do not decide it.
- 120,799 - 1,542 - 116,301 = 2,956 domains were removed by the 20% filter.

**What decides it: the code.** `src/contrasted/datamodule.py` has a validation check that raises "validation and test H-class overlap is non-zero" and "test H classes not subset of training" (around l.322-330, enabled by `strict_split_checks`). `tests/test_s20_split_lock.py` asserts `not val_sfs & test_sfs`, `train_sfs == universe` and `len(universe) == 5_659`. These encode the intent that val and test superfamilies are disjoint and that every superfamily has a training member. The test only runs when `data/cath_s20_split/` is on disk, and it is not (see Q7), so we cannot run it. Reading it is enough to call the disjoint reading the authors' design, not to prove the published sets satisfy it.

**Consistency with "test superfamilies with no remaining training representatives were excluded".** Both readings fit this sentence equally, but there is a tension we cannot resolve:

- The final training set contains *all* 5,659 superfamilies, and the lookup/centroid set has "all 5,659 homologous-superfamily centroids" (CT l.384-385). A superfamily whose training members were all removed by the filter cannot be among them.
- So either the exclusion removed no query superfamily at the end, or the excluded query was handled in a way the paper does not describe (for example put back into training).
- The exact 2:1 ratio (514:1,028) suggests the exclusion either removed nothing from the final counts or ran before the 1:2 assignment. That is an inference from a round number, not a statement.
- The number of excluded superfamilies is not given.

**Default if we must choose.** Disjoint superfamilies. It is the only reading under which a validation-calibrated threshold (CT l.386-387) transfers to a test set of *other* superfamilies without the calibration set having seen the same label, and it is what the code checks.

## 2. ContrasTED: did they also drop queries that still had a hit in the reduced lookup?

**Verdict: no equivalent step is described. Not stated either way in so many words; the evidence points to "no".**

- The only query-side removal in 4.4 is superfamily-based: "Test superfamilies with no remaining training representatives after filtering were excluded" (CT l.850-851). That drops queries that have *no label in the lookup*, not queries that have a *residual hit*.
- The filter works in one direction: lookup domains at >= 20% to any query are removed (CT l.846-848). No residual-hit check, no HVAL or E-value criterion, and no second pass appear anywhere in 4.4 or the main text.
- The full 1,028 queries are evaluated in every method of Table 1 ("n = 1,028", CT l.229-230); the only reduced denominator is HH-suite's reachable subset of 959 (profile coverage), which is a property of that tool.
- The shipped test reads the MMseqs2 parameters from `LOCK.json` (`min_seq_id` 0.2, `cov_mode` "0"), consistent with a single search-and-remove step. `LOCK.json` itself is not in the repo.

**Why it matters.** ProtTucker's step cost 27% of its queries (300 -> 219) and used a length-aware yardstick. For long alignments HVAL > 0 is almost the same as PIDE > 20% (about 20.7% at L = 250, 19.5% at L >= 450 **[computed]** from the HFSP formula in our protocol), but for short alignments it demands more (29% at L = 100, 42% at L = 50). So a second HVAL step is not a stricter identity cut; it catches pairs the profile search missed or scored under a different identity definition. ContrasTED's one-step filter therefore may leave residual high-PIDE pairs in; we cannot tell how many.

**Default.** Do both steps (filter the lookup, then check the queries), as our protocol already says (R5), and report the count after each step. We cannot claim ContrasTED did this.

## 3. ContrasTED: how was the single representative domain per superfamily chosen?

**Verdict: cannot tell from the source.**

- The text says "one representative domain per superfamily" (CT l.294). "Representative" is not defined. It does not say random, by resolution, by length, or as an S35/S40 representative.
- The input is the S100 file. `data/cath-domain-seqs-S100-c123.fasta` (headers `>cath|4_4_0|1oaiA00/561-619`) and `cath-domain-sf-list.txt` (domain ID, superfamily) carry no cluster, resolution or S35 information, so the choice could not have used anything in the shipped files.
- S100 has ~2.9 times more domains than the sampled queries; any rule that picked "the" representative (longest, best resolution) would have had to read extra CATH files. Nothing in the paper mentions one.
- Because the word "sampled" is used (CT l.294: "we sampled one representative domain"), random sampling is the most natural reading, but "sampled" does not exclude a criterion.

**Default.** Uniform random choice among the superfamily's S100 domains with a fixed seed. Choosing by length or resolution would bias the query set toward easy, well-resolved folds, and an S35/S40 representative would not be a random draw from the superfamily either.

## 4. ContrasTED: was the val/test assignment random, stratified, seeded?

**Verdict: unstated (assignment rule and seed).**

- Section 4.4 gives no stratification (class, family size or otherwise), no sampling procedure for the 1:2 allocation and no seed.
- The 1:2 ratio (514:1,028) is exact, so some rule fixed the proportions; it is not described.
- Seeds in the repo do not belong to the split: "seed 40" (and 41, 42, "mean of seeds 40-42", CT l.268) are model-training seeds; `configs/train/held_out_superfamilies_s20.yaml` mentions "seed 42" for a *different* experiment, the 50-superfamily absent-class holdout (>= 11 members, "uniform").
- Old git history contains `scripts/create_cath_datasets.py` (first commit 2025-10-15, untracked 2026-05-05). It does use `random.seed(42)` and stratified validation sampling, but it clusters with `easy-cluster`, uses no profile search, no `--num-iterations`, and is a different, earlier protocol (S30-style multi-identity test sets). It is not evidence for how the published S20 split was drawn. No commit in the history contains `num-iterations`.

**Default.** Random, class-stratified if we need balanced classes, with a seed recorded in the manifest. We should say plainly that ContrasTED's assignment is unspecified, so we do not claim comparability on this point.

## 5. ProtTucker: the same four questions for test300 / val200 / lookup69k

### 5a. Are val and test superfamilies disjoint? **Stated.**
"every homologous superfamily appeared maximally once in test300 ∩ val200" (PT raw l.202; the "∩" is presumably a typo for "∪", since an intersection of the two sets is not what is meant). Checked on the shipped files **[computed]** (EAT `data/ProtTucker/*.fasta` + `data/cath_v430_dom_seqs_S100_161121_labels.txt`): test300 has 300 distinct H-codes, val200 has 200, overlap 0. test219 has 219.

### 5b. Residual-hit step? **Stated: yes, for test only.**
Lookup69k was built by the 20% filter against test300 and val200-ignored, then "we further removed any protein from test300 with HVAL > 0 (10) to any protein in lookup69k (219 proteins, dubbed test219)" (PT raw l.222-225). val200 got no such step. Not stated: which tool or alignment produced the HVAL values, the coverage rule, or what identity definition feeds HVAL (already noted in the requirements document).

### 5c. How were the 300 and 200 chosen?
**Stated: "randomly split off from CATH-S100"** (PT raw l.200-202), with two conditions: (1) each homologous superfamily at most once across the two sets, (2) SSG annotation. Beyond that:
- It is random over S100 *domains* subject to those conditions; no per-superfamily "representative" is mentioned.
- No restriction to superfamilies with >= 2 members is stated, and the files show none was applied **[computed]**: 27 of the 300 test queries and 14 of the 200 val queries come from superfamilies that have only that one domain in S100. For these the H-level answer is not in the lookup at all; this is consistent with 69 of the 219 test219 superfamilies (and 79 of test300's) having no lookup69k member.
- All CATH classes occur: test300 has classes 1/2/3/4/6 = 93/77/123/6/1 **[computed]**; the SSG condition therefore does not exclude classes 4 and 6.

### 5d. What exactly did the SSG condition require?
**Stated only as a definition, not as a selection rule: "each protein in test300 & val200 has a so called Structural Sub-group (SSG) annotation, i.e. clusters of domain structure relatives that superpose within 5 Å (0.5 nm), in CATH"** (PT raw l.202-205). So the requirement is that the domain has an SSG (structure-clustered) assignment in CATH. Not stated: which SSG level, how many domains lacked one, how much the condition removed, and why it was imposed (presumably so a structural neighbour exists for the T/A levels). We cannot reproduce it from the shipped label files: they carry only `domain_id,C.A.T.H` (two columns), no SSG information.

### 5e. Assignment of superfamilies to val vs test: random? seeded?
**Random (stated: "randomly split off"); no stratification stated; no split seed stated.** `train_prottucker.py` sets `SEED = 42` for *training* (l.927-929); `get_predef_splits` only reads predefined FASTAs and therefore does not construct the split.

## 6. Both papers: what is the lookup drawn from, and is it itself clustered?

### ContrasTED. **Stated for the source; clustering unstated, implied absent.**
- The lookup is "only the corresponding CATH training domains" (CT l.301): the c123 S100 domains of v4.4.0 left after the 20% filter (116,301 domains).
- No redundancy reduction of the lookup is mentioned. The arithmetic is consistent with none: 120,799 - 1,542 queries - 2,956 filter removals = 116,301, so there is no room for a substantial further clustering step (a handful of removals cannot be excluded). **Implied** only.
- Classes 1-3 only (class 4 and 6 are not in the benchmark).
- The lookup is the *training* set of the model too (the model is trained on it plus TED); in ours the lookup is never trained on.

### ProtTucker. **Stated, with one ambiguous term.**
- The source is CATH-S100 v4.3, "123k proteins" (the shipped label file has 122,727 IDs **[computed]**), all classes.
- Lookup is redundancy-reduced: (i) remove everything at > 20% PIDE to any val or test protein (profile search, `--num-iterations 3 -s 7.5 --cov-mode 0`); (ii) cluster "large families (>100 members)" at 95% PIDE and 95% bidirectional coverage and keep representatives (PT raw l.206-216). Result 66k (train66k, using both val and test) and 69k (lookup69k, ignoring val200).
- **Ambiguity:** "families" is not defined (superfamily? S35 family?). **[computed]** 221 superfamilies exceed 100 members in S100; lookup69k still has a superfamily with 4,989 members and 92 above 100, so the clustering is a 95%-dedup, not a cap at 100.
- **Consistent with the text [computed]:** lookup69k and train66k differ in exactly the way the text predicts: 3,761 IDs only in lookup69k (val200 relatives that train66k removed), 208 only in train66k (test300 relatives that lookup69k removed). 1 val200 protein is itself in lookup69k; test300 and val200 are not in train66k, and test300 is not in lookup69k.
- Not stated: how a cluster representative is chosen (MMseqs2 default would be the longest), and whether the 95% clustering ran before or after the 20% filter (the text order says after).

## 7. Is a split manifest or split-construction script published?

### ContrasTED repo (`dmmiller597/contrasted`, main). **No.**
- `data/` contains only the whole c123 S100 FASTA and the domain-to-superfamily list. These are the *inputs*, not the split. No `train_ids.txt`, `val_ids.txt`, `test_ids.txt`, `LOCK.json` or `data/cath_s20_split/` is tracked.
- The README says of the split FASTAs: "Those FASTAs are not in git." The test file states: "CI does not ship `data/cath_s20_split/`". `scripts/` is untracked since commit `b15b46c` (2026-05-05); the repo has no split-construction code and no MMseqs2 command anywhere in history (searched `num-iterations`).
- History held an older S30 split (`data/cath_s30_split/*`, 2026-05-06 to 2026-08-08) and a clustering-based generator script. Neither is the S20 split.
- The paper's data-availability statement promises "the S20 benchmark split ... assembled in a versioned archive for release with the preprint" (CT l.1038-1041, paper l.415-419), and the code section says the archive "will be linked from the repository when the preprint is posted". **As of 09.11.2026 we found no link.** Checkpoints ship (`checkpoints/`), the split does not.
- What we can recover without it: the counts (116,301 / 514 / 1,028 / 5,659; `datamodule.py` `S20_SPLIT_EXPECTATIONS`), the filter parameters (identity 0.2, `--cov-mode 0`, 80% coverage, sensitivity 7.5, three iterations), and the full S100 input. We cannot recover which domains are in each set.

### EAT repo (`Rostlab/EAT`, main). **Manifests yes, construction script no.**
- Machine-readable sets: `data/ProtTucker/{test300,val200,test219,lookup69k,train66k}.fasta` (IDs in the FASTA headers), plus `data/cath_v430_dom_seqs_S100_161121.txt` and `..._labels.txt` (ID, C.A.T.H). The README says data was also moved to Zenodo (record 14675997).
- No script that builds the split (no MMseqs2 call, no HVAL code, no random sampling of test/val). `train_prottucker.py` only reads predefined FASTAs. Note it also refers to `train74k.fasta`, a file that is not shipped (the repo has train66k).

---

## Consequences for our roadmap

Where the papers genuinely differ, or are silent, so that we must choose:

| Point | ProtTucker | ContrasTED | What we need to decide |
| --- | --- | --- | --- |
| Query sets | one test, one val; superfamilies disjoint (stated, verified) | one val, one test; disjointness only implied (code) | Make val/test superfamily-disjoint explicitly, and say it. |
| Queries from singleton superfamilies | allowed (27/300 test) | impossible by construction in practice (query SFs need a training member) | Decide: require >= 2 members in S100 so H-level is answerable (our requirements R12 already leans to this; ContrasTED-style). |
| Choice of the domain | random from S100, no "representative" | "representative", rule unknown | Pick a rule, record it: random with seed; do not use S35/S40 reps. |
| Residual-hit step on queries | yes (HVAL > 0; lost 27%) | not described | We choose: do it, and report counts after each step. |
| What happens to a query whose superfamily lost its lookup members | kept (69 of 219 test queries have no H-level answer) | excluded | We choose: exclude, publish the excluded IDs. |
| Lookup redundancy | 95%/95% dedup of families > 100; lookup differs for val vs test | no dedup stated; lookup = all remaining S100 | Choose whether to dedup. Report the member-count distribution (R10). |
| Lookup differs per query set | yes (train66k vs lookup69k) | no | Ours: one lookup for both query sets (calibration and test need the same one). |
| Classes | all | 1-3 only | Open decision 1 stays open. |
| Alignment parameters | `--cov-mode 0`, `-s 7.5`, 3 iterations; identity mode unstated | same plus `-c 0.8`; identity mode unstated | Write ours down in full, including identity and coverage. |
| val/test assignment, seed, stratification | random, no seed | unstated, no seed | Ours: seeded and stratified by class; record seed. |
| Published manifest | yes (IDs only; no script) | no (promised) | Do not depend on ContrasTED's list. Reproduce at best with the same parameters and compare counts (116,301 / 514 / 1,028 / 5,659). |

Two corrections to our own notes follow from this check:

1. "ContrasTED samples one domain per superfamily into disjoint validation and test sets" (R9 evidence) is true for the sets but the superfamily-level disjointness is an inference from code. Say "implied".
2. "ProtTucker allows each superfamily at most once across test300 and val200" is stated, but it does not mean every query has a lookup relative: 27 test300 queries are the only S100 member of their superfamily.
