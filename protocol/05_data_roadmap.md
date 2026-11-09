# 05: Data curation and split roadmap

- **Date:** 09.11.2026
- **Issue / PR:** #26, #27
- **Purpose:** the proposal to agree with the supervisors before anything is frozen. Requirements and their evidence: [`04_split_requirements.md`](04_split_requirements.md). Verification of what the reference papers actually did and a pilot run with real numbers: [`experiments/05_split_pilot/`](../experiments/05_split_pilot/).
- **Status:** proposal

## What the project documents already ask for

The brief (`docs/4_eat_baseline.pdf`, Exploration Step 2) says to "agree with your supervisors on splits and protocol, then freeze them", and lists the questions to answer first, among them: *how do you split lookup and query proteins so that results reflect real use, and how similar are queries to the lookup set?* It also asks for accuracy "binned by sequence identity to the lookup set". The project guide repeats this as Phase 2 and adds that splits must be versioned and frozen rather than changed afterwards. This roadmap is our draft answer.

## What the reference papers actually did

Checked against the papers and their code, not against their abstracts ([`recipe-verification.md`](../experiments/05_split_pilot/notes/recipe-verification.md)).

| | ProtTucker (2022) | ContrasTED (2026) |
| --- | --- | --- |
| Source | CATH v4.3 S100 | CATH v4.4 S100, classes 1-3 |
| Query sets | 300 test, 200 validation, each from distinct superfamilies | 1,028 test, 514 validation, one domain per superfamily |
| Which domain per family | "randomly split off", no seed, no stratification | not stated |
| Lookup filter | remove everything above 20 % identity to the query sets | same, with 80 % bidirectional coverage |
| Second, query-side filter | yes: drop test queries with HVAL > 0 to the lookup, 300 to 219 | no such step |
| Calibration separate from test | no, thresholds read off the test set | yes, this is the one thing they do properly |
| Split published | ID lists yes, construction script no | neither |

Neither paper documents its split well enough to reproduce. That is a low bar to clear, and clearing it is part of our contribution.

## How splits are made in this field

A random split over domains is not acceptable and is not used ([`splitting-practice.md`](../experiments/05_split_pilot/notes/splitting-practice.md)). Related proteins would land on both sides, and the benchmark would measure memorisation of families rather than transfer. The standard is to split at the level of the group, here the CATH superfamily, and then remove remaining sequence similarity between the groups. Randomness enters only inside that constraint: which superfamilies go to which role, and which domain represents a superfamily. That randomness must carry a fixed seed.

So the honest answer to "should we split randomly" is: random at the superfamily level, with a seed, stratified so the roles are balanced; never random at the domain level.

## What the pilot showed

We ran the proposed recipe end to end on CATH v4.4 with seed 42, 1,000 calibration and 1,000 test query superfamilies ([`results/`](../experiments/05_split_pilot/results/)). It is a pilot, not the frozen split.

| Step | Count |
| --- | --- |
| CATH v4.4 S100 domains | 123,251 |
| Superfamilies | 6,630 |
| Superfamilies with at least two domains | 4,339 |
| Candidate lookup before filtering | 121,251 |
| Lookup domains removed at 20 % identity | 21,950 |
| Lookup after filtering | 99,301, covering 5,775 superfamilies |
| Query families that lost every lookup representative | 428 of 1,000 calibration, 424 of 1,000 test |
| Surviving queries | 572 calibration, 576 test |
| Calibration-test pairs above 20 % identity | 0 |

The MMseqs2 step took six seconds on 96 cores, so the whole recipe is cheap to re-run.

**Three findings matter for the design.**

1. **Small superfamilies do not survive the filter.** Survival depends almost entirely on family size: 25 % for families with two domains, 53 % for 3 to 10, 88 % for 11 to 100, 98 % above 100. The mechanism is obvious in hindsight: in a two-member family, one domain becomes the query and the other is usually more than 20 % identical to it, so it is removed and the family has no answer left.
2. **What survives is almost entirely remote.** 96 % of calibration and 94 % of test queries have no detectable hit in the reduced lookup at all. That is not a defect; it follows from removing every relative above 20 % identity. It is the same construction that produced test219.
3. **Calibration and test come out comparable.** Class and size-band counts differ by at most a few families, and the remoteness distributions are close, with test marginally easier (94 % versus 96 % with no hit). This is the condition a threshold needs in order to transfer, so it must be measured and reported every time, not assumed.

Two traps the pilot caught, both worth keeping in the final script: a flat 20 % cut does not imply HVAL ≤ 0 (one test query sits at 19.9 % identity over 412 residues, which is HVAL 0.39), and MMseqs2's default cap on hits per query silently weakens the filter until it is raised and checked.

## The tension we need the supervisors to settle

Finding 2 collides with the brief. A strictly remote benchmark is what ProtTucker and ContrasTED built, and it is where embeddings are interesting. But on such a set there is no region of close neighbours, which means:

- a precision target of 95 % may be unreachable for raw embeddings, so no threshold can be fitted;
- the identity bins the brief asks for are nearly empty, because almost every query has no hit at all;
- the MMseqs2 baseline is close to its floor by construction.

**Our proposal: keep one set of queries and two lookup sets.** The query sets are fixed and shared. What changes is the library they are searched against.

| Track | Lookup | What it answers |
| --- | --- | --- |
| Remote | lookup after the 20 % filter | the headline question, where EAT beats sequence search. Comparable with ProtTucker and ContrasTED. |
| Full | every non-query domain, unfiltered | realistic transfer across the whole similarity range. Gives the identity-binned comparison the brief asks for, and a distance range in which thresholds can actually be calibrated. |

Embeddings are computed once, so the second track costs nothing but search time. Thresholds are fitted separately per track, because a threshold is a property of the model and the library it searches. The family-loss problem disappears on the full track, because nothing is removed.

## Roadmap

**Phase 1, freeze the inputs.** Pin CATH v4.4 S100 sequences and classification with checksums; record the MMseqs2 version. One day.

**Phase 2, agree the rules.** This document plus the open decisions below. The meeting.

**Phase 3, build and verify the CATH split.** Eligibility, assignment, representative choice, filtering, both tracks, and the verification suite: one query per superfamily, roles disjoint by ID and by sequence, calibration and test comparable in class, family size and remoteness, no query at the hit cap, every query answerable at H in its own track. Deliverable: three ID manifests, a counts table after every step, checksums, and the script that regenerates all of it. Two days.

**Phase 4, embed.** All twelve pLMs over the lookup and both query sets, pinned by checkpoint and revision, with an ID-coverage assertion per model. One to two days of GPU time.

**Phase 5, calibrate on the calibration set only.** Choose distance, k and any normalisation by a rule written down in advance; fit distance thresholds per pLM, task and level at 90, 95 and 99 % precision; record "not reachable" where it is not reachable. Freeze.

**Phase 6, test once.** Accuracy per level with bootstrap intervals, coverage and precision at the frozen thresholds, identity-binned comparison against MMseqs2 on the full track, and the residual-redundancy curve as evidence that the split does not leak.

**Phase 7, EC.** The same structure, but the labels are the hard part: evidence is attached per catalytic reaction, so an entry can mix measured and inferred EC numbers. A time-based split on UniProt releases is the natural analogue of the remote track. This needs its own agreement and should not be started before CATH is frozen.

## Decisions we need from the supervisors

1. **Two tracks or one?** We propose two, as above. If only one, which.
2. **Eligibility.** Families with two domains contribute a query that survives only a quarter of the time, which biases the benchmark toward large families. Options: require at least three domains, oversample small families to compensate, or accept the bias and report it. We propose using every eligible family, splitting them 50/50 between calibration and test, and reporting the survival by family size.
3. **CATH classes.** ContrasTED uses 1 to 3. We propose keeping all classes and reporting class 4 and 6 separately, since they are small (around 2 % of surviving queries in the pilot).
4. **Strict remoteness.** Add the HVAL ≤ 0 step on top of the 20 % filter, as ProtTucker did, or stop at 20 % as ContrasTED did. We propose adding it on the remote track only, and reporting how many queries it removes.
5. **Coverage parameter.** We used 80 % bidirectional coverage, following ContrasTED. A lower value filters more aggressively and makes the remote claim stronger but the lookup smaller. We propose keeping 80 % and reporting the sensitivity.
6. **Representative rule.** Random with a seed, or the longest domain, or the best resolution. We propose random with a seed: the alternatives bias length or bias towards well studied proteins, and no published source defends either.
7. **Precision targets.** We propose declaring 90, 95 and 99 % in advance, and publishing the threshold table even where a target is unreachable.

## What we are deliberately not doing

- No structure-based redundancy filter. It would be stronger, but CATH already groups by structure, and we cannot justify the extra complexity for this project.
- No profile-profile search at 10 % identity, which is the only recipe we found that genuinely supports a claim of topology-level separation. We stay at 20 % for comparability and state plainly what that does and does not guarantee.
- No conformal prediction in the first pass. The simple empirical rule, the largest distance that still meets the target precision on calibration, is what ContrasTED used and is enough to answer the research question. Conformal risk control is a sensible second step if time allows.
