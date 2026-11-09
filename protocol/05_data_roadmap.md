# 05: Data split roadmap

- **Date:** 09.11.2026
- **Issue / PR:** #26, #27
- **Status:** proposal, for the supervisor meeting
- **Backing material:** the 28 requirements and their sources in [`04_split_requirements.md`](04_split_requirements.md); what the reference papers actually did in [`recipe-verification.md`](../experiments/05_split_pilot/notes/recipe-verification.md); a full pilot run with real numbers in [`experiments/05_split_pilot/`](../experiments/05_split_pilot/).

## What we want at the end

Three sets of CATH domains:

| Set | Role | Rough size |
| --- | --- | --- |
| **lookup** | the labelled library we search in | ~100,000 domains |
| **calibration** | where we tune settings and fit distance thresholds | ~1,200 queries |
| **test** | the final numbers, touched once | ~1,200 queries |

Why three and not two: a distance threshold is a fitted number. If we fit "0.7 means 95 % precision" on the test set, the test set no longer tests anything. Calibration exists so that test stays untouched.

## The recipe

### Step 0 — the input

CATH v4.4, **S100** level: 123,251 domains, grouped into 6,630 superfamilies. S100 means duplicates are collapsed, nothing else, so it is the largest version of CATH. The superfamily (the "H" in CATH) is the label we try to predict and also our unit of splitting.

*Example:* superfamily `2.60.40.10` (immunoglobulin-like) holds 10,133 domains. Superfamily `1.10.8.130` holds exactly two: `1gvnA00` and `2p5tA00`.

### Step 1 — which superfamilies can supply a query

A superfamily needs **at least two domains**: one to become the query, at least one to stay in the lookup as the correct answer. 4,339 of 6,630 families qualify. The rest are single-domain families; they stay in the lookup but never supply a query.

### Step 2 — one query per superfamily

From each eligible family we pull exactly **one** domain as the query. All its siblings stay in the lookup.

*Example:* from `2.60.40.10` we take `3lmjL01` as a query; the other 10,132 domains remain in the lookup.

One query per family keeps the benchmark from being dominated by the handful of huge families. Both reference papers do the same.

> **Decision A — which domain?** We propose random with a fixed seed. The alternatives (longest domain, best crystal resolution) bias towards long or well studied proteins, and no paper defends either.

### Step 3 — split the families, not the domains

The 4,339 eligible families are dealt out **50/50 to calibration and test**. A superfamily goes entirely to one side; calibration and test never share a superfamily.

This is the one place where "random split" would be wrong. If we shuffled the 123,251 domains, two near-identical immunoglobulins would land one in calibration and one in test, and we would be measuring memorisation rather than transfer. Splitting by group and then removing residual similarity is the standard in this field ([`splitting-practice.md`](../experiments/05_split_pilot/notes/splitting-practice.md)).

Random *within* the constraint, with a seed, and stratified so both roles get a comparable mix of CATH classes and family sizes.

### Step 4 — the 20 % filter

Remove from the lookup **every domain that is more than 20 % sequence-identical to any query**. MMseqs2, iterative profile search, 80 % bidirectional coverage. This is verbatim the ProtTucker recipe and is also what ContrasTED and CATHe use; 20 % is the classic "twilight zone" boundary below which sequence search stops being reliable.

Pilot: 21,950 of 121,251 candidate lookup domains removed, leaving 99,301.

Why: without it, a query's near-twin sits in the lookup, nearest-neighbour transfer is trivial, and the benchmark says nothing about whether embeddings beat sequence search.

> **Decision B — coverage parameter.** We used 80 % bidirectional, following ContrasTED. We propose keeping it and reporting what a stricter value would change.

### Step 5 — drop the queries that no longer have an answer

The filter can delete a family's *only* remaining lookup member. Then that query has no correct answer at the H level and must go.

*Example:* `1.10.8.130` = {`1gvnA00`, `2p5tA00`}. `1gvnA00` becomes the query, `2p5tA00` is the only possible answer — and it is more than 20 % identical to the query, so step 4 deletes it. The family is now unanswerable and the query is dropped.

This hits small families hard. Survival in the pilot:

| Family size | Survives the filter |
| --- | --- |
| 2 domains | 25 % |
| 3–10 | 53 % |
| 11–100 | 88 % |
| more than 100 | 98 % |

Picking a query is size-blind: every eligible family gets exactly one ticket, whether it holds 2 domains or 10,000. The skew appears only here, at survival — so the surviving query set is enriched in large families.

> **Decision C — what to do about that skew.** Options: require at least three domains; oversample small families so survivors come out balanced; or accept it and report it. We propose **accepting and reporting**, with results stratified by family size, since any correction trades one bias for another. Using all 4,339 families should give ~1,240 queries per role.

### Step 6 — same filter between calibration and test

Calibration and test queries must not be each other's near-twins either, or a threshold fitted on one leaks into the other. Pilot: 0 violating pairs, because disjoint superfamilies already implies this in practice — but we check rather than assume.

### Step 7 — freeze

Three ID lists, the seed, checksums of the input files, the MMseqs2 version, a counts table after every step, and the script that regenerates all of it. After the freeze the split does not change; if it must, it gets a new version number.

## The one real problem

After step 4, **94–96 % of surviving queries have no detectable sequence hit in the lookup at all.** That is not a bug, it is the direct consequence of deleting every relative above 20 %. It is also exactly how the published `test219` was built.

But it collides with two things we need:

- the brief asks for accuracy **binned by sequence identity** to the lookup — those bins would be empty;
- a 95 %-precision threshold probably only exists among *close* neighbours. On an all-remote set the best model is right about 64 % of the time at the H level, so there may be no distance at which precision reaches 95 % and no threshold can be fitted.

**Our proposal: one pair of query sets, two lookups.** The queries are fixed. What changes is the library.

| Track | Lookup | Answers |
| --- | --- | --- |
| **remote** | after the 20 % filter (99,301) | does EAT beat sequence search in the hard regime; comparable to ProtTucker and ContrasTED |
| **full** | every non-query domain, unfiltered (121,251) | realistic transfer across the whole identity range; gives the identity bins and a range where thresholds exist |

Embeddings are computed once, so the second track costs only search time. Thresholds are fitted separately per track, because a threshold belongs to a model *and* the library it searches. Step 5 disappears on the full track, since nothing is removed.

> **Decision D — two tracks or one?** We propose two. If one, which.

## What we ask the supervisor to settle

| | Question | Our proposal |
| --- | --- | --- |
| A | Which domain represents a family? | random, fixed seed |
| B | Coverage for the 20 % filter | 80 % bidirectional, report sensitivity |
| C | Small-family skew | accept, report stratified by size |
| D | Two tracks or one | two |
| E | Keep CATH classes 4 and 6? (~2 % of queries; ContrasTED drops them) | keep, report separately |
| F | Add ProtTucker's stricter HVAL ≤ 0 step on top of 20 %? | yes, remote track only |
| G | Which precision targets to declare in advance? | 90 / 95 / 99 %, and publish "not reachable" where it is |

## Then

1. **Freeze inputs** — CATH v4.4 files, checksums, MMseqs2 version. 1 day.
2. **This meeting** — agree A–G.
3. **Build and verify the split** — both tracks, three manifests, verification suite. 2 days.
4. **Embed** — all pLMs over lookup and both query sets, pinned by revision. 1–2 days GPU.
5. **Calibrate** — on calibration only; thresholds per pLM, task, level, track. Freeze.
6. **Test once** — accuracy with intervals, coverage at the frozen thresholds, identity bins, leakage check.
7. **EC** — same skeleton, but the labels are the hard part (evidence is per reaction, so one entry can mix measured and inferred EC numbers). Needs its own agreement, after CATH is frozen.

## Deliberately not doing

- No structure-based redundancy filter — CATH already groups by structure.
- No profile-profile search at 10 % identity, the only recipe that would genuinely support a topology-level separation claim. We stay at 20 % for comparability and say plainly what it does and does not guarantee.
- No conformal prediction in the first pass. The empirical rule — largest distance that still meets the target precision on calibration — is what ContrasTED used and is enough. Conformal risk control is a sensible second step.

## Two traps worth keeping in the script

- A flat 20 % cut does **not** imply HVAL ≤ 0. One pilot test query sits at 19.9 % identity over 412 residues, which is HVAL 0.39.
- MMseqs2's default cap on hits per query silently weakens the filter. Raising `--max-seqs` from 5,000 to 20,000 changed the lookup from 100,540 to 99,301 domains.
