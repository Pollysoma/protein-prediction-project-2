# 03: What a CATH calibration set needs, and what the data allows

- **Date:** 08.11.2026
- **Issue / PR:** #22, #23
- **Code:** [`experiments/03_calibration_set/`](../experiments/03_calibration_set/)
- **Status:** survey done; it moved the calibration set from CATH v4.3 to v4.4

## Question

Thresholds and the reliability score have to be fitted on labelled data that is not the test set. We need a third role next to lookup69k and test219. Before building it: what must such a set satisfy to do its job, and does CATH v4.3 still contain enough material for it?

## Requirements

What the set is for: estimating, per pLM, task and CATH level, the probability that a transfer is correct given the distance to the nearest neighbour, and from that a distance cutoff at a target precision. Everything below follows from that.

1. **Same lookup set as the test.** A distance cutoff is a property of the model *and* the library it searches. Calibrating against a different library would shift every distance. All calibration queries are searched against lookup69k.
2. **Not in the lookup set.** A query that is also in the library finds itself at distance 0. The pool excludes lookup69k by construction.
3. **No close relative of a test query.** Otherwise the threshold is partly fitted on the test material. We drop every candidate with HVAL > 0 to any test219 query, the same criterion the paper used to separate test219 from lookup69k.
4. **Covers the distance range we want to threshold, not only the remote end.** This is the requirement that decides the design. test219 is deliberately all-remote, and at superfamily level even the best model is right in 64 % of cases there. A 95 %-precision region only exists among close neighbours, so a calibration set with the same profile as test219 could never locate a 95 % cutoff. We therefore sample across the whole range of sequence similarity to lookup69k. What transfers to the test set is the conditional curve, precision given distance, not the average accuracy of the calibration set, so a deliberately non-representative mixture is sound. The price is that the average accuracy on the calibration set is meaningless and must never be reported as a result.
5. **Enough queries where the curve is read off.** Reading a 95 % point with a 95 % CI of about ±4 points needs on the order of 300 queries in the relevant distance region; with ±3 points, about 600. This has to hold per CATH level after singleton masking, which in test219 removes 31 % of the queries at level H.
6. **Enough independent superfamilies.** Queries from the same superfamily are not independent observations: they share the same neighbours and succeed or fail together. The usable sample size is closer to the number of superfamilies than to the number of queries, so we cap the number of queries per superfamily and bootstrap confidence intervals over superfamilies rather than queries.
7. **Sampled on sequence criteria only, before any embedding is computed.** Selecting queries using the distances we are about to calibrate would bias the curve.
8. **Two folds.** Choosing a setting (distance measure, k, centring) and fitting a threshold on the same queries overfits the threshold. Fit on one half, check on the other, and only then freeze.
9. **Frozen and shared.** One ID list with a seed and a checksum, the same queries for every pLM, versioned like the other splits.
10. **Label semantics decided in advance,** see the open decision below.

## Results

Candidate pool: CATH-S100 v4.3 minus lookup69k, test300 and val200, i.e. 52,623 domains. Of these, 2,894 have HVAL > 0 to a test219 query and are dropped, leaving 49,729. Grouped by HVAL of the best hit in lookup69k ([`pool_summary.csv`](../experiments/03_calibration_set/results/pool_summary.csv)):

| HVAL to lookup69k | kept | superfamilies | answerable | at most 5 per superfamily |
| --- | --- | --- | --- | --- |
| no hit | 1,377 | 105 | 1,046 | 329 |
| &le; -5 | 1,281 | 102 | 1,193 | 295 |
| -5 to 0 | 1,492 | 97 | 1,256 | 295 |
| 0 to 5 | 2,545 | 83 | 2,538 | 294 |
| 5 to 20 | 2,051 | 59 | 2,047 | 204 |
| &gt; 20 | 40,983 | 1,898 | 40,980 | 5,658 |

- **Remote material exists:** 4,150 candidates are as remote from lookup69k as the test219 queries, nineteen times the size of test219.
- **Its diversity does not:** those 4,150 come from 149 superfamilies, 122 of which exist in lookup69k. test219 has 219 queries from 219 different superfamilies. Capped at five queries per superfamily, the three remote bands yield 919 queries.
- **The pool is dominated by close relatives:** 82 % of it sits above HVAL 20, spread over 1,898 superfamilies.
- **Check of our settings:** for the published test219 queries our search finds HVAL &le; 0 against lookup69k for 208 of 219; 11 reach up to 6.7. The paper reports all 219 below 0, so our search is more sensitive than theirs. Our test-similarity filter is therefore at least as strict as the published one.

## Interpretation

- A calibration set can be built inside the frozen v4.3 benchmark, and it is large enough to fit the curves. It is not diverse enough at the remote end to treat each query as an independent observation: 122 superfamilies is the real sample size there, which gives a CI of roughly ±9 points on a precision estimate, not ±3.
- The reason is structural, not accidental. Proteins that are remote from lookup69k are mostly relatives of the test300 domains, which is exactly why they were removed from the lookup. The remote part of CATH v4.3 is largely consumed by the existing split.
- This limits the sharpness of a 95 %-precision threshold in the remote regime, which is the regime the project cares about most. It does not block the work: the machinery, the curves and the validation protocol can all be developed on this set.
- A larger benchmark built from CATH v4.4, where we define all three roles ourselves, is the way to get remote queries from many superfamilies. That is already planned as a later phase; the survey shows it is also what the calibration needs.

## Conclusion: build the calibration set in CATH v4.4, not v4.3

The survey was meant to decide how to carve a calibration set out of v4.3. Its answer is that the remote part of v4.3 is already spent, and the project brief points the same way: start from lookup69k and test219, then move to the non-redundant CATH v4.4 sets.

ContrasTED (Miller et al. 2026), the paper our brief cites for frozen ProstT5 at 78.5 %, builds exactly the benchmark we need and reports the calibration step we want to reproduce for raw embeddings:

- Splits from CATH v4.4 classes 1 to 3. One representative domain per superfamily is sampled into **disjoint validation and test sets**; from the remaining CATH S100 domains everything with at least 20 % sequence identity to a validation or test domain is removed by MMseqs2 profile search (`-s 7.5`, three iterations, 80 % bidirectional coverage). Result: 116,301 lookup domains over 5,659 superfamilies, 514 validation and 1,028 test queries. Test superfamilies left without a lookup representative are dropped.
- Thresholds: *"calibrated a distance threshold on the validation set by selecting the largest distance retaining at least 99 % empirical precision, and applied this cutoff unchanged to the test set"*. On the test set that gate kept 64.0 % of queries at 98.5 % precision, against 55.9 % and 36.4 % coverage for CATHe2 and ProtTucker. **Frozen ProstT5 embeddings reached no threshold at 99 % validation precision at all.**
- Numbers are reported both over all queries and over the subset whose superfamily is present in the reference set.

Three consequences for us:

1. **Calibration and test should be built by the same procedure**, one query per superfamily, drawn disjointly from the same pool. Then the threshold transfers by construction and the stratified sampling proposed above is unnecessary. Requirement 4 is satisfied by the construction instead of by a sampling scheme.
2. **One query per superfamily also fixes requirement 6.** There is nothing to cap, and every query is an independent observation.
3. **Raw embeddings may not reach a 99 % precision target.** We should state several targets (90, 95, 99 %), report coverage at each, and record "not reachable" as a legitimate result rather than lowering the target afterwards.

## Open decisions

1. **Queries whose superfamily is missing from the lookup set.** ContrasTED removes them at construction time and additionally reports accuracy over all queries next to accuracy over represented queries. Proposal: do both, exactly as they do.
2. **CATH classes.** ContrasTED uses classes 1 to 3; the ProtTucker benchmark uses all four. Proposal: keep all four so our numbers stay comparable with test219, and report class 4 separately if it behaves differently.
3. **Split sizes.** ContrasTED has 514 validation and 1,028 test queries. Proposal: the same order of magnitude, with the final numbers fixed by how many superfamilies survive the filter.

## Next steps

- Confirm the three points above with the supervisors.
- Build the v4.4 split: sample one domain per superfamily into disjoint calibration and test sets, filter the lookup set at 20 % identity, freeze IDs with a seed and a checksum.
- Keep lookup69k and test219 unchanged as the reproduction benchmark.
- Embed calibration and test with every pLM, then fit thresholds on calibration only.
- Repeat for EC once that dataset exists.
