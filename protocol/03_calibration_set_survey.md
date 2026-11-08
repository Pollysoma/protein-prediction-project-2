# 03: What a CATH calibration set needs, and what the data allows

- **Date:** 08.11.2026
- **Issue / PR:** #22, #23
- **Code:** [`experiments/03_calibration_set/`](../experiments/03_calibration_set/)
- **Status:** survey done, construction rules awaiting a decision

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

## Proposed construction

Pending the decisions below: 1,500 queries, 300 from each of the five bands up to HVAL 20, at most 5 per superfamily, seed fixed, sampled from the 49,729 survivors; the very close band above HVAL 20 contributes the remaining anchor points. Split into two folds of equal band composition for fit and check.

## Open decisions

1. **Queries whose superfamily is missing from lookup69k.** 669 of the 4,150 remote candidates have no possible correct answer. Excluding them matches the paper's masking and keeps our numbers comparable; including them matches what a user faces, since nobody knows in advance whether the library contains the right family, and excluding them makes the reliability score optimistic. Proposal: build one set, record both, fit the headline mapping on all queries and report the masked variant alongside.
2. **Difficulty mixture.** Stratified across the whole range as above, or test219-like and all-remote. Proposal: stratified, for the reason in requirement 4.
3. **Stay in v4.3 or move to v4.4 now.** Proposal: build the v4.3 set now to develop the method, and plan the v4.4 rebuild for the final benchmark.

## Next steps

- Decide 1 to 3, with the supervisors where needed.
- Sampling script, frozen ID list, checksum.
- Embed the calibration queries with every pLM and fit the first curves.
- Repeat the exercise for EC once that dataset exists.
