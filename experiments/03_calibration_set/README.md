# Calibration set: survey of the candidate pool

Measurements needed before a CATH calibration set can be constructed. Nothing is sampled or frozen here. Issue #22, write-up with the proposed rules in [`protocol/03_calibration_set_survey.md`](../../protocol/03_calibration_set_survey.md).

## What it does

1. Builds the candidate pool: CATH-S100 v4.3 minus lookup69k, test300 and val200 (52,623 domains), with sequences from the CATH v4.3 release.
2. Runs the paper's redundancy-reduction search (MMseqs2 iterative profile search, `--num-iterations 3 -s 7.5 --cov-mode 0 -e 10`) for pool against lookup69k and pool against test219.
3. Turns each best hit into an HSSP value (`hval.py`): `HVAL = PIDE - 480 * L ** (-0.32 * (1 + e ** (-L / 1000)))` for alignment lengths 11 to 450, which is the criterion the paper used to make test219 non-redundant to lookup69k.
4. Summarises how many candidates fall in each remoteness band, how many must be dropped for similarity to a test219 query, and how many superfamilies each band contains.

```sh
WORK=/scratch/cal THREADS=96 bash survey_pool.sh
```

Both searches together take a few minutes on 96 cores.

## Results

[`results/pool_summary.csv`](results/pool_summary.csv). Bands are the HVAL of the candidate's best hit in lookup69k; `answerable` counts candidates whose superfamily exists in lookup69k; `cap5` counts what is left after keeping at most five candidates per superfamily.

| HVAL to lookup69k | candidates | kept after test filter | superfamilies | answerable | cap 5 |
| --- | --- | --- | --- | --- | --- |
| no hit | 2,469 | 1,377 | 105 | 1,046 | 329 |
| &le; -5 | 2,146 | 1,281 | 102 | 1,193 | 295 |
| -5 to 0 | 1,951 | 1,492 | 97 | 1,256 | 295 |
| 0 to 5 | 2,797 | 2,545 | 83 | 2,538 | 294 |
| 5 to 20 | 2,139 | 2,051 | 59 | 2,047 | 204 |
| &gt; 20 | 41,121 | 40,983 | 1,898 | 40,980 | 5,658 |
| total | 52,623 | 49,729 | 1,954 | 49,060 | 7,075 |

4,150 candidates are as remote from lookup69k as the test219 queries (HVAL &le; 0, including no hit) and survive the test219 filter, but they come from only 149 superfamilies, 122 of which exist in lookup69k.

[`results/test219_hval_to_lookup69k.csv`](results/test219_hval_to_lookup69k.csv) repeats the search for the published test219 queries as a check of our settings: 208 of 219 reach HVAL &le; 0, 11 do not (highest 6.7). The paper reports every test219 query at HVAL &le; 0, so our search is somewhat more sensitive than theirs. Our filter against test219 is therefore at least as strict as the published construction.
