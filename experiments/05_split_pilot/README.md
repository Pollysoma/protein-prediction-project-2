# CATH v4.4 split pilot

A dry run of the split recipe proposed in [`protocol/05_data_roadmap.md`](../../protocol/05_data_roadmap.md), to replace guesses with measurements before anything is frozen. **This is not the split.** The seed is provisional and the rules are still open.

```sh
WORK=/tmp/ra62ler_pp2/pilot_split bash run_pilot.sh
```

MMseqs2 takes about six seconds on 96 cores; the whole pilot runs in under a minute. The three ID manifests are written to the scratch directory on the server, not to the repository.

## Results

[`results/step_counts.csv`](results/step_counts.csv) has the count after every step, [`results/query_balance.csv`](results/query_balance.csv) the class and family-size balance of calibration against test, [`results/hval_distribution.csv`](results/hval_distribution.csv) how remote the surviving queries are, and [`results/reproducibility.txt`](results/reproducibility.txt) the exact commands, the identity definition and the checksums.

Headline: from 1,000 calibration and 1,000 test query superfamilies, 572 and 576 survive. The rest lose every lookup representative to the 20 % filter. Survival tracks family size closely, from 25 % for two-member families to 98 % above a hundred members. What survives is almost entirely remote: 96 % of calibration and 94 % of test queries have no detectable lookup hit at all.

## Notes

The two source notes behind the roadmap live in [`notes/`](notes/): [`recipe-verification.md`](notes/recipe-verification.md) settles what ProtTucker and ContrasTED actually specify, and [`splitting-practice.md`](notes/splitting-practice.md) covers how splits are built in this field, including the parts that turned out to be folklore rather than documented practice.
