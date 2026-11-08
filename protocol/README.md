# Experiment protocol

Lab notebook for the project. Each experiment gets one entry that records what we ran, what came out and what we think it means, written so that it can later be turned into the methods and results sections of the report.

- [`methods.md`](methods.md): data, evaluation and software shared by all experiments. Write it once here; entries only state what differs.
- [`template.md`](template.md): copy this for a new entry.

## Experiments

| ID | Date | Question | Key result | Code | PR |
| --- | --- | --- | --- | --- | --- |
| [01](01_prott5_reproduction.md) | 02.11.2026 | Can we reproduce raw ProtT5 EAT and the MMseqs2 baseline of Heinzinger et al. 2022? | Yes, exactly: H 64.0 ± 7.7 % (EAT), 34.7 ± 7.6 % (MMseqs2) | [`experiments/01_prott5_cath_reproduction`](../experiments/01_prott5_cath_reproduction/) | #17 |
| [02](02_table1_plms_distance.md) | 05.11.2026 | Do the other four pLMs of Table 1 reproduce, and does cosine distance change the result? | 16/20 values exact, 4 off by one query; cosine and Euclidean indistinguishable | [`experiments/02_table1_plms`](../experiments/02_table1_plms/) | #19, #21 |
| [03](03_calibration_set_survey.md) | 08.11.2026 | What must a calibration set satisfy, and does CATH v4.3 still hold enough material? | 4,150 remote candidates but only 149 superfamilies; construction rules proposed | [`experiments/03_calibration_set`](../experiments/03_calibration_set/) | #23 |
| [04](04_split_requirements.md) | 08.11.2026 | What must a benchmark split satisfy, according to the literature? | 28 requirements with evidence; no published EAT paper has a clean calibration set | [`experiments/04_split_requirements`](../experiments/04_split_requirements/) | #25 |

## Rules

- One entry per experiment, numbered like its folder in `experiments/`. Add a row to the table above.
- Every number in an entry comes from a committed script and a CSV under `experiments/<id>/results/`. Link the CSV instead of copying large tables; small summary tables are fine.
- Keep results and interpretation apart. Results are what the numbers say; interpretation is what we think they mean, with the evidence for it and how sure we are.
- Report deviations from the paper or from the plan, and their cause if known. Do not delete dead ends; mark them as such.
- Large outputs (embeddings, MMseqs2 databases, logs over a few hundred lines) stay out of git. Record where they were and how to rebuild them.
