# 05: How we build the data

- **Date:** 09.11.2026
- **Issue / PR:** #26, #27
- **Status:** proposal for the supervisor meeting
- **Evidence behind every number here:** [`04_split_requirements.md`](04_split_requirements.md) for the rules and where they come from, [`experiments/05_split_pilot/`](../experiments/05_split_pilot/) for the test run.

## What we are building and why

Our method is simple: take a protein, find the most similar protein in a labelled library, copy its label. To measure how well that works we need three sets.

| Set | What it is for |
| --- | --- |
| **lookup** | the library we search in. Every entry has a label. |
| **calibration** | questions we practise on |
| **test** | questions we grade on, used once |

Two sets would be enough if we only wanted an accuracy number. We want more than that: we want to say "if the match is closer than X, you can trust it". X is a number we have to fit from data. If we fit X on the test questions, the test stops being a test. So we fit on calibration and leave test alone until the very end.

## Where the data comes from

CATH v4.4. It is a catalogue of protein pieces, called domains, sorted by shape into families. 123,251 domains in 6,630 families. The family is what we try to predict.

Two families we use as examples below: `2.60.40.10` is huge, 10,133 domains. `1.10.8.130` has exactly two, `1gvnA00` and `2p5tA00`.

## The seven steps

**1. Start with all of CATH v4.4.** 123,251 domains, 6,630 families.

**2. Keep families with at least two domains.** One domain becomes the question, so at least one other has to stay in the library as the answer. 4,339 families qualify. The rest stay in the library but never produce a question.

**3. Take one question per family.** Just one, picked at random with a fixed seed. Its siblings stay in the library. One per family stops the few huge families from dominating everything. Both papers we compare against do the same.

**4. Deal the families out, half to calibration and half to test.** A family goes entirely to one side. The two sides never share a family.

This is the step where a plain random split would be wrong. If we shuffled the 123,251 domains instead, near-identical twins would end up one in calibration and one in test, and we would be measuring memory instead of transfer.

**5. Remove library entries that are too similar to a question.** Anything above 20 % sequence similarity to any question is deleted. We use MMseqs2 for this. In the pilot it removed 21,950 of 121,251 entries, leaving 99,301.

Why: if a near-copy of the question sits in the library, finding it is trivial and proves nothing. Plain sequence search already solves that case in a second. 20 % is the standard line in this field, below which sequence comparison stops being reliable.

Two things worth knowing about this step. It runs **once**, before anything else, and produces one fixed library that every question searches. Nothing is filtered later, at search time. And an entry is deleted if it is too similar to *any* question, not just to yours, so most of what is missing from your library was removed because of somebody else's question.

**6. Drop questions that no longer have an answer.** Step 5 can delete a family's last remaining entry. Then the question is unanswerable and has to go.

Example: `1.10.8.130` has two domains. `1gvnA00` becomes the question, so `2p5tA00` is the only possible answer. It is more than 20 % similar to the question, so step 5 deletes it. Nothing in the library carries that label any more, so every answer would be wrong by construction. We drop the question.

This hits small families hardest:

| Family size | Keeps a usable answer |
| --- | --- |
| 2 domains | 25 % |
| 3–10 | 53 % |
| 11–100 | 88 % |
| more than 100 | 98 % |

Being picked is not affected by size: every eligible family gets exactly one ticket, whether it holds 2 domains or 10,000. The imbalance appears only here, at survival. What comes out the other side is weighted towards large families, which are also the easier ones, so we report results split by family size rather than as one average.

Expected size after this step: about 1,240 questions in calibration and about 1,240 in test. Roughly 57 % of 2,170 per side. The pilot drew 1,000 per side and kept 572 and 576.

**7. Freeze.** Three lists of IDs, the random seed, checksums of the input files, the MMseqs2 version, a count after every step, and the script that reproduces all of it. After this the data does not change. If it has to, it gets a new version number.

## The one real problem

After step 5, 94–96 % of the remaining questions have no detectable sequence match in the library at all. That is not a bug. It follows directly from deleting every close relative, and it is exactly how the published benchmark we reproduced was built.

But it blocks two things we need:

- The project brief asks for accuracy grouped by how similar the question is to the library. Those groups would be empty.
- A cutoff that gives 95 % reliability probably only exists among close matches. On an all-remote set the best model is right about 64 % of the time, so there may be no cutoff at all to find.

**Our fix: same questions, two libraries.**

| Library | Contents | What we get from it |
| --- | --- | --- |
| **remote** | after the 20 % deletion, 99,301 entries | the headline result, directly comparable to the published work |
| **full** | everything except the questions, 121,251 entries | the similarity groups the brief asks for, and a range where a cutoff actually exists |

The questions are identical in both. Only the library changes. We compute the embeddings once, so the second version costs nothing but search time. Step 6 does not apply to the full library, because nothing is deleted there.

## Decisions we have made

| | Decision | Why |
| --- | --- | --- |
| A | One question per family, picked at random with a fixed seed | Picking the longest or the best-resolved domain would bias towards long or well-studied proteins. No paper defends either. |
| B | 20 % similarity, 80 % coverage, MMseqs2 | Copied exactly from ProtTucker; ContrasTED and CATHe use the same. |
| C | Accept that large families survive more often, and report results split by size | Correcting it would trade one bias for another. Showing the split is honest and lets others compare. |
| D | Build both libraries, remote and full | One gives comparability with the literature, the other gives what the brief asks for. Cheap, because the embeddings are shared. |
| E | Keep CATH classes 4 and 6, but report them on their own line | They are about 2 % of questions, so folding them into the headline adds noise. ContrasTED deletes them; we would rather keep and label them. |
| F | On the remote library, also apply the length-aware version of the 20 % rule | A flat 20 % ignores length. One pilot question sat at 19.9 % similarity over 412 residues, which is still findable by sequence search. The length-aware rule (HVAL) catches that; it is also what ProtTucker used. |
| G | Declare the reliability targets in advance: 90, 95, 99 % | Picking them after seeing results would be cherry-picking. We publish "not reachable" where no cutoff exists. |

Tell us if you disagree with any of these. D and F are the two where your opinion would help us most, because they change what the benchmark measures rather than how it is built.

## Questions for you

1. **The other group.** Who is working on the same project, and how do we reach them? We would like to avoid building the same thing twice and to agree on a shared format.
2. **CATH v4.3 as well, or only v4.4?** Our plan is to build the new data on v4.4 and keep our v4.3 reproduction as the anchor to the published numbers, since the published benchmark is v4.3. The two versions are almost the same size, so this is about comparability, not about more data. Is that what you expect?
3. **Two libraries or one?** Decision D above.
4. **The stricter remoteness rule?** Decision F above.

## What happens after the meeting

1. Freeze the input files. One day.
2. Build and check the data, both libraries. Two days.
3. Compute embeddings for all models over the library and both question sets. One to two days on the GPU.
4. Fit the cutoffs on calibration only, then freeze them.
5. Run the test once.
6. Then the same structure for EC numbers, which needs its own agreement because the labels are messier there.

## Two traps we already hit

Both are in the script now, but worth stating so nobody repeats them.

- A flat 20 % cut does not mean the pair is undetectable. Over 412 residues, 19.9 % similarity is still a real match.
- MMseqs2 silently caps how many matches it reports per query. Raising that cap changed the library from 100,540 to 99,301 entries, so the first version of the filter was too weak.
