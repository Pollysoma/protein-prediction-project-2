# AGENTS.md

Rules for everyone working in this repo, human or coding agent. `CLAUDE.md` is a symlink to this file, so edit only `AGENTS.md`.

**Project:** EAT, a benchmark for annotation transfer with raw pLM embeddings (PP2, WS 2026/27, RostLab, TUM).

## Git workflow

- **Never commit or push to `main`.** All changes reach `main` through a pull request.
- **One worktree per task.** Leave the main checkout on `main` and do not switch branches in it:
  ```sh
  git fetch origin
  git worktree add ../pp2-wt/<issue>-<slug> -b <issue>-<slug> origin/main
  ```
  Example: issue #12 becomes branch and worktree `12-cath-split`. Never share a worktree between two people or two agents working at the same time.
- Keep commits small and focused, and write messages in the imperative ("Add CATH split script").
- PRs stay small. Link the issue in the description (`Closes #12`) and get one teammate's review before merging. Squash-merge.
- **Clean up after the merge.** Delete the remote branch, then:
  ```sh
  git worktree remove ../pp2-wt/<issue>-<slug>
  git branch -d <issue>-<slug>
  git fetch --prune
  ```
  Leave no orphaned worktrees, branches, stashes or backup files (`*.bak`, `old_*`, `copy of …`). Git history is the backup.
- Never force-push to a branch someone else works on. Never rewrite `main`.

## Issues: tracker and protocol

GitHub Issues hold our tasks, decisions and meeting protocols. If it is not in an issue, it did not happen.

- **One issue per task**, with a clear title, a short goal and a checklist. Search before opening one to avoid duplicates.
- Assign yourself when you start work. Unassigned means nobody is on it.
- Labels: `protocol` (meeting notes), `presentation`, plus the GitHub defaults (`bug`, `documentation`, `enhancement`, `question`).
- **Meeting protocols:** one issue per meeting, titled `Protocol: <meeting>, DD.MM.YYYY`, with sections for Decisions and Action items. Turn each action item into its own issue or a checklist entry with an owner.
- Record results, dead ends and decisions as issue comments, with numbers and the commit hash. Do not let them live only in chat.
- Close an issue with a one-line outcome, or with the PR that resolved it. Close stale or duplicate issues instead of leaving them open.

## Data and code

- **No large files in git.** Embeddings, datasets and model weights go on Zenodo or Hugging Face (or stay local in `data/`, which is ignored). Commit the script that downloads or builds them, plus a checksum.
- No secrets, tokens or personal paths in the repo.
- Clear notebook outputs before committing. Move logic that others reuse out of notebooks into `src/`.
- Pin dependencies, and pin each pLM checkpoint by its exact name and revision.

## Scientific rules

- **Never use the test set to choose settings, thresholds or the reliability score.** Use the calibration set for that. The test set is for final numbers only.
- Freeze splits once agreed with the supervisors. Version them and do not change them silently.
- Every reported number comes with a confidence interval (e.g. bootstrap) and can be regenerated from a committed script and seed.
- "Raw" means a frozen pLM, mean over last-layer residues, and nothing trained on top. Keep anything else clearly separate.
- Compute all embeddings of one pLM with one tool. Do not use the Biocentral API for ESM C, because it currently returns wrong embeddings.

## For coding agents

- Read this file and the issue you are working on before changing anything. Stay within that issue's scope.
- Follow the git workflow above: worktree, branch, PR, cleanup. Do not merge your own PR unless a human asks you to.
- Prefer plain, readable Python that fits the existing code. Avoid speculative abstractions.
- Report results honestly. Never oversell, and flag anything you could not verify. Each of us has to be able to explain every line we present.
