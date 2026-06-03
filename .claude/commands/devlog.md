---
description: Generate devlog bullet points from merged PRs, closed issues, and git history for a date range.
argument-hint: <start-date> <end-date> (both inclusive, YYYY-MM-DD)
allowed-tools: Bash(gh issue *), Bash(gh pr *), Bash(git log *), Bash(gh api *)
---

# Task

Generate a concise list of bullet points summarising work completed in the date range, suitable for a devlog entry.

## Input

$ARGUMENTS must contain two dates in YYYY-MM-DD format: `<start-date> <end-date>`. Both dates are **inclusive**. If the arguments are missing or malformed, ask the user for the correct date range before proceeding.

## Data sources (in priority order)

Commits are the foundation of all progress. Merged PRs group commits into shipped units of work. Closed issues provide higher-level context and motivation. Use all three together:

1. **Merged PRs.** Use `gh pr list --state merged --search "merged:<start-date>..<end-date>" --json number,title,labels,mergedAt,body --limit 100` to list PRs merged within the range. These are the primary units of shipped work.

2. **Closed issues.** Use `gh issue list --state closed --search "closed:<start-date>..<end-date>" --json number,title,labels,closedAt --limit 100` to list issues closed within the range. Cross-reference with merged PRs — an issue linked to a merged PR enriches that PR's bullet; a closed issue without a corresponding PR still gets its own bullet if it represents meaningful work (e.g. research, decisions).

3. **Git history.** Run `git log --oneline --after="<start-date minus 1 day>" --before="<end-date plus 1 day>"` to list commits in the range. Scan for notable commits not covered by any merged PR or closed issue (e.g. drive-by fixes, direct-to-main pushes).

## Synthesise

Cross-reference the three data sources. For each bullet, classify how the work shipped:
- **PR + issue:** a merged PR that closes an issue — reference both, e.g. `(PR #114, #113)`.
- **PR only:** a merged PR with no linked issue — reference the PR, e.g. `(PR #111)`.
- **Issue only:** a closed issue with no merged PR — this is a decision, research outcome, or planning milestone. Tag it as such, e.g. `Decided on X (#71, decision)`.
- **Commits only:** notable commits not covered by any PR or issue. Name the specific change, e.g. `Added time prefix to experiment scripts (948f48f)`. Skip trivial or mechanical commits.

Each bullet:
- Summarises one logical piece of completed work (may combine a PR with its linked issue).
- Is written in past tense, concise, and non-technical enough for a devlog audience.

## Output

Print the following sections under a `## Devlog — <start-date> to <end-date>` heading. Do not include items outside the date range.

### TL;DR
Exactly **3 bullet points**, each a single punchy sentence (max ~15 words). Capture broad themes, not lists of changes. Written in past tense.

### Top 5 — User perspective
The 5 most impactful items from an **end-user's** point of view: bug fixes they'd notice, visible features, UX improvements, accuracy gains. Exclude internal refactors, tooling, and architecture — those belong in the developer section. Each bullet references the relevant PR/issue number(s). Write these as editorial highlights, not just repeating the "All changes" entry — explain *why* it matters to the user. End each bullet with a `(talk: ...)` hint — a short phrase suggesting how to present it on camera, e.g. "show before/after", "live demo", "mention the impact on accuracy", "explain the motivation".

### Top 5 — Developer perspective
The 5 most impactful items from a **developer's** point of view: architecture decisions, tooling, CI/CD, code quality, docs, DX improvements. Each bullet references the relevant PR/issue number(s). Write these as editorial highlights — explain *why* it matters for development. End each bullet with a `(talk: ...)` hint — a short phrase suggesting how to present it on camera, e.g. "show the CI dashboard", "walk through the ADR", "explain the tradeoff".

### All changes
The full bullet list from the synthesise step. Keep these factual and concise.
