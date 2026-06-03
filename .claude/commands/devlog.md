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

Produce a full bullet list where each bullet:
- Summarises one logical piece of completed work (may combine a PR with its linked issue).
- References PR and/or issue numbers in parentheses, e.g. `(PR #114, #113)`.
- Is written in past tense, concise, and non-technical enough for a devlog audience.
- Standalone commits get their own bullet only if they represent meaningful work not already covered.

## Output

Print the following sections under a `## Devlog — <start-date> to <end-date>` heading. Do not include items outside the date range.

### TL;DR
Summarise all completed work into exactly **3 bullet points**. Each bullet should capture a broad theme, not a single issue. Written in past tense.

### Top 5 — User perspective
Pick the 5 most impactful items from the user's point of view (bug fixes, visible features, UX improvements). Each bullet references the relevant PR/issue number(s).

### Top 5 — Developer perspective
Pick the 5 most impactful items from a developer's point of view (architecture, tooling, CI/CD, code quality, docs). Each bullet references the relevant PR/issue number(s).

### All changes
The full bullet list from the synthesise step.
