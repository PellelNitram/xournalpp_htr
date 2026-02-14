---
description: Groups changes into logical chunks and performs multiple atomic commits based on CLAUDE.md rules.
argument-hint: [context or specific files to prioritize]
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git add:*), Bash(git commit:*)
---

# Task
1. Run `git status` and `git diff` to see all unstaged/staged changes.
2. Read the `CLAUDE.md` file to understand the required commit message style.
3. **Analyze and Group:** Categorize the changes into logical, atomic units (e.g., don't mix a refactor with a feature).
4. **Iterate and Commit:** For each logical group:
    a. Stage only the files/lines related to that specific group using `git add`.
    b. Generate a commit message following `CLAUDE.md` rules, incorporating any user context from $ARGUMENTS.
    c. Execute the commit.
5. Provide a summary of all commits made at the end.