# Committing code

- Authorship: Do not add Claude Code as author.
- Formatting: Use short, imperative commit messages (e.g., "Add user authentication."). 
- Punctuation: End the actual message with a period.
- Tagging: Suffix the actual message with `[CC]`.
- Atomic Commits: When performing multiple commits, group files by logical intent

# Code quality

- Run `make tests-not-slow` to confirm no regressions were introduced after code changes, where necessary