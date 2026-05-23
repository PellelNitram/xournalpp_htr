# Committing code

- Authorship: Do not add Claude Code as author.
- Formatting: Use short, imperative commit messages (e.g., "Add user authentication."). 
- Punctuation: End the actual message with a period.
- Tagging: Suffix the actual message with `[CC]`.
- Atomic Commits: When performing multiple commits, group files by logical intent

# Code quality

- Run `make tests-not-slow` to confirm no regressions were introduced after code changes, where necessary

## Issue-Driven Workflow

- Default workflow: issue → branch (`gh issue develop <number>`) → PR referencing the issue (also using `gh`).
- When creating an issue with `gh`, apply labels from the taxonomy in [`docs/labels.md`](docs/labels.md): at least one **type** (`bug`, `enhancement`, `refactor`, `research`, `documentation`, `question`) and, when scoped to a part of the project, one or more **area** labels (`pipeline`, `training`, `models`, `plugin`, `installation`, `devex`, `ci-cd`, `architecture`). Keep the set minimal — usually one type + one area.

## Models

- When working on a model (training, export, inference, or docs), read [`docs/models/index.md`](docs/models/index.md) for the shared conventions before changing files under `xournalpp_htr/training/<model>/` or `docs/models/`.