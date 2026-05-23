# Issue Labels

This page defines the labeling taxonomy used in this repository. Apply at
least one **type** label to every issue, and one or more **area** labels
when the issue is scoped to a specific part of the project. **Meta** labels
are optional and signal triage state.

Aim for the minimum set that accurately describes the issue — typically one
type + one area.

## Type — what kind of work

| Label           | When to use                                                                                |
| --------------- | ------------------------------------------------------------------------------------------ |
| `bug`           | Something is broken or behaves incorrectly.                                                |
| `enhancement`   | New feature or improvement to existing behavior.                                           |
| `refactor`      | Code restructuring without changing behavior.                                              |
| `research`      | Exploratory work: spikes, evaluations, brainstorms, "try out X".                           |
| `documentation` | Changes scoped to docs, READMEs, ADRs, or in-code documentation.                           |
| `question`      | Open question or discussion — no defined deliverable yet.                                  |

## Area — where in the project

| Label          | When to use                                                                                |
| -------------- | ------------------------------------------------------------------------------------------ |
| `pipeline`     | HTR inference pipeline / runtime (`run_htr.py`, `inference_models.py`, etc.).              |
| `training`     | Model training loops, training data handling, training-time evaluation.                    |
| `models`       | Model architectures and the model registry.                                                |
| `plugin`       | Xournal++ Lua plugin and editor integration.                                               |
| `installation` | Installation, packaging, install scripts, PyInstaller, install modes.                      |
| `devex`        | Developer experience: CLI shape, type-checking, harness, agentic tooling.                  |
| `ci-cd`        | CI workflows, release automation, versioning.                                              |
| `architecture` | Cross-cutting architectural decisions that don't fit a single area.                        |

If an issue genuinely spans multiple areas, apply multiple area labels —
but prefer the most specific one when possible.

## Meta

| Label              | When to use                                            |
| ------------------ | ------------------------------------------------------ |
| `good first issue` | Well-scoped and approachable for a new contributor.    |
| `help wanted`      | Maintainers would welcome outside help on this issue.  |
| `duplicate`        | Tracks the same work as another issue.                 |
| `invalid`          | Not actionable as filed.                               |
| `wontfix`          | Acknowledged but intentionally not being addressed.    |

## Examples

- "Profile word detector training loop" → `research` + `training`
- "Fails to OCR with openDialog Lua error" → `bug` + `plugin`
- "Add CalVer versioning and automated release workflow" → `enhancement` + `ci-cd`
- "Decide on CLI shape: single entry point vs multiple commands" → `architecture` + `devex`
- "Add file logging to Lua plugin for easier debugging" → `enhancement` + `plugin` + `good first issue`
