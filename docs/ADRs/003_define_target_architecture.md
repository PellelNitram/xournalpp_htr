# ADR 003 – Define Target Architecture

- Date: 2026-03-03
- Status: Accepted
- PRD: None
- Drivers: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))
- Deciders: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))

## Context

The project has grown organically and now needs a coherent architecture to support multiple entry points (CLI,
Xournal++ plugin, HuggingFace demo), a benchmarking workflow, and a path toward custom model training.
Without a defined structure, the inference code is duplicated or diverges across entry points, making
maintenance and extension harder.

## Decision

Adopt the following architectural foundations. Detailed decisions within each area are deferred to future ADRs.

**Central inference API.** All entry points (CLI, plugin, HF demo) call a single `compute_predictions(document: Path, pipeline: str)` function that returns word-level bounding boxes. This ensures identical inference behaviour regardless of how the tool is invoked.

**Package management.** Use `uv` for all dependency management and virtual environment handling.

**Typed I/O contracts.** Define inputs and outputs at every layer using dataclasses or Pydantic `BaseModel`s to make interfaces explicit and reduce integration errors.

**Model distribution via HuggingFace.** Models are downloaded on first use from the HuggingFace Hub and cached locally. This reuses the infrastructure already adopted in ADR 002.

**Semantic versioning with pipeline names.** The project uses semantic versioning (not date-based). Each released pipeline carries a human-readable name. The current pipeline will be named as part of the first versioned release; future versions are not required to support the first pipeline.

**Unit tests with pytest.** Every feature ships with `pytest` unit tests. Integration test tooling is to be decided in a future ADR.

**Word-level evaluation dataset.** The benchmark dataset consists of hand-annotated Xournal++ pages with word-level bounding boxes and transcriptions. Line-level and stroke-classification-level annotations are explicitly out of scope for now, as line grouping is treated as a post-processing step.

## Rationale

A single `compute_predictions` entry point eliminates divergence between the CLI and demo paths, which was the
main source of duplication. Typed I/O contracts make it practical to swap pipeline components and to write
focused unit tests. HuggingFace is already the chosen ML platform ([ADR 002](002_use_HuggingFace_ecosystem_for_ML.md)), so using its Hub for model
distribution requires no new dependency. Word-level annotation is the minimal useful granularity for an HTR
evaluation dataset and avoids premature complexity.

## Consequences

### Pros

- Inference logic is written and tested once, used everywhere.
- Typed contracts make AI-assisted development more reliable.
- HuggingFace Hub gives free, versioned model hosting with automatic caching.
- A concrete eval dataset definition unblocks benchmarking work.

### Cons

- `compute_predictions` must remain stable once entry points depend on it; breaking changes require a version bump.
- Word-level annotation effort is non-trivial and must be done manually.

## Alternatives

Keeping separate inference paths per entry point (CLI, demo) was rejected because it had already caused
divergence in the existing codebase. Date-based versioning was rejected in favour of semantic versioning to
communicate compatibility intent clearly to users.
