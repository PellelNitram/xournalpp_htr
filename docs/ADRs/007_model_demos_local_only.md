# ADR 007 – Model Demos: Local-Only, No Per-Model HuggingFace Space

- Date: 2026-05-17
- Status: Accepted
- PRD: None
- Drivers: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))
- Deciders: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))

## Context

[ADR 002](002_use_HuggingFace_ecosystem_for_ML.md) adopted the HuggingFace
ecosystem and [ADR 006](006_model_registry_and_training_environment.md) defined
per-model training subfolders under `xournalpp_htr/training/<model>/`.

The first integrated model, WordDetector, was originally shipped with a
[Gradio](https://gradio.app/) app deployed as a HuggingFace
[Space](https://huggingface.co/spaces/PellelNitram/xournalpp_htr_WordDetectorNN),
including a Supabase-backed "donate your image" data-collection path. During the
review of [PR #99](https://github.com/PellelNitram/xournalpp_htr/pull/99) it
became clear that a bespoke hosted Space plus a Supabase schema/bucket *per
model* is disproportionate maintenance: the actual recurring need is simply to
check whether a freshly trained checkpoint detects anything sensible. The Space
deploy was also a manual file copy (documented as "future work" to automate),
and every new model would multiply this cost.

This ADR fixes the demo story for **all** models, not just WordDetector.

## Decision

Every contributed model ships a **local demo**: a small, dependency-light
script in its `training/<model>/` folder (conventionally `demo.py`) that runs a
trained checkpoint on a given image — or on bundled example images when none is
supplied — and writes the annotated output to disk for visual inspection.

- No per-model HuggingFace Gradio Space.
- No web UI by default; no telemetry / data-donation / Supabase.
- Only base + the model's `training-<model>` extra; no `gradio`/`supabase`
  dependency for the demo.

The existing WordDetector HF Space, its Supabase event logging, the HF Space
front-matter/`.env`, and the related demo extras are removed accordingly.

This ADR governs **interactive demos only**. HF Hub remains the store for model
artifacts (ADR 006) and datasets (ADR 002); none of that changes.

## Rationale

A local script has no infrastructure, no secrets, and no deploy step; it runs
offline, is uniform across models, and exactly matches the need (a fast "does
the trained model work?" check). Removing the Supabase data-donation path also
removes a privacy/consent surface that was not worth maintaining per model.

## Consequences

### Pros

- Markedly lower maintenance: no per-model Space, CI deploy, or Supabase infra.
- No secrets or hosted services tied to a model's release.
- Demo runs offline and behaves identically on every contributor's machine.
- A single, consistent expectation for every future model contribution.

### Cons

- No publicly hosted, click-to-try playground for a model. If one is ever
  genuinely wanted, a maintainer can stand one up ad hoc from the local demo
  code; it is no longer a standing per-model obligation.
- Contributors must include a local demo as part of the model's PR.

## Alternatives

- **Keep the HF Space and automate deployment via CI/CD.** Rejected:
  per-model Spaces plus CI/CD plumbing is unbounded maintenance for marginal
  value, and grows with every new model.
- **One shared multi-model HF Space.** Rejected: still hosted infrastructure
  and it couples otherwise independent model releases.
- **No demo at all.** Rejected: a cheap sanity check that a trained model works
  is genuinely useful; as a local script its cost is negligible.
