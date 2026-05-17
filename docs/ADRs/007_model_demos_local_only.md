# ADR 007 – Model Demos: Local Gradio App, No Per-Model HuggingFace Space

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
became clear that a *hosted* Space plus a Supabase schema/bucket *per model* is
disproportionate maintenance: the Space deploy was a manual file copy
(documented as "future work" to automate), the Supabase path added a
privacy/consent surface, and every new model would multiply this cost.

The interactive Gradio UI itself is valuable — it is the quickest way to eyeball
whether a freshly trained checkpoint detects anything sensible. What is *not*
worth the per-model cost is hosting it and wiring telemetry. This ADR fixes the
demo story for **all** models, not just WordDetector.

## Decision

Every contributed model ships an interactive **Gradio** demo (conventionally
`training/<model>/demo.py`) that loads a trained checkpoint and is launched
**locally** via `demo.launch()`.

- The Gradio UI is kept (image in, predictions out, model-specific controls).
- It runs locally on the developer's machine. It is **not** deployed as a
  per-model HuggingFace Space; there is no standing hosting or deploy pipeline.
- No telemetry, no data-donation, no Supabase. No HF Space front-matter / `.env`.
- The demo's dependencies (incl. `gradio`) are declared in the model's
  `training-<model>` extra (ADR 006 section 3), so a single
  `uv sync --extra training-<model>` provides training, export and the demo.

The existing WordDetector HF Space, its Supabase event logging, the HF Space
front-matter/`.env`, and the demo-architecture diagram are removed accordingly.

This ADR governs the *deployment/telemetry* of demos only. HF Hub remains the
store for model artifacts (ADR 006) and datasets (ADR 002); the Gradio UI
itself is unchanged in spirit.

## Rationale

Keeping Gradio preserves the fast visual sanity check. Running it locally
removes all standing infrastructure: no Space to deploy, no secrets, no
Supabase schema, no per-model privacy surface. The cost of an interactive demo
drops to "a script a contributor runs", which is uniform and cheap across
models. A maintainer can still expose a temporary public link ad hoc
(`demo.launch(share=True)`) without it being a standing obligation.

## Consequences

### Pros

- Interactive Gradio UX is retained for every model.
- No per-model Space deploy, hosting, secrets, or Supabase infrastructure.
- Demo runs offline/locally and behaves identically for every contributor.
- One consistent expectation for every future model contribution.

### Cons

- No always-on public playground. A maintainer can spin up a temporary share
  link from the local demo if ever needed; it is not a standing obligation.
- Contributors must include a local Gradio demo as part of the model's PR.
- `gradio` is pulled into each `training-<model>` extra (acceptable: it is only
  installed when explicitly opting into that model's tooling).

## Alternatives

- **Keep the HF Space and automate deployment via CI/CD.** Rejected:
  per-model Spaces plus CI/CD plumbing is unbounded maintenance for marginal
  value, and grows with every new model.
- **One shared multi-model HF Space.** Rejected: still hosted infrastructure
  and it couples otherwise independent model releases.
- **Drop the UI entirely for a CLI script.** Rejected: the interactive Gradio
  UI is the fastest way to inspect detection quality and is cheap to keep when
  run locally.
