# ADR 005 – Prediction Bounding Box Coordinate System

- Date: 2026-04-30
- Status: Accepted
- PRD: None
- Drivers: PellelNitram ([GitHub](https://github.com/PellelNitram))
- Deciders: PellelNitram ([GitHub](https://github.com/PellelNitram))

## Context

Xournal++ documents store stroke coordinates in document units at 72 DPI (`document.DPI = 72`).
The current HTR pipeline (`2024-07-18_htr_pipeline`) renders pages to images at 150 DPI before
running word detection, so the raw bounding boxes returned by the underlying `htr_pipeline`
library are in 150 DPI image pixels — a different coordinate system from the document.

When implementing benchmark code to compare predictions against ground truth annotations
(which reference stroke coordinates in document units), this mismatch would require the
benchmark to know each pipeline's internal rendering DPI. That coupling would break as soon
as a future pipeline operates differently (e.g. a stroke-based model that never renders an
image at all).

## Decision

`compute_predictions()` must always return bounding box coordinates in **document units
(72 DPI)**, regardless of the internal rendering resolution used by the pipeline.

Each pipeline implementation is responsible for converting its raw output to document units
before returning. For image-based pipelines the conversion is:

```
doc_coord = image_pixel_coord * (document.DPI / render_dpi)
```

## Rationale

Ground truth stroke coordinates are natively in document units. Keeping predictions in the
same coordinate system means benchmark code and any other consumer of `compute_predictions`
never need to know how a pipeline works internally. This is consistent with ADR 003's
principle of typed I/O contracts with explicit semantics.

## Consequences

### Pros

- Benchmark code is decoupled from pipeline internals.
- Ground truth and prediction coordinates are directly comparable.
- Future stroke-based models can output document-unit coordinates natively with no conversion.

### Cons

- Pipeline implementations must remember to apply the conversion. A future pipeline that
  skips it will produce silently wrong coordinates. A coordinate-system field on the
  predictions dict could guard against this, but is deferred until a second pipeline exists
  to justify the complexity.

## Alternatives

- **Keep predictions in image pixels and convert in benchmark code.** Rejected because it
  requires benchmark code to know each pipeline's render DPI, coupling them.
- **Define a new canonical coordinate system (e.g. normalised 0–1).** More
  future-proof but unnecessary complexity given that document units are already the natural
  system for this project.
