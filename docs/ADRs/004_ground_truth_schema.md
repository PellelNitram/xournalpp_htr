# ADR 004 – Ground Truth Schema for HTR Evaluation

- Date: 2026-04-11
- Status: Accepted
- PRD: None
- Drivers: PellelNitram ([GitHub](https://github.com/PellelNitram))
- Deciders: PellelNitram ([GitHub](https://github.com/PellelNitram))

## Context

To evaluate HTR model predictions, a ground truth dataset is needed. Before building
annotation tooling and evaluation code, a schema must be defined so that all components
agree on what ground truth data looks like.

The project currently uses an image-based ML model (via `htr_pipeline`) but the long-term
goal is to operate natively on strokes (as stored in Xournal++ `.xopp`/`.xoj` files). The
schema should serve both present and future needs without requiring re-annotation.

## Decision

Ground truth is stored as one `.gt.json` file per source document, named
`<document-stem>.gt.json` (e.g. `2024-07-26_minimal.xopp` →
`2024-07-26_minimal.gt.json`). The format is defined by
`docs/schemas/ground_truth.schema.json`.

Key design choices:

- **Stroke references, not coordinates.** Each annotation identifies a group of strokes
  by `page_index`, `layer_index`, and `stroke_indices` within the source document. No
  pixel coordinates or bounding boxes are stored, making the schema independent of
  rendering resolution and naturally aligned with future stroke-based models.

- **Reference to source document, not embedded data.** The file stores a `filename` and
  `sha256` hash of the source document rather than duplicating its contents. The hash
  detects silent drift if the source file is modified after annotation.

- **Closed annotation class vocabulary.** Annotations are assigned one of a fixed set of
  classes: `word`, `digit`, `mathematical_expression`, `arrow`, `diagram`, `table`,
  `drawing`, `separator`, `correction`, `other`. Extending the vocabulary requires a
  schema version bump, which makes class changes explicit and traceable.

- **`text` conditionally required.** The `text` transcription field is required for
  `word`, `digit`, and `mathematical_expression`, and forbidden for all other classes.
  This is enforced via JSON Schema `if/then/else`.

- **Annotator and timestamp metadata.** `annotator_id` and `created_at` are required
  top-level fields to support inter-annotator agreement analysis and dataset versioning.

- **Annotation tool enforces completeness.** The tool must classify every stroke before
  saving. Therefore the schema has no partial-annotation marker — every saved file is
  complete ground truth.

## Rationale

Storing stroke references rather than pixel coordinates future-proofs the annotations:
the current image-based model is a temporary detour, and re-annotating an entire dataset
to change coordinate systems would be expensive. Stroke indices are cheap to resolve at
evaluation time.

A closed class vocabulary prevents annotator inconsistency (e.g. `"line"` vs `"Line"` vs
`"horizontal_line"`). JSON Schema versioning (`schema_version: "1.0.0"`) makes class
additions explicit.

Stroke references also serve pixel-based models. To evaluate a bounding-box model against
this ground truth, a conversion utility loads the referenced strokes and computes
`(xmin, ymin, xmax, ymax)` at the target DPI. Stroke references are strictly more
information than a stored bounding box — the conversion is one-directional (strokes →
boxes, never boxes → strokes), so storing strokes future-proofs the ground truth for both
model types.

JSON Schema was chosen as the schema language because it is language-agnostic, has
validators in both Python (`jsonschema`) and JavaScript (`ajv`), and the data is already
JSON. Pydantic can generate a JSON Schema from a `BaseModel` if a typed Python
representation is needed later.

## Consequences

### Pros

- Annotations are independent of rendering DPI — usable for both current image-based
  models (after coordinate conversion) and future stroke-based models.
- Source document hash makes dataset integrity verifiable.
- Closed class vocabulary keeps annotations consistent across annotators.
- Single file per document keeps the dataset easy to manage.

### Cons

- Stroke indices are positional and fragile: if the source `.xopp` file is edited after
  annotation (strokes added or removed), indices may silently shift. The SHA-256 hash is
  the only guard — tooling must refuse to load annotations when the hash does not match.
- Duplicate stroke references across annotations (same stroke assigned to two words)
  cannot be caught by JSON Schema and require a separate code-level validator.

## Tooling Requirements

The schema alone is not sufficient to guarantee a valid dataset. Annotation tools must
additionally enforce the following:

1. **Schema validation.** Every saved `.gt.json` file must be validated against
   `ground_truth.schema.json` before writing.

2. **No duplicate stroke references.** JSON Schema cannot enforce that the same stroke is
   not assigned to two annotations (same `page_index`, `layer_index`, `stroke_index`).
   The annotation tool must reject any attempt to assign an already-annotated stroke to a
   second annotation.

3. **Completeness enforcement.** The tool must ensure every stroke in the source document
   is covered by an annotation before saving. Completeness can be verified by cross-
   referencing the total stroke count from the `.xopp`/`.xoj` file against the union of
   all `stroke_indices` in the `.gt.json` file.

4. **Hash mismatch guard.** The tool must refuse to load a `.gt.json` file if the SHA-256
   hash of the currently loaded source document does not match `source_document.sha256`.

## Annotation Granularity

Ground truth is annotated at **word level**, not character level.

Word-level is sufficient for the current evaluation goals: word detection (did the model
find the right strokes?) and transcription accuracy (CER/WER computed over words). It is
also the right starting point because character-level annotation is significantly more
expensive to produce, and stroke-to-character mapping is inherently ambiguous in
handwriting — a single stroke can span multiple characters (e.g. a crossing stroke in `t`
or `f`), and some characters require multiple strokes.

Character-level annotation would additionally enable training and evaluating character
segmentation models and fine-grained per-character error analysis. If that becomes a
requirement, it can be added as an optional field in a future schema version — but the
annotation cost should only be paid when a character-level model exists to benefit from
it.

## Annotation Conventions

The following conventions must be followed consistently across annotators:

- **`digit` granularity.** A sequence of digits written as a single connected unit (e.g.
  `123` with no visible gap between digits) is one `digit` annotation with `text: "123"`.
  Digits that are spatially separated (e.g. `1    2    3`) are each a separate `digit`
  annotation. The determining factor is whether the digits form one visually grouped unit
  or multiple distinct ones.

## Alternatives

- **Store pixel bounding boxes** (as in the existing `v1_2024-10-13` annotation schema):
  simpler for image-based evaluation but ties ground truth to a specific rendering DPI
  and is useless for stroke-based models.

- **Embed stroke data** (x/y arrays) instead of referencing by index: makes files
  self-contained but duplicates data already present in the source document and bloats
  the dataset.

- **Open class vocabulary**: maximum annotator freedom but leads to inconsistent labels
  and requires post-hoc normalisation before evaluation.

- **Protocol Buffers**: stronger typing and cross-language code generation, but requires
  a build step and is heavier than necessary for small annotation files.
