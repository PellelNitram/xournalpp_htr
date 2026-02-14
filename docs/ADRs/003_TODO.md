# ADR 002 – Use Hugging Face Ecosystem for Machine Learning

- Date: YYYY-MM-DD
- Status: Accepted or Ongoing or Superseeded by [ADR]()
- PRD: None
- Drivers: Name ([Link to Github handle](https://github.com/))
- Deciders: Name ([Link to Github handle](https://github.com/))

## TODO: Unsorted list of ideas while writing ADR

- use uv
- set up entry point script to install the plugin
- (maybe) set up entry point script to run htr
- TODO: See my physical notes
- ideally w/ data classes (or BaseModels) to define input and outputs at every point
- ideally w/ type checking using ty
- i want to use the same code in run_htr and demo.
- i want to expand the amount of tests to professionalise the project and allow AI coding to be more efficient.

## Improvement ideas to review

- **Deduplicate `XournalDocument` / `XournalppDocument`**: Both `load_data()` implementations in `documents.py` are identical. Merge into the base class or a shared helper.
- **Unify demo and CLI pipeline**: `scripts/demo.py` reimplements the pipeline steps instead of calling `export_xournalpp_to_pdf_with_htr()`. Changes to the pipeline flow need to be applied in two places.
- **Introduce `Prediction` dataclass**: Predictions are untyped `dict[int, list[dict]]` with string keys. A dataclass (like the existing `Stroke`/`Page`/`Layer` pattern) would add type safety and IDE autocompletion.
- **Extract DPI constants**: The `150` (image rendering DPI) and `72` (PDF DPI) values appear as magic numbers in `xio.py` and `models.py`. If one is changed without the other, text lands in the wrong place silently.
- **Replace `print()` with `logging`**: Makes verbosity controllable and debugging easier in the plugin context where stdout may not be visible.
- **Simplify subprocess check in `export_to_pdf_with_xournalpp()`**: The stderr check for `"PDF file successfully created"` is fragile if Xournal++ changes its output message. Return code + file existence should suffice.
- **Clean up temp files**: The pipeline creates temp files in `/tmp` but never removes them. A `finally` block or context manager would prevent accumulation.
- **Remove `setup.py` side effect on `config.lua`**: `setup.py` mutates a tracked source file (`plugin/config.lua`) at install time. Consider generating config at runtime or via environment variables instead.
- **Model registry for extensibility**: `compute_predictions()` uses a single `if model_name == ...` branch. A registry (dict mapping name to callable) would make adding models cleaner.
- **Expand test coverage**: The riskiest code (coordinate conversion, prediction embedding, empty page handling) lacks tests. A golden-file test with a small fixture `.xopp` and known expected predictions would catch regressions.

## Context

*(Add text here.)*

## Decision

*(Add text here.)*

## Rationale

*(Add text here.)*

## Consequences

### Pros

*(Add bulletpoints here).*

### Cons

*(Add bulletpoints here).*

## Alternatives

*(Add text here.)*