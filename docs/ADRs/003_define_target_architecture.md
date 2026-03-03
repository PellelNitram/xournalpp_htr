# ADR 003 – Define Target Architecture

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
- See my paper notes!
- Unclear to me: i can design the inference side probably well. how to define the training side? should i at all??

compute_predictions takes path and pipeline string (or workflow??), ground truth shall be on word level and not sentence level bc thats a post processing step, first design the inference part and later look at model training approach and how to allow different installation modes, the compute fit returns bounding boxes with a string attached, inference for device should be same as for HF demo ideally, add tests, eval script to give pipeline and eval set to, evaluator allows multiple eval metrics to report, challenge the new architecture document, 

*2026-03-01* add version to this current model as well as code version to do it properly from here on.

*2026-03-01* check HF demo image in CI/CD

0302: will the above architecture also allow splitting installation dependencies into prod, dev, test? Or is tjis w pyinstaller irrelevant , how to store eval set? Have different commands or rather one w different argparsers?, how to solve training a model? For that, check if HF is indeed good enough. If I work on it, I need content pipeline


## Next actionable steps

1. Add a version to the current status to then work on future versions. We will use regular versioning (NOT date-based).

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

## Brainstorm: Additional improvement ideas

### Packaging & Tooling

- Migrate from `requirements.txt` + `setup.py` to a single `pyproject.toml` with `uv` as the package manager — modern standard, simplifies install, dependency resolution, and entry point declaration.
- Define CLI entry points in `pyproject.toml` (e.g. `xournalpp-htr = "xournalpp_htr.run_htr:main"`) so users get a proper command after `pip install`.
- Pin `ruff` version in `pyproject.toml` (currently only pinned in `.pre-commit-config.yaml`).
- Add `ty` (or `mypy`/`pyright`) to pre-commit hooks, not just as an aspiration.

### CI/CD

- Replace the fragile Dropbox model download in CI with GitHub Releases or HF Hub artifacts — Dropbox links rot and are slow.
- Expand CI test scope beyond just `installation` markers — run at least `correctness` tests on PRs to catch regressions.
- Add a CI job for type checking once `ty` is adopted.
- Consider caching model downloads in CI (GitHub Actions cache) to speed up runs.

### Architecture & Code Quality

- Remove legacy files: `xournalpp_htr/demo_concept_1.py` and `scripts/demo_concept_1.sh` appear obsolete and add confusion.
- Introduce a `Pipeline` class or builder pattern to replace the procedural `export_xournalpp_to_pdf_with_htr()` — makes the pipeline composable and testable (e.g., swap out the model, skip PDF export, etc.).
- Define a `Config` dataclass (or Pydantic `BaseSettings`) that consolidates all configuration: DPI values, model name, detector parameters (scale, margin), debug flags. Pass it through the pipeline instead of scattering hardcoded values.
- Make the coordinate conversion (150 DPI image → 72 DPI PDF) an explicit, tested function rather than inline math in `xio.py`.

### Plugin

- Replace `os.execute()` (blocking) with an async approach or at least a progress indicator — currently the Xournal++ UI freezes during HTR.
- Generate `config.lua` from a template at runtime (e.g., using environment variables or a `.xournalpp-htr.toml` in the user's home directory) instead of mutating a tracked source file during install.
- Add user-facing error messages when Python or dependencies are missing (currently fails silently or with cryptic Lua errors).

### Testing

- Add a golden-file regression test: small fixture `.xopp` → run pipeline → compare output PDF text layer against a known-good snapshot.
- Unit-test the DPI coordinate conversion in isolation — most critical correctness path with currently zero coverage.
- Test edge cases: single-stroke pages, pages with only background images, very long text lines, non-ASCII characters.
- Add a test that the CLI `--help` works and covers all arguments (cheap smoke test).

### Developer Experience

- Add `Makefile` targets (or `uv` scripts) for common workflows: `make test`, `make lint`, `make demo`, `make install-plugin`.
- Document the local development setup in the developer guide (currently sparse).
- Consider a `devcontainer.json` for VS Code / Codespaces to standardize the dev environment.

### Bundled External Dependency

- The `external/htr_pipeline` is vendored directly. Decide whether to: (a) fork & maintain it as a proper dependency, (b) absorb it into `xournalpp_htr` (the `WordDetectorNN` training code suggests this direction), or (c) keep vendoring but pin to a specific commit/tag.

### Runtime & Performance

- Multi-page parallel inference — pages are independent and could be processed concurrently (e.g., `concurrent.futures.ProcessPoolExecutor`).
- Lazy model loading — currently models are loaded per invocation. A long-running service mode (or caching the loaded model) would help repeated use.

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