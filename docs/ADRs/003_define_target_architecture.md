# ADR 003 – Define Target Architecture

- Date: YYYY-MM-DD
- Status: Accepted or Ongoing or Superseeded by [ADR]()
- PRD: None
- Drivers: Name ([Link to Github handle](https://github.com/))
- Deciders: Name ([Link to Github handle](https://github.com/))

## TODO: Unsorted list of ideas while writing ADR

TODO: See my physical paper notes

this ADR sets direction but some detailed decisions will be fixed in future ADRs.

technical aspects:
- use uv
- ideally w/ data classes (or BaseModels) to define input and outputs at every point
- maybe use type checking using ty?

i want to use the same code in run_htr and demo. for that, the central function will be `compute_predictions` that will accept a `pathlib.Path` object to a document and a pipeline string. it will then return bounding boxes with words as a result.

this result can either be added to an exported PDF (using the CLI tool, potentially through the Xournal++ plugin, or the HuggingFace demo) or used to compute the performance of a pipeline.

for the latter, the performance, we need an evaluation dataset. i will annotate a few pages of Xournal++ documents by hand and then write code to benchmark future pipelines using this dataset (of course, future such datasets can be built as well). the evaluation pipeline will allow to implement a number of evaluation metrics to remain flexible with a few reasonable defaults to start with. for the eval dataset, we will draw bounding boxes around single words and note the corresponding words. In the future, we might also want to add annotations to strokes that belong to diagrams or sketches and also add line level annotations to predict full lines of text as opposed to single words; we don't annotate on the line level for now because grouping words into lines can be considered a post-processing step.

inference: when the `compute_predictions` function is called - through CLI or demo -, the models corresponding to the chosen pipeline are loaded. if they have not been used before, then they are downloaded from the internet, otherwise from the model cache. we are going to use HuggingFace for that [1]. relying on a central `compute_predictions` (which is a short cut to a set of function calls in a particular order) allows inference on device to be same as for HF demo.

training: TODO. ask claude code! TODO(how to solve training a model? For that, check if HF is indeed good enough.) TODO(Unclear to me: i can design the inference side probably well. how to define the training side? should i at all?? how about just providing a docker container to use w/ full installation? this could be a cool feat to test integration simulatenously)

evaluation: a command to run a pipeline against an eval dataset so that everyone can run the benchmarking themselves if they want. the eval pipeline (called through an eval script) will be given a pipeline name and an eval dataset name. the evaluator called in the eval pipeline will allow multiple eval metrics to report. TODO(question: how to best store eval dataset?)

code splitting. TODO. write it down here and then ask CC. installation modes! to allow efficient installation modes, we want to allow multiple installation methods - one for normal users and one for developers and, potentially, one for model developers. TODO(question: is this relevant for pyinstaller delivery?)

code quality: we want to add tests for each feature (unit tests using `pytest`) as well as integration tests; i want to expand the amount of tests to professionalise the project and allow AI coding to be more efficient.. for integration tests, we have to do some research what to use ideally; potentially `testcontainers`? how about adding some telemetry and a user feedback mechanism (the latter could be a simple link to github issues); the telemetry could be added to the config. speaking of config, how to do that :-D? a setup could allow users to create a config and ask for telemetry?

plugin: set up entry point script to install the plugin

questions to solve before finishing this ADR:
- shall i call it "pipeline" or "workflow"?
- [1] can i really use huggingface as model registry and store given the diversity of models i will build and given that the architectures will not be existing ones but new ones (like Carbune).
- challenge the new architecture document
- Have different commands or rather one w different argparsers?
- set up entry point script to run htr?

## Next actionable steps

1. Add a version to the current status to then work on future versions. We will use regular versioning (NOT date-based). Next to the version for the code, we also want to add a name for the current pipeline; note that future versions won't support this pipeline anymore because of the planned removal of the `htr_pipeline` dependency.
2. We need integration tests for HF demo image in CI/CD. We want to be able to run the same tests locally.

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