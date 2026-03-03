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