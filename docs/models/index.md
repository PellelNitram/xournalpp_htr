# Models

Documentation for each model trained and shipped as part of Xournal++ HTR.
Each page covers setup, training, evaluation, export and inference for that
model.

- [WordDetector](word_detector.md) — word-level bounding-box detector
  (reimplementation of
  [WordDetectorNN](https://github.com/githubharald/WordDetectorNN)).
- [SimpleHTR](simple_htr.md) — word-level text recognition
  (reimplementation of
  [SimpleHTR](https://github.com/githubharald/SimpleHTR)).

New models added in the future should follow the same structure: a page under
`docs/models/<model_name>.md` linked from this index and from the `Models`
section of the navigation. The source lives in
`xournalpp_htr/training/<model_name>/` with a lean `README.md` that refers
back to its docs page.

## Conventions for new models

These conventions apply to every model under `xournalpp_htr/training/<model_name>/`
and its docs page. They exist so that training, export, inference and
documentation stay uniform across models. See the
[WordDetector page](word_detector.md) as a reference example.

### Source layout

- Per-model code lives in `xournalpp_htr/training/<model_name>/`.
- Reusable code lives in `xournalpp_htr/training/shared/` (base deps only,
  so they remain importable in the lean inference install).
- The HF-Hub-backed inference class lives in `xournalpp_htr/inference_models.py`
  and is exposed as `XxxModel.from_pretrained()`, returning typed outputs. No
  `transformers` dependency — inference is uniform across all custom models
  (see ADR 003 for the pipeline contract: word-level boxes *and*
  transcriptions).
- Training deps are gated behind a per-model `uv` extra named
  `training-<model-name>`. The base install stays lean (ADR 006).
- Demos are local Gradio apps invoked via `python -m
  xournalpp_htr.training.<model_name>.demo`, not HF Spaces (ADR 007).

### Training, export and publish workflow

- Configure via Hydra.
- Call the actual training script in a `run_training.sh` bash script.
- Store experiments under `experiment/<experiment>/<run>` inside
  the model folder.
- In each run's output folder, store TensorBoard logs, the trained model
  and its config.
- I want to inspect the best model visually with the Gradio `demo.py`.
- I want to export the model to ONNX and `config.json` via `export.py`
  to be able to load it in the inference code after uploading to HF-Hub.
  The export script should take `--checkpoint`, `--output-dir` and
  `--export` flags.
- Validate the ONNX export against PyTorch checkpoint with a Jupyter notebook.
  I am reviewing the notebook manually.
- Upload the ONNX model via `export.py --upload` after validation passes.

### HuggingFace artifact naming

- Model repo: `PellelNitram/xournalpp-htr-<model-name>`.
- Dataset repo: `PellelNitram/xournalpp_htr_<dataset>`.

### Required docs sections

Each `docs/models/<model_name>.md` page should contain, in order:

1. Short intro (what the model does, source link, ADR refs).
2. **GPU training setup** — step-by-step from clone → install → dataset →
   train → eval → demo → export → validate → upload.
3. **Inference** — minimal `from_pretrained()` usage example.
4. **Experiments** — log of training experiments, newest first, using the
   template below.
5. **Current status** — what is implemented, known issues.
6. **Outlook** — planned follow-ups.

### Experiments log template

```
### <date> — <short title>

- **Hypothesis:** what question this experiment is meant to answer.
- **Setup:** dataset split, config overrides, code revision (commit hash).
- **Command:** the exact training/eval command used.
- **Results:** key metrics, path to artefacts under `experiments/`,
  TensorBoard run name.
- **Conclusion:** what was learned and what to try next.
```
