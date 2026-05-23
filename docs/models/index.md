# Models

Documentation for each model trained and shipped as part of Xournal++ HTR.
Each page covers setup, training, evaluation, export and inference for that
model.

- [WordDetector](word_detector.md) — word-level bounding-box detector
  (reimplementation of
  [WordDetectorNN](https://github.com/githubharald/WordDetectorNN)).

New models added in the future should follow the same structure: a page under
`docs/models/<model_name>.md` linked from this index and from the `Models`
section of the navigation. The source lives in
`xournalpp_htr/training/<model_name>/` with a lean `README.md` that refers
back to its docs page.

## Conventions for new models

These conventions apply to every model under `xournalpp_htr/training/<model_name>/`
and its docs page. They exist so that training, export, inference and
documentation stay uniform across models.

### Source layout

- Per-model code lives in `xournalpp_htr/training/<model_name>/`.
- Generic geometry, decoders, clustering and metrics live in
  `xournalpp_htr/training/shared/` (base deps only, so they remain
  importable in the lean inference install).
- The HF-Hub-backed inference class lives in
  `xournalpp_htr/inference_models.py` and is exposed as
  `XxxModel.from_pretrained()`, returning typed outputs. No
  `transformers` dependency — inference is uniform across all custom models
  (see ADR 003 for the pipeline contract: word-level boxes *and*
  transcriptions).
- Training deps are gated behind a per-model `uv` extra named
  `training-<model-name>`. The base install stays lean (ADR 006).
- Demos are local Gradio apps invoked via `python -m
  xournalpp_htr.training.<model_name>.demo`, not HF Spaces (ADR 007).

### Expected modules

Each model directory should contain (names may vary slightly, but the roles
should map 1:1):

| File | Purpose |
| --- | --- |
| `config.py` | Hydra structured config — single source of truth for all constants |
| `network.py` | Architecture + training loss |
| `dataset.py` | Dataset loading + ground-truth encoding |
| `train.py` | Training entrypoint (Hydra CLI) |
| `export.py` | ONNX + `config.json` export, with `--upload` for HF Hub |
| `infer.py` | Local torch inference from a `.pth` checkpoint |
| `demo.py` | Local Gradio demo (ADR 007) |
| `utils.py` | Git-hash, JSON encoder, example-image list |
| `test_best_model.ipynb` | Offline ONNX-vs-PyTorch parity check |
| `run_training.sh` | Hyperparameter sweep (grid search) |
| `run_training.eval.sh` | Find the best model from a sweep |

### Training, export and publish workflow

1. Configure via Hydra (`config.py` is authoritative; CLI overrides only).
2. Run a sweep with `run_training.sh`; evaluate with `run_training.eval.sh`.
3. Sweep results land under `experiments/<experiment>/<run>/` with
   `best_model.pth`, `best_model.json` (best val metric + epoch),
   TensorBoard logs in `summary_writer/`, and `config.yaml`.
4. Inspect the best model visually with the Gradio `demo.py`.
5. Export to ONNX + `config.json` via `export.py`; validate against the
   PyTorch checkpoint with `test_best_model.ipynb`.
6. Upload via `export.py --upload`.

### HuggingFace artifact naming

- Model repo: `PellelNitram/xournalpp-htr-<model-name>`.
- Dataset repo: `PellelNitram/xournalpp_htr_<dataset>`.

### Required docs sections

Each `docs/models/<model_name>.md` page should contain, in order:

1. Short intro (what the model does, source link, ADR refs).
2. **Structure** — the per-module table above, with the `uv` extra each
   module needs.
3. **GPU training setup** — step-by-step from clone → install → dataset →
   train → eval → demo → export → validate → upload.
4. **Inference** — minimal `from_pretrained()` usage example.
5. **Experiments** — log of training experiments, newest first, using the
   template below.
6. **Current status** — what is implemented, known issues.
7. **Outlook** — planned follow-ups.

### Experiments log template

```
### <date> — <short title>

- **Goal:** what question this experiment is meant to answer.
- **Setup:** dataset split, config overrides, code revision (commit hash).
- **Command:** the exact training/eval command used.
- **Results:** key metrics, path to artefacts under `experiments/`,
  TensorBoard run name.
- **Conclusion:** what was learned and what to try next.
```
