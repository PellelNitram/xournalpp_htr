# ADR 006 – Model Registry and Training Environment

- Date: 2026-05-09
- Status: Accepted
- PRD: None
- Drivers: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))
- Deciders: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))

## Context

[ADR 002](002_use_HuggingFace_ecosystem_for_ML.md) adopted the [HuggingFace](https://huggingface.co/)
ecosystem and noted two possible strategies for hosting custom model architectures on the
[HF Hub](https://huggingface.co/docs/hub/index): (a) subclassing
[`PreTrainedModel`](https://huggingface.co/docs/transformers/main_classes/model) for full
[`transformers`](https://huggingface.co/docs/transformers/index) integration, or (b) using the Hub as
plain artifact storage. [ADR 003](003_define_target_architecture.md) established
`compute_predictions(document, pipeline)` as the central inference API, with models downloaded from HF
Hub on first use.

HF Hub is already used in this project for dataset distribution:
[`snapshot_download`](https://huggingface.co/docs/huggingface_hub/main/en/guides/download#download-an-entire-repository)
in [`xio.py`](../../xournalpp_htr/xio.py) fetches the
[IAM-OnDB training dataset](https://huggingface.co/datasets/PellelNitram/xournalpp_htr_IAM_OnDB), the
[benchmark dataset](https://huggingface.co/datasets/PellelNitram/xournalpp_htr_benchmark), and the
[examples dataset](https://huggingface.co/datasets/PellelNitram/xournalpp_htr_examples). This ADR
extends that existing infrastructure to cover trained model artifacts.

The project is now at the point of training its first custom model:
[Carbune](https://github.com/PellelNitram/OnlineHTR), a bidirectional LSTM stack with CTC loss. Carbune
operates on **online ink strokes** (x/y/t time series from a stylus), a non-standard input domain that
the `transformers` library has no built-in support for. It is implemented as a
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) `LightningModule`
([`LitModule1`](https://github.com/PellelNitram/OnlineHTR/blob/main/src/models/carbune_module.py)) and
uses a custom `AlphabetMapper` tokeniser and a greedy CTC decoder — none of which map onto standard
`transformers` abstractions. This is the root reason why off-the-shelf `PreTrainedModel` integration is
impractical. This ADR decides:

1. How custom models are stored on and retrieved from HF Hub.
2. What format trained models are exported to for inference.
3. How training code and its dependencies are organised within the repository.
4. How HF Hub model repositories are named.

Constraints:
- The training framework is not fixed: different models may use Lightning,
  [HF Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), or raw
  [PyTorch](https://pytorch.org/).
- The inference artifact must work independently of the training framework used.
- The inference environment must be lean ([PyInstaller](https://pyinstaller.org/) compatibility is a
  future goal; see issues [#66](https://github.com/PellelNitram/xournalpp_htr/issues/66),
  [#67](https://github.com/PellelNitram/xournalpp_htr/issues/67)).
- A pipeline may use multiple models ([ADR 003](003_define_target_architecture.md)); naming must reflect
  individual models, not pipelines.

## Decision

### 1. HF Hub as plain artifact storage

Custom models are stored on HF Hub as raw files — not by subclassing `PreTrainedModel` or using
[`PyTorchModelHubMixin`](https://huggingface.co/docs/huggingface_hub/main/en/guides/integrations#a-concrete-example-pytorch).
Each model repository contains the inference artifact(s) and whatever supporting files the model
requires to run (e.g. `config.json`, alphabet, tokeniser). The model builder decides what to upload
alongside the primary artifact; there is no enforced schema for supporting files. Files are downloaded
at inference time via
[`hf_hub_download`](https://huggingface.co/docs/huggingface_hub/main/en/guides/download#download-a-single-file).

A typical post-training upload looks like:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(path_or_fileobj="exports/model.onnx", path_in_repo="model.onnx",
                repo_id="PellelNitram/xournalpp-htr-carbune")
api.upload_file(path_or_fileobj="exports/config.json", path_in_repo="config.json",
                repo_id="PellelNitram/xournalpp-htr-carbune")
```

A typical inference load looks like:

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

onnx_path = hf_hub_download(repo_id="PellelNitram/xournalpp-htr-carbune", filename="model.onnx")
session = ort.InferenceSession(onnx_path)
```

Subsequent calls to `hf_hub_download` with the same arguments hit the local cache
(`~/.cache/huggingface/hub/`) and do not re-download unless the file changed on the Hub.

### 2. ONNX as the inference export format

After training, models are exported to [ONNX](https://onnx.ai/) and this export is the canonical
inference artifact. The training checkpoint (Lightning `.ckpt`, HF Trainer output, etc.) may also be
uploaded to the same HF Hub repository for resuming training, but inference always uses the ONNX export.

Rationale: ONNX is training-framework-agnostic, works with
[`onnxruntime`](https://onnxruntime.ai/) (a lean dependency), and is more amenable to PyInstaller
bundling than full PyTorch.

### 3. Per-model training extras and subfolders

Training code lives under [`xournalpp_htr/training/<model-name>/`](../../xournalpp_htr/training/) and
its dependencies are declared as a named
[optional extra](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#extras)
in [`pyproject.toml`](../../pyproject.toml):

```toml
[project.optional-dependencies]
training-carbune = ["lightning", "hydra-core", ...]
training-<next-model> = ["transformers", "datasets", ...]
training = [
    "xournalpp_htr[training-carbune]",
    "xournalpp_htr[training-<next-model>]",
]
```

Install options:
- `uv add xournalpp_htr` — inference only (lean)
- `uv add xournalpp_htr[training-carbune]` — inference + Carbune training
- `uv add xournalpp_htr[training]` — everything

Shared training utilities (CTC decoder, evaluation metrics, dataset loaders used across models) live in
`xournalpp_htr/training/shared/` with no extra dependencies beyond the base package.

Each training subfolder's `__init__.py` guards against missing dependencies:

```python
try:
    import lightning
except ImportError as e:
    raise ImportError(
        "Carbune training requires additional dependencies. "
        "Install with: uv add xournalpp_htr[training-carbune]"
    ) from e
```

### 4. HF Hub repository naming

Individual model repositories follow the convention:

```
PellelNitram/xournalpp-htr-<model-name>
```

Examples: `PellelNitram/xournalpp-htr-carbune`, `PellelNitram/xournalpp-htr-word-detector`.

The model card documents the dataset the model was trained on. A pipeline (as defined in
[ADR 003](003_define_target_architecture.md)) may reference one or more model repositories;
pipeline-to-model mapping is done inside each pipeline.

## Rationale

**Plain artifact storage over `PreTrainedModel`**: subclassing `PreTrainedModel` would require a
significant rewrite of the Carbune architecture and a fixed training framework, while providing little
benefit for a model with a non-standard input domain (online ink strokes). The plain storage approach
unblocks model sharing immediately with no model code changes.

**ONNX over Lightning checkpoints at inference**: `load_from_checkpoint` binds inference to Lightning and
to the `LitModule1` class definition. ONNX removes that binding entirely: any training framework can
produce the export, and any runtime that supports `onnxruntime` can consume it. This is also the path of
least resistance for future PyInstaller packaging.

**Per-model extras over a single `[training]` extra**: different models need incompatible frameworks.
A union extra would bloat every training environment. Named per-model extras keep environments minimal and
make dependency intent explicit.

**`PyTorchModelHubMixin` deferred**: this would give `from_pretrained` / `push_to_hub` on custom
`nn.Module`s without a full `PreTrainedModel` rewrite, and is the preferred long-term path — the goal is
a uniform `from_pretrained` interface across both off-the-shelf `transformers` models and custom
architectures. The blocker is concrete: the Carbune network (`Carbune2020NetAttempt1`) is referenced in
the Hydra config but has not been extracted into its own class — the LSTM layers and linear head live
directly inside `LitModule1`. Until the network is separated from the Lightning training wrapper, the
mixin cannot be applied. This upgrade should be revisited once the model architecture stabilises.

## Consequences

### Pros

- Training framework is fully flexible — Lightning, HF Trainer, or raw PyTorch are all valid.
- Inference has minimal dependencies: `onnxruntime` + `huggingface_hub`.
- Model sharing is unblocked immediately without any model code changes.
- Per-model extras keep training environments lean and explicit.
- ONNX export is compatible with the PyInstaller packaging path (issues
  [#66](https://github.com/PellelNitram/xournalpp_htr/issues/66),
  [#67](https://github.com/PellelNitram/xournalpp_htr/issues/67)).
- Naming convention is simple and consistent across all models.

### Cons

- ONNX export must be written and validated for each model. ONNX export traces the model with example
  inputs and freezes the operations into a static graph, so Python-level control flow whose path depends
  on tensor values at runtime is captured as whichever branch the trace happened to take. For Carbune
  this affects mostly the CTC decoder (greedy/beam-search loops over output probabilities), so the
  decoder will likely run in Python outside the ONNX graph rather than being exported. Variable-length
  sequence handling also requires explicit dynamic axes declarations. Workarounds when control flow is
  needed inside the graph: use [`torch.jit.script`](https://pytorch.org/docs/stable/generated/torch.jit.script.html)
  before export to preserve control flow, use the newer
  [`torch.onnx.export(..., dynamo=True)`](https://pytorch.org/docs/stable/onnx_dynamo.html) path, or
  restructure the model so that dynamic logic lives outside the exported portion.
- Consumers cannot use `from_pretrained`; they must know to call `hf_hub_download` and load the ONNX
  manually.
- Supporting files (alphabet, config) alongside the ONNX are model-specific with no enforced schema —
  the model builder is responsible for documenting what is required.
- The `[training]` umbrella extra installs all training frameworks and will grow heavy over time.

## Open Questions

- **Issue [#64](https://github.com/PellelNitram/xournalpp_htr/issues/64)** — Pipeline configuration: how
  does a pipeline name map to the one or more HF Hub model repos it requires? This is unresolved and
  will be addressed in a future ADR or issue.
- **Issue [#73](https://github.com/PellelNitram/xournalpp_htr/issues/73)** — Training delivery:
  [Docker](https://www.docker.com/) containers are a reasonable fit for cloud and CI training (each
  model's `Dockerfile` runs `uv add xournalpp_htr[training-<model-name>]`), but add friction for local
  experimentation. Decision deferred.

## Related Decisions

- **Issue [#71](https://github.com/PellelNitram/xournalpp_htr/issues/71)** (eval dataset storage) is
  largely resolved by existing infrastructure: the benchmark dataset already lives on HF Hub
  ([`PellelNitram/xournalpp_htr_benchmark`](https://huggingface.co/datasets/PellelNitram/xournalpp_htr_benchmark))
  and is consumed via `snapshot_download` in [`xio.py`](../../xournalpp_htr/xio.py). The existing
  dataset will be extended rather than replaced; no new eval dataset format decision is needed.
- **Issues [#66](https://github.com/PellelNitram/xournalpp_htr/issues/66),
  [#67](https://github.com/PellelNitram/xournalpp_htr/issues/67)** (installation modes, PyInstaller):
  the per-model extras and ONNX inference format directly support the planned installation mode split.
  The lean base install (`uv add xournalpp_htr`) corresponds to the end-user installation mode.
- **Issues [#62](https://github.com/PellelNitram/xournalpp_htr/issues/62),
  [#69](https://github.com/PellelNitram/xournalpp_htr/issues/69)** (CLI shape, HTR entry point): the
  inference loading pattern decided here (`hf_hub_download` + `onnxruntime`) is what the future CLI
  entry point will use internally.

## Alternatives

- **Subclass `PreTrainedModel`**: full transformers integration, inference via standard `from_pretrained`.
  Rejected: requires rewriting the model interface, fixes the training framework, and provides little
  benefit for a non-standard input domain.
- **`PyTorchModelHubMixin`**: adds `from_pretrained` / `push_to_hub` to any `nn.Module` without
  `transformers`. Deferred: requires extracting the network from the Lightning wrapper first.
- **TorchScript instead of ONNX**: also framework-agnostic but harder to bundle with PyInstaller and
  requires full PyTorch at inference time. Rejected in favour of ONNX.
- **Single `[training]` extra**: simpler but bloated when multiple incompatible training frameworks
  coexist. Rejected in favour of per-model named extras.
