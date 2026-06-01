# SimpleHTR

Handwritten word recognition model — reimplementation of
[Harald Scheidl's SimpleHTR](https://github.com/githubharald/SimpleHTR).
Architecture: 5 CNN layers + 2 bidirectional LSTM layers + CTC output.
Takes a single word image and predicts the text.

Source: `xournalpp_htr/training/simple_htr/`

## File structure

| File | Purpose | Deps |
|------|---------|------|
| `config.py` | Hydra structured config | base |
| `network.py` | `SimpleHTRNet` (CNN+BiLSTM+CTC) | training |
| `dataset.py` | IAM word-level dataset loader | training |
| `train.py` | Training entrypoint (`@hydra.main`) | training |
| `infer.py` | Local torch inference from `.pth` | training |
| `demo.py` | Local Gradio demo (ADR 007) | training |
| `export.py` | ONNX export + HF Hub upload | training |
| `utils.py` | Device, git hash, JSON encoder | training |
| `run_training.sh` | Hyperparameter sweep script | shell |

## GPU training setup

### 1. Clone and install base

```bash
git clone https://github.com/PellelNitram/xournalpp_htr.git
cd xournalpp_htr
bash INSTALL_LINUX.sh
```

### 2. Install training extra with CUDA PyTorch

```bash
uv sync --extra training-simple-htr
```

### 3. Verify GPU

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Authenticate with HuggingFace

```bash
uv run huggingface-cli login
```

### 5. Download dataset

The IAM-DB dataset is downloaded automatically on first training run.
It includes pre-cropped word images (`data/words/`) and ground truth
(`data/ascii/words.txt`).

## Training

### Single run

```bash
uv run python -m xournalpp_htr.training.simple_htr.train \
    training.learning_rate=0.001 \
    training.batch_size=64 \
    training.epoch_max=100 \
    output_path=outputs
```

### Hyperparameter sweep

```bash
bash xournalpp_htr/training/simple_htr/run_training.sh
```

## Evaluation

Find the best model from a sweep:

```bash
find experiments/ -name "best_model.json" -exec sh -c \
    'echo "--- $1 ---"; cat "$1"' _ {} \;
```

## Inspection

### Gradio demo

```bash
uv run python -m xournalpp_htr.training.simple_htr.demo \
    --model-path <path>/best_model.pth --device auto
```

## Export

### ONNX export

```bash
uv run python -m xournalpp_htr.training.simple_htr.export \
    --checkpoint <path>/best_model.pth --output-dir exports/
```

### Validate ONNX

Use the `test_best_model.ipynb` notebook to compare PyTorch and ONNX outputs.

### Upload to HF Hub

```bash
uv run python -m xournalpp_htr.training.simple_htr.export \
    --checkpoint <path>/best_model.pth --output-dir exports/ --upload
```

## Inference

```python
from xournalpp_htr.inference_models import SimpleHTRModel

model = SimpleHTRModel.from_pretrained()
text = model.recognize(word_image_grayscale)
print(text)
```

## Experiments

_No experiments logged yet._

## Current status

- [x] Network architecture (CNN + BiLSTM + CTC)
- [x] IAM word-level dataset loader with caching
- [x] Training loop with CER/word accuracy validation
- [x] ONNX export and HF Hub upload
- [x] Inference model (`SimpleHTRModel`)
- [x] Local Gradio demo
- [x] Hyperparameter sweep script
- [ ] First training run and experiment log
- [ ] ONNX validation notebook

## Outlook

- Run experiment 1 (LR/BS sweep) and document results
- Add augmentation experiment
- Add beam search decoding
- Integrate into full pipeline (word detection + recognition)
