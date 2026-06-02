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

### 2026-06-02 — Experiment 1: learning rate and batch size sweep

- **Hypothesis:** Find the best learning rate and batch size combination
  for the default SimpleHTR architecture on IAM words.
- **Setup:** IAM word-level dataset, 95/5 train/val split, no augmentation,
  early stopping with patience 15, max 100 epochs. Grid: LR ∈ {0.0005,
  0.001, 0.002} × BS ∈ {32, 64, 128}. NVIDIA A100 40GB.
- **Command:** `bash xournalpp_htr/training/simple_htr/run_training.sh`
- **Code revision:** `a72b64c`
- **Results:**

| LR | BS | Best Epoch | CER | Word Acc | Path |
|---|---|---|---|---|---|
| 0.001 | 64 | 64 | **0.072** | **78.9%** | `experiments/experiment1/lr0.001_bs64/` |
| 0.0005 | 32 | 15 | 0.075 | 78.1% | `experiments/experiment1/lr0.0005_bs32/` |
| 0.0005 | 64 | 28 | 0.076 | 77.1% | `experiments/experiment1/lr0.0005_bs64/` |
| 0.001 | 32 | 37 | 0.078 | 77.4% | `experiments/experiment1/lr0.001_bs32/` |
| 0.001 | 128 | 29 | 0.078 | 77.7% | `experiments/experiment1/lr0.001_bs128/` |
| 0.0005 | 128 | 20 | 0.078 | 77.2% | `experiments/experiment1/lr0.0005_bs128/` |
| 0.002 | 128 | 41 | 0.078 | 77.1% | `experiments/experiment1/lr0.002_bs128/` |
| 0.002 | 64 | 32 | 0.080 | 76.4% | `experiments/experiment1/lr0.002_bs64/` |
| 0.002 | 32 | 44 | 0.081 | 76.3% | `experiments/experiment1/lr0.002_bs32/` |

- **Conclusion:** LR=0.001 with BS=64 achieves the best CER (0.072) and
  word accuracy (78.9%), matching the original SimpleHTR's reported ~75%.
  Performance is fairly stable across configurations (CER 0.072–0.081).
  Lower learning rates (0.0005) converge in fewer epochs but reach similar
  accuracy; higher LR (0.002) slightly underperforms. Recommended defaults:
  LR=0.001, BS=64.

## Current status

- [x] Network architecture (CNN + BiLSTM + CTC)
- [x] IAM word-level dataset loader with caching
- [x] Training loop with CER/word accuracy validation
- [x] ONNX export and HF Hub upload
- [x] Inference model (`SimpleHTRModel`)
- [x] Local Gradio demo
- [x] Hyperparameter sweep script
- [x] First training run and experiment log
- [ ] ONNX validation notebook

## Outlook

- Add augmentation experiment
- Add beam search decoding
- Integrate into full pipeline (word detection + recognition)
