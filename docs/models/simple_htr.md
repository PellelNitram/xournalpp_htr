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

### 2026-06-03 — Experiment 1: learning rate and batch size sweep

- **Hypothesis:** Find the best learning rate and batch size combination
  for the default SimpleHTR architecture on IAM words.
- **Setup:** IAM word-level dataset, 95/5 train/val split, no augmentation,
  early stopping with patience 25, max 100 epochs. Grid: LR ∈ {0.0005,
  0.001, 0.002} × BS ∈ {32, 64, 128}. NVIDIA A100 40GB.
- **Command:** `bash xournalpp_htr/training/simple_htr/run_training.sh`
- **Code revision:** `89971fe`
- **Results:**

| LR | BS | Best Epoch | CER | Word Acc | Path |
|---|---|---|---|---|---|
| 0.0005 | 64 | 54 | **0.070** | **79.2%** | `experiments/experiment1/lr0.0005_bs64/` |
| 0.0005 | 32 | 27 | 0.070 | 79.3% | `experiments/experiment1/lr0.0005_bs32/` |
| 0.0005 | 128 | 83 | 0.073 | 78.8% | `experiments/experiment1/lr0.0005_bs128/` |
| 0.001 | 64 | 33 | 0.074 | 78.8% | `experiments/experiment1/lr0.001_bs64/` |
| 0.001 | 128 | 41 | 0.074 | 77.8% | `experiments/experiment1/lr0.001_bs128/` |
| 0.002 | 128 | 56 | 0.075 | 77.8% | `experiments/experiment1/lr0.002_bs128/` |
| 0.001 | 32 | 59 | 0.076 | 77.3% | `experiments/experiment1/lr0.001_bs32/` |
| 0.002 | 32 | 20 | 0.086 | 74.9% | `experiments/experiment1/lr0.002_bs32/` |
| 0.002 | 64 | 29 | 0.089 | 75.1% | `experiments/experiment1/lr0.002_bs64/` |

- **Conclusion:** LR=0.0005 with BS=64 achieves the best CER (0.070) and
  word accuracy (79.2%). The top-3 configs all use LR=0.0005, showing that
  lower learning rates consistently outperform. LR=0.002 clearly
  underperforms. Recommended defaults: LR=0.0005, BS=64.

### 2026-06-05 — Experiment 2: augmentation

- **Hypothesis:** Data augmentation (Gaussian blur, geometric transforms,
  morphological ops, contrast/noise) improves generalisation.
- **Setup:** LR=0.0005 (best from exp1), BS ∈ {32, 64, 128} ×
  augmentation {on, off}. Max 200 epochs, patience 25. NVIDIA A100 40GB.
- **Command:** `bash xournalpp_htr/training/simple_htr/run_training.sh`
- **Code revision:** `f5f8c60`
- **Results:**

| Aug | BS | Best Epoch | CER | Word Acc | Path |
|---|---|---|---|---|---|
| true | 128 | 142 | **0.060** | **82.9%** | `experiments/experiment2/augtrue_bs128/` |
| true | 32 | 91 | 0.061 | 82.7% | `experiments/experiment2/augtrue_bs32/` |
| true | 64 | 62 | 0.062 | 81.8% | `experiments/experiment2/augtrue_bs64/` |
| false | 32 | 27 | 0.070 | 79.3% | `experiments/experiment2/augfalse_bs32/` |
| false | 64 | 54 | 0.070 | 79.2% | `experiments/experiment2/augfalse_bs64/` |
| false | 128 | 83 | 0.073 | 78.8% | `experiments/experiment2/augfalse_bs128/` |

- **Conclusion:** Augmentation consistently improves results across all
  batch sizes, reducing CER from ~0.070 to ~0.060 and lifting word accuracy
  from ~79% to ~83%. Augmented runs need more epochs to converge (62–142
  vs 27–83). Batch size has minimal effect with augmentation enabled.

### 2026-06-05 — Experiment 3: dropout

- **Hypothesis:** Dropout between RNN layers provides additional
  regularisation, especially when combined with augmentation.
- **Setup:** LR=0.0005, BS=64. Dropout ∈ {0.0, 0.2, 0.5} ×
  augmentation {on, off}. Max 200 epochs, patience 25. NVIDIA A100 40GB.
- **Command:** `bash xournalpp_htr/training/simple_htr/run_training.sh`
- **Code revision:** `f5f8c60`
- **Results:**

| Dropout | Aug | Best Epoch | CER | Word Acc | Path |
|---|---|---|---|---|---|
| 0.5 | true | 67 | **0.056** | **84.2%** | `experiments/experiment3/do0.5_augtrue/` |
| 0.2 | true | 79 | 0.058 | 83.2% | `experiments/experiment3/do0.2_augtrue/` |
| 0.0 | true | 62 | 0.062 | 81.8% | `experiments/experiment3/do0.0_augtrue/` |
| 0.5 | false | 64 | 0.067 | 80.4% | `experiments/experiment3/do0.5_augfalse/` |
| 0.2 | false | 22 | 0.070 | 78.8% | `experiments/experiment3/do0.2_augfalse/` |
| 0.0 | false | 54 | 0.070 | 79.2% | `experiments/experiment3/do0.0_augfalse/` |

- **Conclusion:** Dropout and augmentation are complementary. The best
  config (dropout=0.5, augmentation on) achieves CER 0.056 and 84.2% word
  accuracy — a major improvement over the experiment 1 baseline (CER 0.070,
  79.2%). Higher dropout consistently helps; the effect is strongest when
  combined with augmentation. Recommended defaults: dropout=0.5,
  augmentation enabled.

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

- Add beam search decoding (issue #120)
- Integrate into full pipeline (word detection + recognition)
