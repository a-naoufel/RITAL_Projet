# Sentiment Classification — RITAL M1

Binary sentiment classification (positive / negative) on English movie reviews.
Comparative study of 7 model families: TF-IDF, ensemble, fastText, BiLSTM + GloVe,
SetFit, DistilBERT, and DeBERTa.

**Best result: SetFit full (all-mpnet-base-v2) — 0.9125 CV accuracy**

---

## Table of contents

1. [Project structure](#project-structure)
2. [Quick start](#quick-start)
3. [Full install guide](#full-install-guide)
4. [Data setup](#data-setup)
5. [Running each model](#running-each-model)
6. [Results summary](#results-summary)
7. [Generating predictions](#generating-predictions)
8. [Troubleshooting](#troubleshooting)

---

## Project structure

```
RITAL_Projet/
│
├── data/
│   └── raw/
│       ├── movies1000/          # 2000 labelled reviews
│       │   ├── pos/             # 1000 positive .txt files
│       │   └── neg/             # 1000 negative .txt files
│       └── testSentiment.txt    # 25000 unlabelled test lines
│
├── models/
│   ├── nbsvm/                   # Naive Bayes SVM baseline
│   ├── tfidf_svc/               # Model 1 — TF-IDF + LinearSVC
│   ├── tfidf_ensemble/          # Model 2 — word + char ensemble
│   ├── fasttext/                # Model 3 — fastText supervised
│   ├── bilstm_glove/            # Model 4 — RNN / BiLSTM + GloVe
│   ├── setfit/                  # Model 5 — SetFit
│   ├── distilbert/              # Model 6 — DistilBERT (base + SST-2)
│   │                            #           + Model 7 DeBERTa + Appendix RoBERTa
│   └── roberta/                 # Appendix — RoBERTa SST-2
│
├── results/                     # Generated automatically on first run
│   ├── model1_tfidf_svc/
│   ├── model2_tfidf_ensemble/
│   ├── model3_fasttext/
│   ├── model4_bilstm/
│   ├── model4_setfit/
│   ├── model5_distilbert_base/
│   ├── model5_distilbert_sst2/
│   ├── model6_deberta/
│   └── appendix_roberta/
│
├── submissions/                 # Prediction files for the test set
│   └── results/
│       └── scores.md            # Manual results log
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick start

```bash
# 1 — Clone the repo
git clone https://github.com/<your-username>/RITAL_Projet.git
cd RITAL_Projet

# 2 — Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3 — Install PyTorch for your hardware (pick ONE)
# GPU (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
# GPU (CUDA 11.8):
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4 — Install all other dependencies
pip install -r requirements.txt

# 5 — Run the fastest model to verify everything works
python models/tfidf_svc/train_tfidf_svc_benchmark.py
```

Expected output on step 5:
```
Loaded 1000 pos + 1000 neg = 2000 reviews
5-fold CV grid search
...
Best C=4.0  CV acc=0.8820 +/- 0.0318
All outputs -> results/model1_tfidf_svc/
```

---

## Full install guide

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | — | 8 GB (for DeBERTa) |
| Disk space | 5 GB | 10 GB (models cached) |

### Step-by-step

**Step 1 — Check your CUDA version (if using GPU)**

```bash
nvidia-smi | head -3
# Look for "CUDA Version: XX.X" in the top-right corner
```

**Step 2 — Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

> Always activate the venv before running any script:
> `source venv/bin/activate`

**Step 3 — Install PyTorch (must be done before requirements.txt)**

```bash
# CUDA 12.1 (most common on recent machines):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only (slower, but works for all models except DeBERTa in reasonable time):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify GPU is detected:
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

**Step 4 — Install all remaining dependencies**

```bash
pip install -r requirements.txt
```

**Step 5 — Install fastText from source (recommended over pip wheel)**

```bash
git clone https://github.com/facebookresearch/fastText.git /tmp/fasttext
cd /tmp/fasttext
pip install .
cd -
```

If source build fails (missing gcc/cmake), use the pip fallback:
```bash
pip install fasttext-wheel
```

**Step 6 — Download GloVe vectors (required for BiLSTM only)**

```bash
mkdir -p data/glove
wget https://nlp.stanford.edu/data/glove.6B.zip -P data/glove/
unzip data/glove/glove.6B.zip -d data/glove/
# You only need glove.6B.100d.txt — the others can be deleted
rm data/glove/glove.6B.{50,200,300}d.txt
```

---

## Data setup

The data is **not committed to the repo** (too large, see `.gitignore`).
Place your files as follows:

```
data/raw/movies1000/pos/cv000_29416.txt   ← positive reviews
data/raw/movies1000/neg/cv000_29590.txt   ← negative reviews
data/raw/testSentiment.txt                ← 25000 test lines
```

If your data is still in the old `dataset/` location, either rename the
folder or update the `DATA_DIR` path at the top of each benchmark script:

```python
# In each train_*.py — change this line:
DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"
# To:
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "movies1000"
```

---

## Running each model

All benchmark scripts follow the same pattern:
- Train with cross-validation / grid search
- Save figures to `results/<model_name>/`
- Save `scores.json` and `report_snippet.txt`

### Model 1 — TF-IDF + LinearSVC

```bash
cd models/tfidf_svc
python train_tfidf_svc_benchmark.py
# Runtime: ~1 min (CPU)
# Output: results/model1_tfidf_svc/
```

### Model 2 — TF-IDF word + char ensemble

```bash
cd models/tfidf_ensemble
python train_tfidf_ensemble_benchmark.py
# Runtime: ~17 min (CPU, 225 combinations × 5 folds)
# Output: results/model2_tfidf_ensemble/
```

### Model 3 — fastText

```bash
cd models/fasttext
python train_fasttext_benchmark.py
# Runtime: ~4 min (CPU, 60 combinations × 5 folds)
# Output: results/model3_fasttext/
```

### Model 4 — RNN / BiLSTM + GloVe

> Requires GloVe vectors (see [Data setup](#data-setup)).

```bash
cd models/bilstm_glove
python train_bilstm_glove.py
# Runtime: ~12 min (CPU)
# Output: results/model4_bilstm/
```

### Model 5 — SetFit

> Requires a GPU for reasonable runtime. CPU is possible but takes ~4.5 hours.

```bash
cd models/setfit
python train_setfit_benchmark.py
# Runtime: ~20 min (GPU) | ~4.5 hours (CPU)
# Output: results/model4_setfit/
```

### Models 6 & 7 — DistilBERT and DeBERTa

> GPU strongly recommended. CPU will take many hours.

```bash
cd models/distilbert

# Run one at a time (recommended to monitor GPU memory):
python train_transformer_benchmark.py --model distilbert-base
# Runtime: ~15 min (GPU)

python train_transformer_benchmark.py --model distilbert-sst2
# Runtime: ~15 min (GPU)

python train_transformer_benchmark.py --model deberta
# Runtime: ~40 min (GPU, needs ~8 GB VRAM)

# Or run all three sequentially:
python train_transformer_benchmark.py --model all
```

### Appendix — RoBERTa

```bash
cd models/distilbert
python train_transformer_benchmark.py --model roberta
# Runtime: ~25 min (GPU)
```

---

## Results summary

| # | Model | CV accuracy | Std | Hardware |
|---|---|---|---|---|
| 1 | TF-IDF + LinearSVC | 0.882 | 0.032 | CPU |
| 2 | TF-IDF word+char ensemble | 0.887 | 0.029 | CPU |
| 3 | fastText supervised | 0.841 | 0.018 | CPU |
| 4a | RNN simple + GloVe | ~0.72 | — | CPU |
| 4b | BiLSTM + GloVe | ~0.75 | — | CPU |
| 5a | SetFit few-shot (8/class) | 0.717 | 0.114 | GPU |
| **5b** | **SetFit full** | **0.913** | **0.012** | **GPU** |
| 6a | DistilBERT base | 0.797 | 0.007 | GPU |
| 6b | DistilBERT SST-2 | ~0.858 | ~0.015 | GPU |
| 7 | DeBERTa-v3-base | 0.891 | 0.012 | GPU |
| A | RoBERTa SST-2 (appendix) | 0.866 | 0.002 | GPU |

---

## Generating predictions

Each model has a matching predict script. Run the benchmark first to
generate `scores.json`, then run the predict script — it reads the best
hyperparameters automatically.

```bash
# Model 1
python models/tfidf_svc/predict_tfidf_svc.py
# → submissions/preds_model1_tfidf_svc.txt

# Model 2
python models/tfidf_ensemble/predict_tfidf_ensemble.py
# → submissions/preds_model2_tfidf_ensemble.txt

# Model 3
python models/fasttext/predict_fasttext.py
# → submissions/preds_model3_fasttext.txt

# Model 5 (SetFit)
python models/setfit/predict_setfit.py
# → submissions/preds_model4_setfit.txt

# Models 6, 7, appendix (one script, --model argument)
python models/distilbert/predict_transformer.py --model distilbert-base
python models/distilbert/predict_transformer.py --model distilbert-sst2
python models/distilbert/predict_transformer.py --model deberta
python models/distilbert/predict_transformer.py --model roberta
```

All prediction files contain exactly 25000 lines, one label per line
(`P` for positive, `N` for negative). The script prints a line-count
assertion to confirm correctness.

---

## Troubleshooting

### `torch.cuda.is_available()` returns `False`

```bash
# 1 — Check driver version
nvidia-smi | grep "CUDA Version"

# 2 — Uninstall current torch
pip uninstall torch torchvision torchaudio -y

# 3 — Reinstall matching wheel (example for CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 4 — Verify
python3 -c "import torch; print(torch.cuda.is_available())"
```

### `ModuleNotFoundError: No module named 'fasttext'`

```bash
# Try source build first:
git clone https://github.com/facebookresearch/fastText.git /tmp/fasttext
cd /tmp/fasttext && pip install .

# If that fails (no gcc/cmake), use the wheel:
pip install fasttext-wheel
```

### `FileNotFoundError` on data files

All scripts expect data at `data/raw/movies1000/`. If your data is
elsewhere, update `DATA_DIR` at the top of the relevant script.

### DeBERTa OOM (out of memory) on GPU

Reduce batch size in `MODEL_CONFIGS` inside `train_transformer_benchmark.py`:
```python
"deberta": {
    "batch_size": 4,   # reduce from 8 to 4
    ...
}
```

### SetFit is very slow on CPU

Set `num_iterations=5` in `TrainingArguments` (down from 20) to reduce
contrastive pair generation — modest accuracy loss, 4× faster.

### `scores.json not found` when running predict scripts

You must run the corresponding benchmark script first to generate
`results/<model>/scores.json`. The predict scripts read hyperparameters
from that file.

---

## Reproducing all results from scratch

```bash
# Activate environment
source venv/bin/activate

# Classical models (CPU, ~25 min total)
python models/tfidf_svc/train_tfidf_svc_benchmark.py
python models/tfidf_ensemble/train_tfidf_ensemble_benchmark.py
python models/fasttext/train_fasttext_benchmark.py
python models/bilstm_glove/train_bilstm_glove.py

# Neural models (GPU recommended)
python models/setfit/train_setfit_benchmark.py
python models/distilbert/train_transformer_benchmark.py --model distilbert-base
python models/distilbert/train_transformer_benchmark.py --model distilbert-sst2
python models/distilbert/train_transformer_benchmark.py --model deberta
python models/distilbert/train_transformer_benchmark.py --model roberta
```

Results land in `results/*/scores.json` and `results/*/report_snippet.txt`.

---

*Sorbonne Université — M1 RITAL — 2024--2025*
