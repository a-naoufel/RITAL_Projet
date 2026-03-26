"""
models/setfit/train_setfit_benchmark.py
========================================
Model 4 — SetFit (Sentence Transformer + logistic head)

Why SetFit
----------
SetFit (Tunstall et al., 2022) fine-tunes a sentence encoder via contrastive
learning on sentence pairs, then trains a lightweight classifier head on the
resulting embeddings. Its key claim: strong performance with very few labelled
examples (few-shot regime). Testing it at both few-shot and full-data regimes
lets us directly evaluate whether pretrained sentence representations add value
beyond what a classical model achieves on this corpus size.

Architecture
------------
1. Sentence encoder : all-mpnet-base-v2 (768-dim embeddings)
2. Contrastive fine-tuning on (anchor, positive, negative) sentence triplets
   sampled from the labelled set — the encoder learns to pull same-class
   sentences together and push different-class apart.
3. Classification head : logistic regression on frozen post-fine-tuning embeddings.

Two regimes compared
--------------------
- Few-shot  : 8 examples per class (16 total) — SetFit's intended use case
- Full data : all 1600 training examples (80/20 split of 2000)

For few-shot: repeated 5 times with different random seeds to estimate variance.
For full data: 5-fold stratified CV to match other models.

Outputs  →  results/model4_setfit/
  scores.json
  regime_comparison.png   few-shot vs full bar chart
  confusion_matrix_full.png
  report_snippet.txt

Requirements
------------
  pip install setfit
"""

from pathlib import Path
import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datasets import Dataset

warnings.filterwarnings("ignore")

try:
    from setfit import SetFitModel, Trainer, TrainingArguments
except ImportError:
    raise ImportError("setfit not found. Install with: pip install setfit")

# ── Config ─────────────────────────────────────────────────────────────────────
SEED           = 42
MODEL_NAME     = "sentence-transformers/all-mpnet-base-v2"
N_FOLDS        = 5
FEW_SHOT_K     = 8      # examples per class
FEW_SHOT_REPS  = 5      # repeated sampling to estimate variance

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"
OUT_DIR      = PROJECT_ROOT / "results" / "model4_setfit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

# ── Data ───────────────────────────────────────────────────────────────────────
def load_docs(data_dir: Path):
    pos = sorted((data_dir / "pos").glob("*.txt"))
    neg = sorted((data_dir / "neg").glob("*.txt"))
    if not pos:
        raise FileNotFoundError(f"No reviews in {data_dir}/pos/")
    texts  = [p.read_text(encoding="utf-8", errors="ignore") for p in pos]
    texts += [p.read_text(encoding="utf-8", errors="ignore") for p in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"Loaded {len(pos)} pos + {len(neg)} neg = {len(texts)} reviews")
    return texts, labels


def make_hf_dataset(texts, labels):
    return Dataset.from_dict({"text": texts, "label": labels})


def sample_few_shot(texts, labels, k, seed):
    """Sample k examples per class, return (texts, labels)."""
    rng    = np.random.RandomState(seed)
    X, y   = np.array(texts, dtype=object), np.array(labels)
    chosen = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        chosen.extend(rng.choice(idx, size=k, replace=False).tolist())
    rng.shuffle(chosen)
    return X[chosen].tolist(), y[chosen].tolist()

# ── Training helpers ───────────────────────────────────────────────────────────
def train_and_eval(train_texts, train_labels, val_texts, val_labels,
                   num_epochs=1, batch_size=16):
    """
    Fine-tune a fresh SetFit model and return accuracy on val set.
    A new model is loaded each call to avoid state leakage between folds.
    """
    model = SetFitModel.from_pretrained(MODEL_NAME)

    train_ds = make_hf_dataset(train_texts, train_labels)
    val_ds   = make_hf_dataset(val_texts,   val_labels)

    args = TrainingArguments(
        num_epochs              = num_epochs,
        batch_size              = batch_size,
        num_iterations          = 20,      # contrastive pairs per sample
        evaluation_strategy     = "no",
        seed                    = SEED,
    )

    trainer = Trainer(
        model     = model,
        args      = args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
    )
    trainer.train()

    preds = model.predict(val_texts)
    preds = np.array(preds)
    return accuracy_score(val_labels, preds), preds

# ── Regime 1: few-shot (k per class, repeated FEW_SHOT_REPS times) ────────────
def run_few_shot(X, y):
    print(f"\n── Few-shot regime  ({FEW_SHOT_K} examples/class × {FEW_SHOT_REPS} seeds) ──")
    # Hold out 20% as fixed test set across all repetitions
    Xtr_pool, Xva, ytr_pool, yva = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    accs = []
    for rep in range(FEW_SHOT_REPS):
        seed_rep = SEED + rep
        Xtr_few, ytr_few = sample_few_shot(Xtr_pool, ytr_pool, FEW_SHOT_K, seed_rep)
        print(f"  Rep {rep+1}/{FEW_SHOT_REPS}  ({len(Xtr_few)} train examples, seed={seed_rep})")
        acc, _ = train_and_eval(Xtr_few, ytr_few, Xva, yva)
        accs.append(acc)
        print(f"  Accuracy: {acc:.4f}")

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    print(f"\nFew-shot mean: {mean_acc:.4f} +/- {std_acc:.4f}")
    return mean_acc, std_acc, accs

# ── Regime 2: full training — 5-fold CV ───────────────────────────────────────
def run_full_cv(X, y):
    print(f"\n── Full-data regime  (5-fold CV, {len(X)} examples) ──")
    cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    X_arr = np.array(X, dtype=object)
    y_arr = np.array(y)

    fold_accs = []
    all_preds = np.zeros(len(y_arr), dtype=int)
    all_true  = np.zeros(len(y_arr), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr), 1):
        print(f"  Fold {fold}/{N_FOLDS}")
        Xtr = X_arr[train_idx].tolist()
        Xva = X_arr[val_idx].tolist()
        ytr = y_arr[train_idx].tolist()
        yva = y_arr[val_idx].tolist()

        acc, preds = train_and_eval(Xtr, ytr, Xva, yva)
        fold_accs.append(acc)
        all_preds[val_idx] = preds
        all_true[val_idx]  = y_arr[val_idx]
        print(f"  Fold {fold} accuracy: {acc:.4f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    cm       = confusion_matrix(all_true, all_preds)
    report   = classification_report(
        all_true, all_preds, target_names=["Negative", "Positive"]
    )
    print(f"\nFull CV mean: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(report)
    return mean_acc, std_acc, cm, report

# ── Figures ────────────────────────────────────────────────────────────────────
def plot_regime_comparison(few_mean, few_std, full_mean, full_std,
                           ref_model1, ref_model2, path):
    """Bar chart comparing few-shot vs full SetFit vs TF-IDF baselines."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = [
        f"SetFit\nfew-shot\n({FEW_SHOT_K}/class)",
        "SetFit\nfull data",
        "TF-IDF+SVC\n(model 1)",
        "Ensemble\n(model 2)",
    ]
    means  = [few_mean,  full_mean,  ref_model1, ref_model2]
    errors = [few_std,   full_std,   0,          0]
    colors = ["#5DCAA5", "#1D9E75",  "#B4B2A9",  "#888780"]

    bars = ax.bar(labels, means, yerr=errors, capsize=5,
                  color=colors, width=0.55,
                  error_kw={"elinewidth": 1.5, "ecolor": "#444441"})

    ax.set_ylim(0.75, 0.95)
    ax.set_ylabel("CV accuracy", fontsize=11)
    ax.set_title("SetFit — few-shot vs full training vs classical baselines",
                 fontsize=11, pad=10)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cm(cm, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = ["Negative", "Positive"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("SetFit full — confusion matrix (CV)", fontsize=12, pad=10)
    ax.grid(False)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    X, y = load_docs(DATA_DIR)

    print(f"Base model : {MODEL_NAME}")
    print(f"Device     : (SetFit auto-selects GPU if available)")

    # Few-shot regime
    few_mean, few_std, few_accs = run_few_shot(X, y)

    # Full CV regime
    full_mean, full_std, cm, clf_report = run_full_cv(X, y)

    # Reference scores from Models 1 and 2
    ref_m1 = 0.8820
    ref_m2 = 0.8870

    plot_regime_comparison(few_mean, few_std, full_mean, full_std,
                           ref_m1, ref_m2,
                           OUT_DIR / "regime_comparison.png")
    plot_cm(cm, OUT_DIR / "confusion_matrix_full.png")

    out = {
        "model":           "SetFit — all-mpnet-base-v2",
        "base_model":      MODEL_NAME,
        "few_shot_k":      FEW_SHOT_K,
        "few_shot_reps":   FEW_SHOT_REPS,
        "few_shot_accs":   few_accs,
        "few_shot_mean":   few_mean,
        "few_shot_std":    few_std,
        "full_cv_folds":   N_FOLDS,
        "full_cv_mean":    full_mean,
        "full_cv_std":     full_std,
        "confusion_matrix_full": cm.tolist(),
        "elapsed_s":       round(time.time() - t0, 1),
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2))

    snippet = (
        f"=== Model 4 - SetFit ({MODEL_NAME}) ===\n\n"
        f"Few-shot ({FEW_SHOT_K}/class, {FEW_SHOT_REPS} seeds):\n"
        f"  Mean acc : {few_mean:.4f} +/- {few_std:.4f}\n"
        f"  Per-seed : {[round(a,4) for a in few_accs]}\n\n"
        f"Full data (5-fold CV):\n"
        f"  CV acc   : {full_mean:.4f} +/- {full_std:.4f}\n\n"
        f"Confusion matrix (full CV):\n"
        f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n"
        f"{clf_report}\n"
        f"Figures    : regime_comparison.png  confusion_matrix_full.png\n"
        f"Time       : {round(time.time()-t0,1)}s\n"
    )
    (OUT_DIR / "report_snippet.txt").write_text(snippet)
    print(snippet)
    print(f"All outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
