"""
models/fasttext/train_fasttext_benchmark.py
============================================
Model 3 — fastText supervised classification

Why fastText
------------
fastText learns subword embeddings (character n-grams hashed into a shared
embedding table) combined with word-level bag-of-words. Unlike TF-IDF, it
produces dense representations that generalise across morphological variants
("horrible" / "horribly" share subword structure). Training is extremely fast
even on CPU. It is the natural bridge between counting-based models (Models 1-2)
and full neural models (Models 4-6).

Architecture
------------
- Input     : raw text (fastText handles its own tokenisation)
- Embedding : word vectors + subword (char n-gram hashes) averaged into sentence vector
- Classifier: softmax over label set (binary here: __label__pos / __label__neg)
- Loss       : softmax (default) or hs (hierarchical softmax, faster, used here)

Grid
----
lr          : [0.05, 0.1, 0.25, 0.5, 1.0]
epoch       : [5, 10, 25]
wordNgrams  : [1, 2]          (word n-gram context window)
dim         : [50, 100]       (embedding dimension)

Total combinations : 5 × 3 × 2 × 2 = 60
CV protocol        : 5-fold stratified, seed=42
Expected runtime   : 3-8 min on CPU

Outputs  →  results/model3_fasttext/
  scores.json           full grid + best config
  lr_curve.png          lr vs CV accuracy at best (epoch, wordNgrams, dim)
  confusion_matrix.png  hold-out confusion matrix
  report_snippet.txt    ready-to-paste numbers

Requirements
------------
  pip install fasttext        # official compiled wheel
  OR build from source:
    git clone https://github.com/facebookresearch/fastText
    cd fastText && pip install .
"""

from pathlib import Path
import json, time, tempfile, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    import fasttext
except ImportError:
    raise ImportError(
        "fasttext not found. Install with:\n"
        "  pip install fasttext\n"
        "or build from source: https://github.com/facebookresearch/fastText"
    )

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 42

LR_GRID         = [0.05, 0.1, 0.25, 0.5, 1.0]
EPOCH_GRID      = [5, 10, 25]
WORD_NGRAM_GRID = [1, 2]
DIM_GRID        = [50, 100]

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"
OUT_DIR      = PROJECT_ROOT / "results" / "model3_fasttext"
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


def to_fasttext_label(y: int) -> str:
    return "__label__pos" if y == 1 else "__label__neg"


def write_fasttext_file(texts, labels, path: Path):
    """Write a fastText-format training file: one doc per line, label first."""
    with path.open("w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            # Collapse newlines — fastText treats each line as one document
            clean = text.replace("\n", " ").replace("\r", " ").strip()
            f.write(f"{to_fasttext_label(label)} {clean}\n")

# ── Single fold evaluation ─────────────────────────────────────────────────────
def eval_fold(X_tr, y_tr, X_va, y_va, lr, epoch, wordNgrams, dim):
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "train.txt"
        write_fasttext_file(X_tr, y_tr, train_file)

        model = fasttext.train_supervised(
            input       = str(train_file),
            lr          = lr,
            epoch       = epoch,
            wordNgrams  = wordNgrams,
            dim         = dim,
            loss        = "softmax",
            seed        = SEED,
            verbose     = 0,
        )

        preds = []
        for text in X_va:
            clean = text.replace("\n", " ").replace("\r", " ").strip()
            label, _ = model.predict(clean)
            preds.append(1 if label[0] == "__label__pos" else 0)

        return accuracy_score(y_va, preds)

# ── 5-fold CV grid search ──────────────────────────────────────────────────────
def run_grid(X, y):
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    X_arr = np.array(X, dtype=object)
    y_arr = np.array(y)

    combos = list(itertools.product(LR_GRID, EPOCH_GRID, WORD_NGRAM_GRID, DIM_GRID))
    total  = len(combos)
    print(f"\n5-fold CV grid search: {total} combinations")
    print(f"{'lr':<7} {'ep':<5} {'ng':<4} {'dim':<5} {'cv_mean':>8} {'std':>7}")
    print("-" * 44)

    results = []
    best    = {"cv_mean": -1}

    for i, (lr, epoch, wng, dim) in enumerate(combos, 1):
        fold_accs = []
        for train_idx, val_idx in cv.split(X_arr, y_arr):
            acc = eval_fold(
                X_arr[train_idx].tolist(), y_arr[train_idx],
                X_arr[val_idx].tolist(),   y_arr[val_idx],
                lr=lr, epoch=epoch, wordNgrams=wng, dim=dim,
            )
            fold_accs.append(acc)

        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))

        entry = {
            "lr": lr, "epoch": epoch, "wordNgrams": wng, "dim": dim,
            "cv_mean": mean_acc, "cv_std": std_acc,
        }
        results.append(entry)

        marker = " <-" if mean_acc > best["cv_mean"] else ""
        print(f"{lr:<7} {epoch:<5} {wng:<4} {dim:<5} {mean_acc:>8.4f} {std_acc:>7.4f}{marker}")

        if mean_acc > best["cv_mean"]:
            best = entry.copy()

        if i % 10 == 0:
            print(f"  [{i}/{total} done]")

    return results, best

# ── Hold-out evaluation ────────────────────────────────────────────────────────
def holdout_eval(X, y, best):
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "train.txt"
        write_fasttext_file(Xtr, ytr, train_file)

        model = fasttext.train_supervised(
            input      = str(train_file),
            lr         = best["lr"],
            epoch      = best["epoch"],
            wordNgrams = best["wordNgrams"],
            dim        = best["dim"],
            loss       = "softmax",
            seed       = SEED,
            verbose    = 0,
        )

        preds = []
        for text in Xva:
            clean = text.replace("\n", " ").replace("\r", " ").strip()
            label, _ = model.predict(clean)
            preds.append(1 if label[0] == "__label__pos" else 0)

    acc    = accuracy_score(yva, preds)
    cm     = confusion_matrix(yva, preds)
    report = classification_report(yva, preds, target_names=["Negative", "Positive"])
    print(f"\nHold-out accuracy: {acc:.4f}")
    print(report)
    return acc, cm, report

# ── Figures ────────────────────────────────────────────────────────────────────
def plot_lr_curve(results, best, path):
    sub = [r for r in results
           if r["epoch"]      == best["epoch"]
           and r["wordNgrams"] == best["wordNgrams"]
           and r["dim"]        == best["dim"]]
    lrs   = [r["lr"]      for r in sub]
    means = [r["cv_mean"] for r in sub]
    stds  = [r["cv_std"]  for r in sub]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(lrs, means, "o-", color="#1a5f9e", lw=2, ms=6,
                label=f"CV acc  (epoch={best['epoch']}, ng={best['wordNgrams']}, dim={best['dim']})")
    ax.fill_between(lrs,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color="#1a5f9e", label="+/- 1 std")
    best_i = int(np.argmax(means))
    ax.axvline(best["lr"], color="#c0392b", lw=1.5, ls="--",
               label=f"Best lr = {best['lr']}")
    ax.scatter([best["lr"]], [means[best_i]], color="#c0392b", zorder=5, s=80)
    ax.annotate(f"{means[best_i]:.4f}",
                xy=(best["lr"], means[best_i]),
                xytext=(best["lr"] * 1.3, means[best_i] - 0.004),
                fontsize=9, color="#c0392b")
    ax.set_xlabel("Learning rate (log scale)", fontsize=11)
    ax.set_ylabel("5-fold CV accuracy", fontsize=11)
    ax.set_title("fastText — learning rate tuning", fontsize=12, pad=10)
    ax.set_xticks(lrs)
    ax.set_xticklabels([str(lr) for lr in lrs])
    ax.legend(fontsize=9)
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
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion matrix — hold-out set", fontsize=12, pad=10)
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

    grid_results, best = run_grid(X, y)

    print(f"\nBest config:")
    print(f"  lr         = {best['lr']}")
    print(f"  epoch      = {best['epoch']}")
    print(f"  wordNgrams = {best['wordNgrams']}")
    print(f"  dim        = {best['dim']}")
    print(f"  CV acc     = {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}")

    holdout_acc, cm, clf_report = holdout_eval(X, y, best)

    plot_lr_curve(grid_results, best, OUT_DIR / "lr_curve.png")
    plot_cm(cm,                       OUT_DIR / "confusion_matrix.png")

    out = {
        "model":        "fastText supervised",
        "cv_folds":     5,
        "seed":         SEED,
        "grid":         grid_results,
        "best_lr":      best["lr"],
        "best_epoch":   best["epoch"],
        "best_ngrams":  best["wordNgrams"],
        "best_dim":     best["dim"],
        "best_cv_acc":  best["cv_mean"],
        "best_cv_std":  best["cv_std"],
        "holdout_acc":  holdout_acc,
        "confusion_matrix": cm.tolist(),
        "elapsed_s":    round(time.time() - t0, 1),
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2))

    snippet = (
        f"=== Model 3 - fastText supervised ===\n"
        f"lr={best['lr']}  epoch={best['epoch']}  "
        f"wordNgrams={best['wordNgrams']}  dim={best['dim']}\n\n"
        f"5-fold CV  : {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}\n"
        f"Hold-out   : {holdout_acc:.4f}  (80/20, seed={SEED})\n\n"
        f"Confusion matrix (hold-out):\n"
        f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n"
        f"{clf_report}\n"
        f"Figures    : lr_curve.png  confusion_matrix.png\n"
        f"Time       : {round(time.time()-t0,1)}s\n"
    )
    (OUT_DIR / "report_snippet.txt").write_text(snippet)
    print(snippet)
    print(f"All outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
