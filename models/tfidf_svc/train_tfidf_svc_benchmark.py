"""
models/tfidf_svc/train_tfidf_svc_benchmark.py
==============================================
TF-IDF (word n-grams) + LinearSVC — Model 1
Clean benchmark replacing the old train_tfidf_svc.py.

Fixes vs old script
-------------------
- Vectorizer was passed by reference → char model reused fitted word vectorizer.
  Fix: pipeline instantiated fresh for every (C, analyzer) combination.
- No random_state / max_iter on LinearSVC → non-reproducible, convergence warnings.
  Fix: random_state=SEED, max_iter=2000.
- Single 80/20 split → high-variance estimate (±~2.5% 95-CI on 400 examples).
  Fix: 5-fold stratified CV on the full 2000 reviews as primary metric.

Outputs  →  results/model1_tfidf_svc/
  scores.json          CV means/stds, best C, hold-out accuracy
  tuning_curve.png     C vs CV accuracy (report figure)
  confusion_matrix.png hold-out confusion matrix (report figure)
  report_snippet.txt   ready-to-paste numbers
"""

from pathlib import Path
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# ── Config ─────────────────────────────────────────────────────────────────────
SEED    = 42
C_GRID  = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
N_FOLDS = 5

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]          # models/tfidf_svc/ -> project root
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"   # update path after repo rename
OUT_DIR      = PROJECT_ROOT / "results" / "model1_tfidf_svc"
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
        raise FileNotFoundError(f"No .txt files found in {data_dir}/pos/")
    texts  = [p.read_text(encoding="utf-8", errors="ignore") for p in pos]
    texts += [p.read_text(encoding="utf-8", errors="ignore") for p in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"Loaded {len(pos)} pos + {len(neg)} neg = {len(texts)} reviews")
    return texts, labels

# ── Pipeline factory — always returns a FRESH unfitted pipeline ────────────────
def make_pipeline(C: float) -> Pipeline:
    """
    Key design choices:
    - word analyzer + (1,2)-grams: captures local phrase patterns ("not good")
    - sublinear_tf: log-dampens high-frequency terms (better than raw TF)
    - min_df=2: drops hapax legomena (noise on 2000 docs)
    - max_df=0.95: drops near-universal terms without a stopword list
    - token_pattern: alpha tokens >= 2 chars, drops digits and punctuation
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer      = "word",
            ngram_range   = (1, 2),
            sublinear_tf  = True,
            min_df        = 2,
            max_df        = 0.95,
            strip_accents = "unicode",
            token_pattern = r"\b[a-zA-Z]{2,}\b",
        )),
        ("clf", LinearSVC(
            C            = C,
            max_iter     = 2000,
            random_state = SEED,
        )),
    ])

# ── 5-fold stratified CV grid search over C ────────────────────────────────────
def cv_grid(X, y):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    means, stds = [], []
    print(f"\n{N_FOLDS}-fold CV grid search")
    print(f"{'C':<8} {'mean acc':>10} {'std':>8}")
    print("-" * 30)
    for C in C_GRID:
        scores = cross_val_score(
            make_pipeline(C), X, y,
            cv=cv, scoring="accuracy", n_jobs=-1
        )
        means.append(scores.mean())
        stds.append(scores.std())
        print(f"{C:<8} {scores.mean():>10.4f} {scores.std():>8.4f}")
    return np.array(means), np.array(stds)

# ── Hold-out evaluation at best C ─────────────────────────────────────────────
def holdout_eval(X, y, best_C):
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    pipe = make_pipeline(best_C)
    pipe.fit(Xtr, ytr)
    preds  = pipe.predict(Xva)
    acc    = accuracy_score(yva, preds)
    cm     = confusion_matrix(yva, preds)
    report = classification_report(yva, preds, target_names=["Negative", "Positive"])
    print(f"\nHold-out accuracy (C={best_C}): {acc:.4f}")
    print(report)
    return acc, cm, report

# ── Figures ────────────────────────────────────────────────────────────────────
def plot_tuning(C_grid, means, stds, best_C, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(C_grid, means, "o-", color="#1a5f9e", lw=2, ms=6,
                label=f"{N_FOLDS}-fold CV accuracy")
    ax.fill_between(C_grid, means - stds, means + stds,
                    alpha=0.15, color="#1a5f9e", label="+/- 1 std")
    best_i = int(np.argmax(means))
    ax.axvline(best_C, color="#c0392b", lw=1.5, ls="--",
               label=f"Best C = {best_C}")
    ax.scatter([best_C], [means[best_i]], color="#c0392b", zorder=5, s=80)
    ax.annotate(f"{means[best_i]:.4f}",
                xy=(best_C, means[best_i]),
                xytext=(best_C * 1.5, means[best_i] - 0.005),
                fontsize=9, color="#c0392b")
    ax.set_xlabel("Regularisation parameter C  (log scale)", fontsize=11)
    ax.set_ylabel(f"{N_FOLDS}-fold CV accuracy", fontsize=11)
    ax.set_title("TF-IDF + LinearSVC — hyperparameter tuning", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.set_xticks(C_grid)
    ax.set_xticklabels([str(c) for c in C_grid], rotation=45, ha="right", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
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
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    X, y = load_docs(DATA_DIR)

    means, stds  = cv_grid(X, y)
    best_i       = int(np.argmax(means))
    best_C       = C_GRID[best_i]
    best_cv_mean = float(means[best_i])
    best_cv_std  = float(stds[best_i])
    print(f"\nBest C={best_C}  CV acc={best_cv_mean:.4f} +/- {best_cv_std:.4f}")

    holdout_acc, cm, clf_report = holdout_eval(X, y, best_C)

    plot_tuning(C_GRID, means, stds, best_C, OUT_DIR / "tuning_curve.png")
    plot_cm(cm, OUT_DIR / "confusion_matrix.png")

    results = {
        "model":            "TF-IDF word (1,2) + LinearSVC",
        "cv_folds":         N_FOLDS,
        "seed":             SEED,
        "C_grid":           C_GRID,
        "cv_means":         means.tolist(),
        "cv_stds":          stds.tolist(),
        "best_C":           best_C,
        "best_cv_acc":      best_cv_mean,
        "best_cv_std":      best_cv_std,
        "holdout_acc":      holdout_acc,
        "confusion_matrix": cm.tolist(),
        "elapsed_s":        round(time.time() - t0, 1),
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(results, indent=2))

    snippet = (
        f"=== Model 1 - TF-IDF + LinearSVC ===\n"
        f"Vectoriser : word, ngram=(1,2), sublinear_tf, min_df=2, max_df=0.95\n"
        f"Classifier : LinearSVC  best C={best_C}\n\n"
        f"5-fold CV  : {best_cv_mean:.4f} +/- {best_cv_std:.4f}\n"
        f"Hold-out   : {holdout_acc:.4f}  (80/20, seed={SEED})\n\n"
        f"Confusion matrix (hold-out):\n"
        f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n"
        f"{clf_report}\n"
        f"Figures : tuning_curve.png  confusion_matrix.png\n"
        f"Time    : {round(time.time()-t0, 1)}s\n"
    )
    (OUT_DIR / "report_snippet.txt").write_text(snippet)
    print(snippet)
    print(f"All outputs -> {OUT_DIR}")

if __name__ == "__main__":
    main()
