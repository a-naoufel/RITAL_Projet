"""
models/tfidf_ensemble/train_tfidf_ensemble_benchmark.py
========================================================
Model 2 — TF-IDF word + char ensemble
Full 3-dimensional grid search: C_word × C_char × alpha

Architecture
------------
Two independent LinearSVC pipelines:
  - word pipeline : TF-IDF word (1,2)-grams  + LinearSVC(C_word)
  - char pipeline : TF-IDF char_wb (3,5)-grams + LinearSVC(C_char)

Combination:
  score = alpha * decision_word + (1 - alpha) * decision_char
  label = 1 if score > 0 else 0

The decision_function() output is the signed distance to the hyperplane —
a richer signal than hard labels, allowing soft interpolation between models.

Why this works
--------------
Word n-grams capture semantic content ("not good", "brilliant performance").
Char n-grams capture morphological patterns ("terrible", "terribl-", suffixes)
and are robust to typos and informal spelling. The two models make partially
independent errors, so their combination outperforms either alone.

Grid
----
C_word  : [0.5, 1, 2, 4, 8]
C_char  : [0.1, 0.25, 0.5, 1, 2]
alpha   : [0.1, 0.2, ..., 0.9]   (weight on word model)

Total combinations : 5 × 5 × 9 = 225
CV protocol        : 5-fold stratified, seed=42
Expected runtime   : 5-10 min on CPU

Outputs  →  results/model2_tfidf_ensemble/
  scores.json           full grid results + best config
  alpha_curve.png       alpha vs accuracy at best (C_word, C_char)
  c_heatmap.png         C_word × C_char heatmap at best alpha
  confusion_matrix.png  hold-out confusion matrix
  report_snippet.txt    ready-to-paste numbers
"""

from pathlib import Path
import json, time, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Config ─────────────────────────────────────────────────────────────────────
SEED     = 42
N_FOLDS  = 5

C_WORD_GRID  = [0.5, 1.0, 2.0, 4.0, 8.0]
C_CHAR_GRID  = [0.1, 0.25, 0.5, 1.0, 2.0]
ALPHA_GRID   = [round(a, 1) for a in np.arange(0.1, 1.0, 0.1)]  # 0.1 … 0.9

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"
OUT_DIR      = PROJECT_ROOT / "results" / "model2_tfidf_ensemble"
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
        raise FileNotFoundError(f"No reviews found in {data_dir}/pos/")
    texts  = [p.read_text(encoding="utf-8", errors="ignore") for p in pos]
    texts += [p.read_text(encoding="utf-8", errors="ignore") for p in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"Loaded {len(pos)} pos + {len(neg)} neg = {len(texts)} reviews")
    return texts, labels

# ── Pipeline factories ─────────────────────────────────────────────────────────
def make_word_pipeline(C: float) -> Pipeline:
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
        ("clf", LinearSVC(C=C, max_iter=2000, random_state=SEED)),
    ])

def make_char_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer     = "char_wb",
            ngram_range  = (3, 5),
            sublinear_tf = True,
            min_df       = 2,
            max_df       = 0.95,
        )),
        ("clf", LinearSVC(C=C, max_iter=2000, random_state=SEED)),
    ])

# ── Ensemble scorer ────────────────────────────────────────────────────────────
def ensemble_predict(word_pipe, char_pipe, X, alpha: float):
    """
    Combine signed decision scores, return binary labels.
    decision_function() is always available for LinearSVC.
    """
    score_word = word_pipe.decision_function(X)
    score_char = char_pipe.decision_function(X)
    combined   = alpha * score_word + (1.0 - alpha) * score_char
    return (combined > 0).astype(int)

# ── 5-fold CV for one (C_word, C_char, alpha) triple ──────────────────────────
def cv_ensemble(X, y, C_word, C_char, alpha, cv):
    """
    Runs one full 5-fold CV. Returns array of fold accuracies.
    Each fold: fit both pipelines on train split, combine scores on val split.
    """
    X = np.array(X, dtype=object)
    y = np.array(y)
    fold_accs = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X[train_idx].tolist(), X[val_idx].tolist()
        y_tr, y_va = y[train_idx], y[val_idx]

        wp = make_word_pipeline(C_word)
        cp = make_char_pipeline(C_char)
        wp.fit(X_tr, y_tr)
        cp.fit(X_tr, y_tr)

        preds = ensemble_predict(wp, cp, X_va, alpha)
        fold_accs.append(accuracy_score(y_va, preds))
    return np.array(fold_accs)

# ── Full grid search ───────────────────────────────────────────────────────────
def run_grid(X, y):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    total = len(C_WORD_GRID) * len(C_CHAR_GRID) * len(ALPHA_GRID)
    print(f"\nFull grid search: {total} combinations ({N_FOLDS}-fold CV each)")
    print(f"{'C_word':<8} {'C_char':<8} {'alpha':<7} {'cv_mean':>8} {'std':>7}")
    print("-" * 44)

    results = []
    best    = {"cv_mean": -1}
    done    = 0

    for C_word, C_char in itertools.product(C_WORD_GRID, C_CHAR_GRID):
        # Fit word and char once per (C_word, C_char) fold combination
        # then sweep alpha over the saved decision scores — 9× cheaper
        X_arr = np.array(X, dtype=object)
        y_arr = np.array(y)

        word_scores_folds = []   # list of (val_idx, scores, y_val) per fold
        char_scores_folds = []

        for train_idx, val_idx in cv.split(X_arr, y_arr):
            X_tr = X_arr[train_idx].tolist()
            X_va = X_arr[val_idx].tolist()
            y_tr = y_arr[train_idx]
            y_va = y_arr[val_idx]

            wp = make_word_pipeline(C_word)
            cp = make_char_pipeline(C_char)
            wp.fit(X_tr, y_tr)
            cp.fit(X_tr, y_tr)

            word_scores_folds.append((wp.decision_function(X_va), y_va))
            char_scores_folds.append((cp.decision_function(X_va), y_va))

        # Now sweep alpha without refitting
        for alpha in ALPHA_GRID:
            fold_accs = []
            for (sw, y_va), (sc, _) in zip(word_scores_folds, char_scores_folds):
                combined = alpha * sw + (1.0 - alpha) * sc
                preds    = (combined > 0).astype(int)
                fold_accs.append(accuracy_score(y_va, preds))

            mean_acc = float(np.mean(fold_accs))
            std_acc  = float(np.std(fold_accs))
            done    += 1

            entry = {
                "C_word": C_word, "C_char": C_char, "alpha": alpha,
                "cv_mean": mean_acc, "cv_std": std_acc,
            }
            results.append(entry)

            marker = " ←" if mean_acc > best["cv_mean"] else ""
            print(f"{C_word:<8} {C_char:<8} {alpha:<7} {mean_acc:>8.4f} {std_acc:>7.4f}{marker}")

            if mean_acc > best["cv_mean"]:
                best = entry.copy()

        print(f"  [{done}/{total} done — C_word={C_word}, C_char={C_char}]")

    return results, best

# ── Hold-out evaluation at best config ────────────────────────────────────────
def holdout_eval(X, y, best):
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    wp = make_word_pipeline(best["C_word"])
    cp = make_char_pipeline(best["C_char"])
    wp.fit(Xtr, ytr)
    cp.fit(Xtr, ytr)

    preds  = ensemble_predict(wp, cp, Xva, best["alpha"])
    acc    = accuracy_score(yva, preds)
    cm     = confusion_matrix(yva, preds)
    report = classification_report(yva, preds, target_names=["Negative", "Positive"])
    print(f"\nHold-out accuracy (best config): {acc:.4f}")
    print(report)
    return acc, cm, report

# ── Figures ────────────────────────────────────────────────────────────────────
def plot_alpha_curve(results, best, path):
    """Alpha vs CV accuracy at the best (C_word, C_char)."""
    sub = [r for r in results
           if r["C_word"] == best["C_word"] and r["C_char"] == best["C_char"]]
    alphas = [r["alpha"]   for r in sub]
    means  = [r["cv_mean"] for r in sub]
    stds   = [r["cv_std"]  for r in sub]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, means, "o-", color="#1a5f9e", lw=2, ms=6,
            label=f"CV acc  (C_word={best['C_word']}, C_char={best['C_char']})")
    ax.fill_between(alphas,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color="#1a5f9e", label="+/- 1 std")
    ax.axvline(best["alpha"], color="#c0392b", lw=1.5, ls="--",
               label=f"Best alpha = {best['alpha']}")
    ax.scatter([best["alpha"]], [best["cv_mean"]],
               color="#c0392b", zorder=5, s=80)
    ax.annotate(f"{best['cv_mean']:.4f}",
                xy=(best["alpha"], best["cv_mean"]),
                xytext=(best["alpha"] + 0.05, best["cv_mean"] - 0.003),
                fontsize=9, color="#c0392b")
    ax.set_xlabel("alpha  (weight on word model)", fontsize=11)
    ax.set_ylabel(f"{N_FOLDS}-fold CV accuracy", fontsize=11)
    ax.set_title("Ensemble — alpha tuning at best (C_word, C_char)", fontsize=12, pad=10)
    ax.set_xticks(ALPHA_GRID)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_c_heatmap(results, best, path):
    """C_word × C_char heatmap of best CV accuracy at best alpha."""
    grid = np.zeros((len(C_CHAR_GRID), len(C_WORD_GRID)))
    for r in results:
        if abs(r["alpha"] - best["alpha"]) < 1e-9:
            i = C_CHAR_GRID.index(r["C_char"])
            j = C_WORD_GRID.index(r["C_word"])
            grid[i, j] = r["cv_mean"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, cmap="Blues", aspect="auto",
                   vmin=grid.min() - 0.002, vmax=grid.max() + 0.001)
    fig.colorbar(im, ax=ax, label="5-fold CV accuracy")
    ax.set_xticks(range(len(C_WORD_GRID)))
    ax.set_yticks(range(len(C_CHAR_GRID)))
    ax.set_xticklabels([str(c) for c in C_WORD_GRID])
    ax.set_yticklabels([str(c) for c in C_CHAR_GRID])
    ax.set_xlabel("C_word", fontsize=11)
    ax.set_ylabel("C_char", fontsize=11)
    ax.set_title(f"Ensemble — C_word x C_char heatmap  (alpha={best['alpha']})",
                 fontsize=11, pad=10)
    ax.grid(False)

    thresh = grid.mean()
    for i in range(len(C_CHAR_GRID)):
        for j in range(len(C_WORD_GRID)):
            ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if grid[i, j] > thresh else "black")

    # Mark best cell
    bi = C_CHAR_GRID.index(best["C_char"])
    bj = C_WORD_GRID.index(best["C_word"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                                fill=False, edgecolor="#c0392b", lw=2))
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
    print(f"  C_word = {best['C_word']}")
    print(f"  C_char = {best['C_char']}")
    print(f"  alpha  = {best['alpha']}")
    print(f"  CV acc = {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}")

    holdout_acc, cm, clf_report = holdout_eval(X, y, best)

    plot_alpha_curve(grid_results, best, OUT_DIR / "alpha_curve.png")
    plot_c_heatmap(grid_results, best,   OUT_DIR / "c_heatmap.png")
    plot_cm(cm,                          OUT_DIR / "confusion_matrix.png")

    results_out = {
        "model":        "TF-IDF word+char ensemble",
        "cv_folds":     N_FOLDS,
        "seed":         SEED,
        "grid":         grid_results,
        "best_C_word":  best["C_word"],
        "best_C_char":  best["C_char"],
        "best_alpha":   best["alpha"],
        "best_cv_acc":  best["cv_mean"],
        "best_cv_std":  best["cv_std"],
        "holdout_acc":  holdout_acc,
        "confusion_matrix": cm.tolist(),
        "elapsed_s":    round(time.time() - t0, 1),
    }
    (OUT_DIR / "scores.json").write_text(json.dumps(results_out, indent=2))

    snippet = (
        f"=== Model 2 - TF-IDF word+char ensemble ===\n"
        f"Word : ngram=(1,2), sublinear_tf, min_df=2  C_word={best['C_word']}\n"
        f"Char : char_wb (3,5), sublinear_tf, min_df=2  C_char={best['C_char']}\n"
        f"Alpha (word weight) = {best['alpha']}\n\n"
        f"5-fold CV  : {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}\n"
        f"Hold-out   : {holdout_acc:.4f}  (80/20, seed={SEED})\n\n"
        f"Confusion matrix (hold-out):\n"
        f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n"
        f"{clf_report}\n"
        f"Figures : alpha_curve.png  c_heatmap.png  confusion_matrix.png\n"
        f"Time    : {round(time.time()-t0,1)}s\n"
    )
    (OUT_DIR / "report_snippet.txt").write_text(snippet)
    print(snippet)
    print(f"All outputs -> {OUT_DIR}")

if __name__ == "__main__":
    main()
