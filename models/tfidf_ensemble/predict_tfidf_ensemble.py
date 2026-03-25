"""
models/tfidf_ensemble/predict_tfidf_ensemble.py
================================================
Generates the submission file for Model 2 (word+char ensemble).
Reads best config from results/model2_tfidf_ensemble/scores.json
so the hyperparameters are always in sync with the benchmark run.

Output  →  submissions/preds_model2_tfidf_ensemble.txt
"""

from pathlib import Path
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

SEED = 42

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"
TEST_FILE    = PROJECT_ROOT / "dataset" / "raw" / "testSentiment.txt"
SCORES_JSON  = PROJECT_ROOT / "results" / "model2_tfidf_ensemble" / "scores.json"
OUT_DIR      = PROJECT_ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE     = OUT_DIR / "preds_model2_tfidf_ensemble.txt"


def load_train(data_dir):
    pos = sorted((data_dir / "pos").glob("*.txt"))
    neg = sorted((data_dir / "neg").glob("*.txt"))
    texts  = [p.read_text(encoding="utf-8", errors="ignore") for p in pos]
    texts += [p.read_text(encoding="utf-8", errors="ignore") for p in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"Training data : {len(pos)} pos + {len(neg)} neg")
    return texts, labels


def load_test(test_file):
    text  = test_file.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    print(f"Test file     : {test_file.name}  ({len(lines)} lines)")
    return lines


def make_word_pipeline(C):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, max_df=0.95, strip_accents="unicode",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )),
        ("clf", LinearSVC(C=C, max_iter=2000, random_state=SEED)),
    ])


def make_char_pipeline(C):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
            min_df=2, max_df=0.95,
        )),
        ("clf", LinearSVC(C=C, max_iter=2000, random_state=SEED)),
    ])


def main():
    # Load best hyperparameters from benchmark run
    if not SCORES_JSON.exists():
        raise FileNotFoundError(
            f"scores.json not found at {SCORES_JSON}\n"
            f"Run train_tfidf_ensemble_benchmark.py first."
        )
    cfg = json.loads(SCORES_JSON.read_text())
    C_word = cfg["best_C_word"]
    C_char = cfg["best_C_char"]
    alpha  = cfg["best_alpha"]
    print(f"Config loaded : C_word={C_word}, C_char={C_char}, alpha={alpha}")
    print(f"CV accuracy   : {cfg['best_cv_acc']:.4f} +/- {cfg['best_cv_std']:.4f}")

    X_train, y_train = load_train(DATA_DIR)
    test_lines       = load_test(TEST_FILE)

    print(f"\nFitting word pipeline  (C={C_word})...")
    wp = make_word_pipeline(C_word)
    wp.fit(X_train, y_train)

    print(f"Fitting char pipeline  (C={C_char})...")
    cp = make_char_pipeline(C_char)
    cp.fit(X_train, y_train)

    print("Predicting...")
    score_word = wp.decision_function(test_lines)
    score_char = cp.decision_function(test_lines)
    combined   = alpha * score_word + (1.0 - alpha) * score_char
    preds      = (combined > 0).astype(int)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(("P" if p == 1 else "N") + "\n")

    n_pos = int(np.sum(preds == 1))
    n_neg = int(np.sum(preds == 0))
    print(f"\nWritten       : {OUT_FILE.resolve()}")
    print(f"Total         : {len(preds)}")
    print(f"Positive (P)  : {n_pos}  ({100*n_pos/len(preds):.1f}%)")
    print(f"Negative (N)  : {n_neg}  ({100*n_neg/len(preds):.1f}%)")

    written = sum(1 for _ in OUT_FILE.open())
    assert written == len(test_lines), \
        f"Line count mismatch: wrote {written}, expected {len(test_lines)}"
    print(f"Line count    : OK ({written} lines)")


if __name__ == "__main__":
    main()
