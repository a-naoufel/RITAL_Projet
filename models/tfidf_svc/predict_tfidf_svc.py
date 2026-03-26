"""
models/tfidf_svc/predict_tfidf_svc.py
======================================
Generates the submission file for Model 1 (TF-IDF word + LinearSVC).

Protocol
--------
- Trains on the FULL 2000 labelled reviews (no hold-out — we already have
  CV estimates from the benchmark, so we use all available signal here).
- Best C=4.0 confirmed by 5-fold CV in train_tfidf_svc_benchmark.py.
- Reads testSentiment.txt line by line (one review per line).
- Writes one label per line: P (positive) or N (negative).

Output
------
    submissions/preds_model1_tfidf_svc.txt
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# ── Config — must match benchmark ─────────────────────────────────────────────
SEED   = 42
BEST_C = 4.0

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"
TEST_FILE    = PROJECT_ROOT / "dataset" / "raw" / "testSentiment.txt"
OUT_DIR      = PROJECT_ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE     = OUT_DIR / "preds_model1_tfidf_svc.txt"

# ── Data loaders ───────────────────────────────────────────────────────────────
def load_train(data_dir: Path):
    pos = sorted((data_dir / "pos").glob("*.txt"))
    neg = sorted((data_dir / "neg").glob("*.txt"))
    if not pos:
        raise FileNotFoundError(f"No reviews found in {data_dir}/pos/")
    texts  = [p.read_text(encoding="utf-8", errors="ignore") for p in pos]
    texts += [p.read_text(encoding="utf-8", errors="ignore") for p in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"Training data : {len(pos)} pos + {len(neg)} neg = {len(texts)} reviews")
    return texts, labels


def load_test(test_file: Path):
    """
    One review per line. Matches wc -l: split on \n, drop trailing empty line.
    Empty lines (blank reviews) are kept — TF-IDF returns all-zeros → valid input.
    """
    text  = test_file.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    print(f"Test file     : {test_file.name}  ({len(lines)} lines)")
    return lines

# ── Pipeline — identical to benchmark ─────────────────────────────────────────
def make_pipeline(C: float) -> Pipeline:
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

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Load training data
    X_train, y_train = load_train(DATA_DIR)

    # 2. Fit on full training set (best C from CV)
    print(f"Fitting pipeline  C={BEST_C} on full training set...")
    pipe = make_pipeline(BEST_C)
    pipe.fit(X_train, y_train)
    print("Fit complete.")

    # 3. Load test lines
    test_lines = load_test(TEST_FILE)

    # 4. Predict in one batch (LinearSVC on sparse TF-IDF is fast)
    preds = pipe.predict(test_lines)

    # 5. Write submission file
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(("P" if int(p) == 1 else "N") + "\n")

    # 6. Sanity checks
    n_pos = int(np.sum(preds == 1))
    n_neg = int(np.sum(preds == 0))
    print(f"\nPredictions written : {OUT_FILE.resolve()}")
    print(f"Total lines         : {len(test_lines)}")
    print(f"Total predictions   : {len(preds)}")
    print(f"Positive (P)        : {n_pos}  ({100*n_pos/len(preds):.1f}%)")
    print(f"Negative (N)        : {n_neg}  ({100*n_neg/len(preds):.1f}%)")

    # Check line count matches
    written = sum(1 for _ in OUT_FILE.open())
    assert written == len(test_lines), \
        f"Line count mismatch: wrote {written}, expected {len(test_lines)}"
    print(f"Line count check    : OK ({written} lines)")

if __name__ == "__main__":
    main()
