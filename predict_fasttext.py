"""
models/fasttext/predict_fasttext.py
=====================================
Generates submission file for Model 3 (fastText supervised).
Reads best config from results/model3_fasttext/scores.json.
Trains on full 2000 reviews, predicts testSentiment.txt.

Output  →  submissions/preds_model3_fasttext.txt
"""

from pathlib import Path
import json, tempfile
import numpy as np
import fasttext

SEED = 42

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"
TEST_FILE    = PROJECT_ROOT / "dataset" / "raw" / "testSentiment.txt"
SCORES_JSON  = PROJECT_ROOT / "results" / "model3_fasttext" / "scores.json"
OUT_DIR      = PROJECT_ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE     = OUT_DIR / "preds_model3_fasttext.txt"


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


def write_fasttext_file(texts, labels, path):
    with path.open("w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            ft_label = "__label__pos" if label == 1 else "__label__neg"
            clean    = text.replace("\n", " ").replace("\r", " ").strip()
            f.write(f"{ft_label} {clean}\n")


def main():
    if not SCORES_JSON.exists():
        raise FileNotFoundError(
            f"scores.json not found at {SCORES_JSON}\n"
            "Run train_fasttext_benchmark.py first."
        )
    cfg = json.loads(SCORES_JSON.read_text())
    lr      = cfg["best_lr"]
    epoch   = cfg["best_epoch"]
    wng     = cfg["best_ngrams"]
    dim     = cfg["best_dim"]
    print(f"Config  : lr={lr}, epoch={epoch}, wordNgrams={wng}, dim={dim}")
    print(f"CV acc  : {cfg['best_cv_acc']:.4f} +/- {cfg['best_cv_std']:.4f}")

    X_train, y_train = load_train(DATA_DIR)
    test_lines       = load_test(TEST_FILE)

    # fastText requires a file — use a temp file, train, then delete
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "train.txt"
        write_fasttext_file(X_train, y_train, train_file)

        print("\nFitting fastText on full training set...")
        model = fasttext.train_supervised(
            input      = str(train_file),
            lr         = lr,
            epoch      = epoch,
            wordNgrams = wng,
            dim        = dim,
            loss       = "softmax",
            seed       = SEED,
            verbose    = 1,
        )

        print("Predicting test set...")
        preds = []
        for text in test_lines:
            clean = text.replace("\n", " ").replace("\r", " ").strip()
            label, _ = model.predict(clean)
            preds.append(1 if label[0] == "__label__pos" else 0)

    preds = np.array(preds)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(("P" if p == 1 else "N") + "\n")

    n_pos = int(np.sum(preds == 1))
    n_neg = int(np.sum(preds == 0))
    print(f"\nWritten      : {OUT_FILE.resolve()}")
    print(f"Total        : {len(preds)}")
    print(f"Positive (P) : {n_pos}  ({100*n_pos/len(preds):.1f}%)")
    print(f"Negative (N) : {n_neg}  ({100*n_neg/len(preds):.1f}%)")

    written = sum(1 for _ in OUT_FILE.open())
    assert written == len(test_lines), \
        f"Line count mismatch: wrote {written}, expected {len(test_lines)}"
    print(f"Line count   : OK ({written} lines)")


if __name__ == "__main__":
    main()
