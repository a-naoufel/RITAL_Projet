"""
models/setfit/predict_setfit.py
=================================
Generates submission for Model 4 (SetFit, full-data regime).
Trains on full 2000 reviews, predicts testSentiment.txt.
Reads CV accuracy from scores.json for logging.

Output  →  submissions/preds_model4_setfit.txt
"""

from pathlib import Path
import json, warnings
import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

warnings.filterwarnings("ignore")

SEED       = 42
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"
TEST_FILE    = PROJECT_ROOT / "dataset" / "raw" / "testSentiment.txt"
SCORES_JSON  = PROJECT_ROOT / "results" / "model4_setfit" / "scores.json"
OUT_DIR      = PROJECT_ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE     = OUT_DIR / "preds_model4_setfit.txt"


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


def main():
    if SCORES_JSON.exists():
        cfg = json.loads(SCORES_JSON.read_text())
        print(f"Full CV acc : {cfg['full_cv_mean']:.4f} +/- {cfg['full_cv_std']:.4f}")

    X_train, y_train = load_train(DATA_DIR)
    test_lines       = load_test(TEST_FILE)

    print(f"\nFitting SetFit on full {len(X_train)} reviews...")
    model    = SetFitModel.from_pretrained(MODEL_NAME)
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})

    args = TrainingArguments(
        num_epochs          = 1,
        batch_size          = 16,
        num_iterations      = 20,
        evaluation_strategy = "no",
        seed                = SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    print("Predicting test set...")
    preds  = np.array(model.predict(test_lines))
    n_pos  = int(np.sum(preds == 1))
    n_neg  = int(np.sum(preds == 0))

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(("P" if p == 1 else "N") + "\n")

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
