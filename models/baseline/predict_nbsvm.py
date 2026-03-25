from pathlib import Path
import joblib
import numpy as np

def load_lines_wc_like(fp: Path):
    text = fp.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEST_FILE = PROJECT_ROOT / "dataset" / "testSentiment.txt"

    MODEL_DIR = Path(__file__).resolve().parent
    OUT_DIR = MODEL_DIR / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    obj = joblib.load(MODEL_DIR / "nbsvm.joblib")
    nb, clf = obj["nb"], obj["clf"]

    X = load_lines_wc_like(TEST_FILE)
    X_nb = nb.transform(X)
    pred = clf.predict(X_nb)

    out_path = OUT_DIR / "preds.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for p in pred:
            f.write(("P" if int(p) == 1 else "N") + "\n")

    print("Wrote:", out_path, "Total:", len(pred))

if __name__ == "__main__":
    main()