from pathlib import Path
import joblib

def load_lines_wc_like(fp: Path):
    """
    Load lines in a way that matches wc -l expectation:
    - split only on '\n'
    - keep empty lines if they exist (still count as a line)
    - if the file ends with a trailing newline, drop the final empty element
    """
    text = fp.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    if len(lines) > 0 and lines[-1] == "":
        lines = lines[:-1]
    return lines

def main():
    # project root = .../RITAL (two levels above models/baseline/)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEST_FILE = PROJECT_ROOT / "dataset" / "testSentiment.txt"

    MODEL_DIR = Path(__file__).resolve().parent
    OUT_DIR = MODEL_DIR / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "tfidf_linearsvc.joblib"
    model = joblib.load(model_path)

    lines = load_lines_wc_like(TEST_FILE)

    # Predict even for empty lines (if any). For empty strings, TF-IDF returns all-zeros;
    preds = model.predict(lines)

    out_path = OUT_DIR / "preds.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(("P" if int(p) == 1 else "N") + "\n")

    print("TEST_FILE:", TEST_FILE.resolve())
    print("Wrote:", out_path.resolve())
    print("Total lines:", len(lines))
    print("Total predictions:", len(preds))

if __name__ == "__main__":
    main()