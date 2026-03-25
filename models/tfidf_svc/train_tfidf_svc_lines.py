from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")


def file_to_lines(text: str):
    # Use non-empty lines as samples
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines


def load_files(movies_dir: Path):
    pos_files = sorted((movies_dir / "pos").glob("*.txt"))
    neg_files = sorted((movies_dir / "neg").glob("*.txt"))
    if not pos_files or not neg_files:
        raise FileNotFoundError("Missing pos/neg files")

    files = pos_files + neg_files
    labels = [1] * len(pos_files) + [0] * len(neg_files)
    return files, labels


def expand_to_lines(files, labels):
    X_lines, y_lines = [], []
    for fp, y in zip(files, labels):
        text = read_text(fp)
        lines = file_to_lines(text)
        # if a file somehow has no lines, skip it
        for ln in lines:
            X_lines.append(ln)
            y_lines.append(y)
    return X_lines, y_lines


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"
    MODEL_DIR = Path(__file__).resolve().parent
    OUT_DIR = MODEL_DIR / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files, labels = load_files(DATA_DIR)

    # Split by FILE (prevents leakage)
    f_train, f_val, y_train_f, y_val_f = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Expand each side into line-level samples
    X_train, y_train = expand_to_lines(f_train, y_train_f)
    X_val, y_val = expand_to_lines(f_val, y_val_f)

    print(f"Train files: {len(f_train)} -> train lines: {len(X_train)}")
    print(f"Val files:   {len(f_val)} -> val lines:   {len(X_val)}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("clf", LinearSVC(C=1.0))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    print("Validation accuracy:", acc)
    print(classification_report(y_val, preds, target_names=["neg", "pos"]))

    out = MODEL_DIR / "tfidf_linearsvc_lines.joblib"
    joblib.dump(model, out)
    print("Saved:", out)


if __name__ == "__main__":
    main()