from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")


def load_dataset(movies_dir: Path):
    pos_dir = movies_dir / "pos"
    neg_dir = movies_dir / "neg"

    pos_files = sorted(pos_dir.glob("*.txt"))
    neg_files = sorted(neg_dir.glob("*.txt"))

    if not pos_files or not neg_files:
        raise FileNotFoundError(f"Expected txt files in {pos_dir} and {neg_dir}")

    X = [read_text(p) for p in pos_files] + [read_text(n) for n in neg_files]
    y = [1] * len(pos_files) + [0] * len(neg_files)
    return X, y


def main():
    # project root = .../project (based on your paths)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"

    MODEL_DIR = Path(__file__).resolve().parent
    OUT_DIR = MODEL_DIR / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(DATA_DIR)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Strong default sentiment baseline:
    # word ngrams (1,2) + LinearSVC
    from sklearn.pipeline import FeatureUnion

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("clf", LinearSVC(
            C=1.0,
            random_state=42,
            tol=1e-5,
            dual=False
        ))
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print("Validation accuracy:", acc)
    print(classification_report(y_val, preds, target_names=["neg", "pos"]))

    model_path = MODEL_DIR / "tfidf_linearsvc.joblib"
    joblib.dump(model, model_path)
    print("Saved model to:", model_path)


if __name__ == "__main__":
    main()