from pathlib import Path
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def load_docs(movies_dir: Path):
    pos = sorted((movies_dir / "pos").glob("*.txt"))
    neg = sorted((movies_dir / "neg").glob("*.txt"))
    X = [read_text(p) for p in pos] + [read_text(n) for n in neg]
    y = np.array([1]*len(pos) + [0]*len(neg))
    return X, y

def train_one(Xtr, Xva, ytr, yva, vec, C):
    model = Pipeline([("tfidf", vec), ("clf", LinearSVC(C=C))])
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)
    return accuracy_score(yva, pred), model

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"
    MODEL_DIR = Path(__file__).resolve().parent

    X, y = load_docs(DATA_DIR)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vec_word = TfidfVectorizer(
        lowercase=True, ngram_range=(1,2),
        min_df=2, max_df=0.9,
        sublinear_tf=True, strip_accents="unicode"
    )
    vec_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,5),
        min_df=2, sublinear_tf=True
    )

    Cs = [0.25, 0.5, 1, 2, 4]
    best = (-1, None, None)

    for C in Cs:
        acc, m = train_one(Xtr, Xva, ytr, yva, vec_word, C)
        print(f"WORD  C={C}: acc={acc:.4f}")
        if acc > best[0]: best = (acc, f"word_C{C}", m)

    for C in Cs:
        acc, m = train_one(Xtr, Xva, ytr, yva, vec_char, C)
        print(f"CHAR  C={C}: acc={acc:.4f}")
        if acc > best[0]: best = (acc, f"char_C{C}", m)

    best_acc, best_name, best_model = best
    print("\nBEST:", best_name, "acc=", best_acc)

    out = MODEL_DIR / "best_tfidf_linearsvc.joblib"
    joblib.dump(best_model, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()