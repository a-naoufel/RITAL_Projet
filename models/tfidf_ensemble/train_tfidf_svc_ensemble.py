from pathlib import Path
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")


def load_docs(movies_dir: Path):
    pos = sorted((movies_dir / "pos").glob("*.txt"))
    neg = sorted((movies_dir / "neg").glob("*.txt"))
    X = [read_text(p) for p in pos] + [read_text(n) for n in neg]
    y = np.array([1] * len(pos) + [0] * len(neg))
    return X, y


class EnsembleSVC:
    """
    Simple ensemble wrapper: average decision_function scores.
    """
    def __init__(self, model_word, model_char, alpha=0.5):
        self.model_word = model_word
        self.model_char = model_char
        self.alpha = alpha  # weight on word model

    def decision_function(self, X):
        s_word = self.model_word.decision_function(X)
        s_char = self.model_char.decision_function(X)
        return self.alpha * s_word + (1 - self.alpha) * s_char

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"
    MODEL_DIR = Path(__file__).resolve().parent

    X, y = load_docs(DATA_DIR)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Word model (best from your run)
    word_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2),
            min_df=2, max_df=0.9,
            sublinear_tf=True, strip_accents="unicode"
        )),
        ("clf", LinearSVC(C=4))
    ])

    # Char model (even if weaker alone, can help as ensemble)
    char_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, sublinear_tf=True
        )),
        ("clf", LinearSVC(C=0.5))
    ])

    word_model.fit(Xtr, ytr)
    char_model.fit(Xtr, ytr)

    best = (-1, None)
    for alpha in [0.9, 0.8, 0.7, 0.6, 0.5]:
        ens = EnsembleSVC(word_model, char_model, alpha=alpha)
        pred = ens.predict(Xva)
        acc = accuracy_score(yva, pred)
        print(f"alpha={alpha:.1f} acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, alpha)

    best_acc, best_alpha = best
    print("\nBEST ensemble acc:", best_acc, "alpha:", best_alpha)

    # Evaluate + save best ensemble
    ens = EnsembleSVC(word_model, char_model, alpha=best_alpha)
    pred = ens.predict(Xva)
    print(classification_report(yva, pred, target_names=["neg", "pos"]))

    out = MODEL_DIR / "ensemble_word_char.joblib"
    joblib.dump({"word": word_model, "char": char_model, "alpha": best_alpha}, out)
    print("Saved:", out)


if __name__ == "__main__":
    main()