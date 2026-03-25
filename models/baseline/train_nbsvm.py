from pathlib import Path
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

class NBLogCountRatio(BaseEstimator, TransformerMixin):
    """
    NB-SVM feature transform:
    r_j = log( (count_pos_j + alpha) / (count_neg_j + alpha) )
    X_nb = X * r
    """
    def __init__(self, alpha=1.0, ngram_range=(1,2), min_df=2, max_features=None):
        self.alpha = alpha
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features

    def fit(self, X, y):
        self.vect_ = CountVectorizer(
            binary=True,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features
        )
        Xc = self.vect_.fit_transform(X)
        y = np.asarray(y)

        # sums are 1 x V sparse matrices
        pos = Xc[y == 1].sum(axis=0) + self.alpha
        neg = Xc[y == 0].sum(axis=0) + self.alpha

        r = np.log(pos / neg)
        self.r_ = np.asarray(r).ravel()  # shape (V,)
        return self

    def transform(self, X):
        Xc = self.vect_.transform(X)
        return Xc.multiply(self.r_)  # column-wise scaling


def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def load_docs(movies_dir: Path):
    pos = sorted((movies_dir / "pos").glob("*.txt"))
    neg = sorted((movies_dir / "neg").glob("*.txt"))
    X = [read_text(p) for p in pos] + [read_text(n) for n in neg]
    y = [1]*len(pos) + [0]*len(neg)
    return X, y

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"
    MODEL_DIR = Path(__file__).resolve().parent

    X, y = load_docs(DATA_DIR)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # small grid (you can extend)
    alphas = [0.5, 1.0, 2.0]
    Cs = [0.5, 1, 2, 4]

    best = (-1, None)
    best_obj = None

    for a in alphas:
        nb = NBLogCountRatio(alpha=a, ngram_range=(1,2), min_df=2)
        Xtr_nb = nb.fit_transform(Xtr, ytr)
        Xva_nb = nb.transform(Xva)

        for C in Cs:
            clf = LinearSVC(C=C)
            clf.fit(Xtr_nb, ytr)
            pred = clf.predict(Xva_nb)
            acc = accuracy_score(yva, pred)
            print(f"NB-SVM alpha={a} C={C} acc={acc:.4f}")
            if acc > best[0]:
                best = (acc, (a, C))
                best_obj = {"nb": nb, "clf": clf}

    best_acc, (best_a, best_C) = best
    print("\nBEST NB-SVM:", f"alpha={best_a} C={best_C}", "acc=", best_acc)

    # report
    Xva_nb = best_obj["nb"].transform(Xva)
    pred = best_obj["clf"].predict(Xva_nb)
    print(classification_report(yva, pred, target_names=["neg", "pos"]))

    out = MODEL_DIR / "nbsvm.joblib"
    joblib.dump(best_obj, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()