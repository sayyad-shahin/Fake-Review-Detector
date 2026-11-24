"""Training, saving, loading and explaining the fake review model."""
import csv, json, os
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer  # noqa: F401  (kept for easy extension)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import FEATURE_NAMES, red_flags, style_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "data", "reviews.csv")
DEFAULT_MODEL = os.path.join(ROOT, "models", "detector.joblib")
MAX_CHARS = 5000

TEXT_COLUMNS = ("text", "review", "review_text", "text_", "content", "reviewtext")
LABEL_COLUMNS = ("label", "class", "target", "is_fake", "fake")
FAKE_VALUES = {"1", "fake", "cg", "spam", "deceptive", "fraud", "true", "yes"}
REAL_VALUES = {"0", "genuine", "or", "real", "truthful", "original", "false", "no"}


def load_dataset(path):
    """Read a CSV with a review-text column and a label column (1/fake or 0/genuine)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        tcol = next((cols[c] for c in TEXT_COLUMNS if c in cols), None)
        lcol = next((cols[c] for c in LABEL_COLUMNS if c in cols), None)
        if not tcol or not lcol:
            raise ValueError(f"Could not find text and label columns in {path}. Found: {list(cols)}. "
                             f"Expected a text column ({TEXT_COLUMNS}) and a label column ({LABEL_COLUMNS}).")
        texts, labels = [], []
        for row in reader:
            text, lab = (row[tcol] or "").strip(), str(row[lcol]).strip().lower()
            if not text:
                continue
            if lab in FAKE_VALUES:
                labels.append(1)
            elif lab in REAL_VALUES:
                labels.append(0)
            else:
                raise ValueError(f"Unknown label value '{row[lcol]}'. Use 1/fake or 0/genuine.")
            texts.append(text)
    if len(set(labels)) < 2:
        raise ValueError("The dataset needs both fake and genuine examples.")
    return texts, np.array(labels)


def build_pipeline(min_df=2):
    union = FeatureUnion([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, max_features=30000)),
        ("style", Pipeline([("raw", FunctionTransformer(style_features)), ("scale", StandardScaler())])),
    ])
    return Pipeline([("features", union), ("clf", LogisticRegression(C=2.0, max_iter=3000))])


def train_model(data_path=DEFAULT_DATA, model_path=DEFAULT_MODEL, test_size=0.2, seed=42, verbose=True):
    """Train, evaluate on a held-out test set, save the model and a metrics file."""
    texts, y = load_dataset(data_path)
    X_tr, X_te, y_tr, y_te = train_test_split(texts, y, test_size=test_size, random_state=seed, stratify=y)
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    cm = confusion_matrix(y_te, pred)
    metrics = {
        "samples": len(texts), "train_size": len(X_tr), "test_size": len(X_te),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision_fake": round(float(precision_score(y_te, pred)), 4),
        "recall_fake": round(float(recall_score(y_te, pred)), 4),
        "f1_fake": round(float(f1_score(y_te, pred)), 4),
        "confusion_matrix": {"true_genuine_pred_genuine": int(cm[0, 0]), "true_genuine_pred_fake": int(cm[0, 1]),
                             "true_fake_pred_genuine": int(cm[1, 0]), "true_fake_pred_fake": int(cm[1, 1])},
    }
    cv = cross_val_score(build_pipeline(), texts, y, cv=5, scoring="f1")
    metrics["cv_f1_mean"], metrics["cv_f1_std"] = round(float(cv.mean()), 4), round(float(cv.std()), 4)

    pipe.fit(texts, y)  # final model learns from all data
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(pipe, model_path)
    with open(os.path.join(os.path.dirname(model_path), "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"Reviews: {metrics['samples']} (train {metrics['train_size']}, test {metrics['test_size']})")
        print(f"Accuracy {metrics['accuracy']:.3f} | Precision {metrics['precision_fake']:.3f} | "
              f"Recall {metrics['recall_fake']:.3f} | F1 {metrics['f1_fake']:.3f}")
        print(f"5-fold cross-validated F1: {metrics['cv_f1_mean']:.3f} (+/- {metrics['cv_f1_std']:.3f})")
        print("Confusion matrix (rows = truth, columns = prediction):")
        print("                 pred genuine   pred fake")
        print(f"  true genuine   {cm[0, 0]:12d} {cm[0, 1]:11d}")
        print(f"  true fake      {cm[1, 0]:12d} {cm[1, 1]:11d}")
        print(f"Model saved to {model_path}")
    return metrics


class Detector:
    """Wraps the trained pipeline and explains its decisions."""

    def __init__(self, pipeline):
        self.pipe = pipeline
        union = pipeline.named_steps["features"]
        self.vec = union.transformer_list[0][1]
        self.names = list(self.vec.get_feature_names_out()) + [f"style: {n}" for n in FEATURE_NAMES]
        self.n_words = len(self.names) - len(FEATURE_NAMES)

    @classmethod
    def load(cls, model_path=DEFAULT_MODEL):
        return cls(joblib.load(model_path))

    @staticmethod
    def verdict(p_fake):
        if p_fake >= 0.65:
            return "Likely fake"
        if p_fake <= 0.35:
            return "Likely genuine"
        return "Uncertain"

    def analyze(self, text, top=6, min_weight=0.12):
        text = (text or "").strip()
        if not text:
            raise ValueError("Please enter a review to analyze.")
        text = text[:MAX_CHARS]
        p_fake = float(self.pipe.predict_proba([text])[0][1])
        x = self.pipe.named_steps["features"].transform([text])
        contrib = np.asarray(x.multiply(self.pipe.named_steps["clf"].coef_[0]).todense()).ravel()
        order = np.argsort(contrib)
        words = [(self.names[i], float(contrib[i])) for i in range(self.n_words) if contrib[i] != 0]
        words.sort(key=lambda t: t[1])
        flags, good = red_flags(text)
        return {
            "verdict": self.verdict(p_fake),
            "fake_probability": round(p_fake, 4),
            "genuine_probability": round(1 - p_fake, 4),
            "warning_signs": flags,
            "trust_signs": good,
            "words_pointing_to_fake": [w for w, c in reversed(words[-top:]) if c >= min_weight],
            "words_pointing_to_genuine": [w for w, c in words[:top] if c <= -min_weight],
        }

    def analyze_many(self, texts):
        probs = self.pipe.predict_proba([str(t)[:MAX_CHARS] for t in texts])[:, 1]
        return [(round(float(p), 4), self.verdict(p)) for p in probs]


def load_or_train(model_path=DEFAULT_MODEL, data_path=DEFAULT_DATA):
    """Load the saved model. If it is missing or was saved by an incompatible scikit-learn, retrain."""
    try:
        return Detector.load(model_path)
    except Exception as e:  # noqa: BLE001
        print(f"[info] Could not load saved model ({type(e).__name__}); training a new one from {data_path} ...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No model and no dataset found. Run: python generate_dataset.py") from e
        train_model(data_path, model_path, verbose=False)
        return Detector.load(model_path)
