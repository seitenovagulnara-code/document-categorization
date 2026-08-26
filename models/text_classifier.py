"""
Классификаторы документов.

Две модели-соперника на ОДНУ задачу (классификация по 18 сценариям):
  - BaselineClassifier    — TF-IDF + LogisticRegression (этап 4, точка отсчёта).
  - TransformerClassifier — дообученный DistilBERT (этап 5, идёт в пайплайн).

BERT обязан обогнать бейзлайн на ≥5% и взять accuracy ≥85%, F1-macro ≥0.80.
Метрики считаем ОТДЕЛЬНО по языкам (условие: ≥80% на каждый язык).
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.text_preprocessing import classic_preprocess

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
BASELINE_PATH = CHECKPOINT_DIR / "baseline_tfidf_logreg.joblib"


class BaselineClassifier:
    """
    TF-IDF + LogisticRegression. Быстрый, интерпретируемый, считается на CPU.
    Лёгкий препроцессинг (очистка + нижний регистр) применяется к текстам
    внутри методов, поэтому сохранённый объект — чистый sklearn-пайплайн
    (портируется без кастомных функций).
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),     # униграммы + биграммы (полезно на коротких текстах)
                min_df=2,               # выбросить сверхредкие токены
                sublinear_tf=True,      # сгладить частоты
            )),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    @staticmethod
    def _prep(texts):
        return [classic_preprocess(t) for t in texts]

    def fit(self, texts, labels):
        self.pipeline.fit(self._prep(list(texts)), list(labels))
        return self

    def predict(self, texts):
        return self.pipeline.predict(self._prep(list(texts)))

    def save(self, path=BASELINE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        return path

    def load(self, path=BASELINE_PATH):
        self.pipeline = joblib.load(path)
        return self


def train_baseline(splits):
    """Обучить бейзлайн на train (en+ru вместе). splits — из get_train_val_test()."""
    tr = splits["train"]
    model = BaselineClassifier()
    model.fit(tr["text"], tr["label_name"])
    return model


def evaluate(model, splits, languages=("en", "ru"), report=False):
    """
    Посчитать accuracy и F1-macro ОТДЕЛЬНО по языкам на test-выборках.
    Возвращает dict: {"en": {...}, "ru": {...}, "overall": {...}}.
    report=True — дополнительно печатает classification_report по каждому языку.
    """
    out = {}
    all_true, all_pred = [], []
    for lang in languages:
        te = splits["test"][lang]
        y_true = list(te["label_name"])
        y_pred = list(model.predict(te["text"]))
        all_true += y_true
        all_pred += y_pred
        out[lang] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "n": len(te),
        }
        if report:
            print(f"\n=== {lang} ===")
            print(classification_report(y_true, y_pred, zero_division=0))

    out["overall"] = {
        "accuracy": round(accuracy_score(all_true, all_pred), 4),
        "f1_macro": round(f1_score(all_true, all_pred, average="macro"), 4),
        "n": len(all_true),
    }
    return out


class TransformerClassifier:
    """Дообученная многоязычная DistilBERT. Основной классификатор (этап 5)."""

    def __init__(self, model_dir=None):
        self.model = None
        self.tokenizer = None
        self.model_dir = model_dir

    def load(self, model_dir=None):
        # TODO (этап 5/9): загрузить веса и токенизатор из checkpoints/
        raise NotImplementedError

    def predict(self, texts):
        # TODO (этап 5)
        raise NotImplementedError

    def predict_batch(self, texts, batch_size=32):
        # TODO (этап 8): батчинг для ≥100 док/сек
        raise NotImplementedError
        