"""
Классификаторы документов.

Две модели-соперника на ОДНУ задачу (классификация по темам):
  - BaselineClassifier    — TF-IDF + LogisticRegression (этап 4, точка отсчёта).
  - TransformerClassifier — дообученный DistilBERT (этап 5, идёт в пайплайн).

BERT обязан обогнать бейзлайн на ≥5% и взять accuracy ≥85%, F1-macro ≥0.80.

Заготовка: интерфейсы классов. Логику наполняем на этапах 4-5.
"""


class BaselineClassifier:
    """TF-IDF + LogisticRegression. Быстрый, интерпретируемый, считается на CPU."""

    def __init__(self):
        self.vectorizer = None   # TfidfVectorizer
        self.model = None        # LogisticRegression

    def fit(self, texts, labels):
        # TODO (этап 4): обучить TF-IDF + LogReg
        raise NotImplementedError

    def predict(self, texts):
        # TODO (этап 4)
        raise NotImplementedError

    def save(self, path):
        # TODO (этап 4): joblib.dump
        raise NotImplementedError

    def load(self, path):
        # TODO (этап 4): joblib.load
        raise NotImplementedError


class TransformerClassifier:
    """Дообученная многоязычная DistilBERT. Основной классификатор проекта."""

    def __init__(self, model_dir=None):
        self.model = None
        self.tokenizer = None
        self.model_dir = model_dir   # models/checkpoints/

    def load(self, model_dir=None):
        # TODO (этап 5/9): загрузить веса и токенизатор из checkpoints/
        raise NotImplementedError

    def predict(self, texts):
        """Вернуть категорию (+ уверенность) для одного/нескольких документов."""
        # TODO (этап 5): токенизация -> forward -> softmax -> argmax
        raise NotImplementedError

    def predict_batch(self, texts, batch_size=32):
        """Батчевое предсказание для пайплайна реального времени (этап 8)."""
        # TODO (этап 8): батчинг для скорости ≥100 док/сек
        raise NotImplementedError
