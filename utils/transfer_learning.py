"""
Этап 5 — Transfer learning: дообучение DistilBERT.

Многоязычная DistilBERT дообучается на английском train (кросс-язычный перенос
на русский). Обучение тяжёлое — запускать в Google Colab с GPU.
Чекпоинты сохраняются после каждой эпохи в models/checkpoints/.

Заготовка: сигнатуры и параметры из условия. Логику наполняем на этапе 5.
"""

from pathlib import Path

MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 256          # обрезка длины (скорость; attention квадратичен по длине)
LEARNING_RATE = 3e-5      # условие: 2e-5 … 5e-5
EPOCHS = 5                # условие: минимум 5 эпох

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"


def get_tokenizer(model_name: str = MODEL_NAME):
    """Загрузить токенизатор DistilBERT."""
    # TODO (этап 5): AutoTokenizer.from_pretrained(model_name)
    raise NotImplementedError


def build_classifier(num_labels: int):
    """Собрать TF-модель DistilBERT + голову классификации на num_labels классов."""
    # TODO (этап 5): TFAutoModelForSequenceClassification.from_pretrained(...)
    raise NotImplementedError


def tokenize_dataset(texts, tokenizer):
    """Токенизировать тексты (subword) с обрезкой до MAX_LENGTH."""
    # TODO (этап 5): реализовать
    raise NotImplementedError


def train(model, train_ds, val_ds, epochs: int = EPOCHS):
    """
    Дообучить модель; сохранять чекпоинт после каждой эпохи в CHECKPOINT_DIR;
    следить за validation loss (ранняя остановка при переобучении);
    вернуть историю обучения (для training_history.csv).
    """
    # TODO (этап 5): реализовать цикл обучения + колбэки
    raise NotImplementedError
