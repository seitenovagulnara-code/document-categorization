"""
Этап 5 — Transfer learning: дообучение DistilBERT (TensorFlow/Keras).

Многоязычная distilbert-base-multilingual-cased дообучается на train (en+ru).
Обучение тяжёлое — запускать в Google Colab с GPU.

Стратегия (решение: «надёжно»):
  - 5 эпох, learning rate 3e-5, max_length=64;
  - следим за validation loss, сохраняем ЛУЧШИЙ чекпоинт (save_best_only);
  - плюс сохраняем веса после каждой эпохи (требование условия);
  - история обучения пишется в training_history.csv.

Артефакты (в models/checkpoints/):
  text_classifier_best.h5   — лучшие веса,
  config.json               — конфиг модели,
  training_history.csv      — метрики по эпохам,
  + токенизатор (для инференса).

TF/transformers импортируются ВНУТРИ функций, чтобы модуль грузился без них
(например, при импорте из дашборда до установки TF).
"""

from pathlib import Path

import numpy as np

MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 64           # по EDA: медиана 6 слов, max 61 -> 64 с запасом и быстро
LEARNING_RATE = 3e-5      # условие: 2e-5 … 5e-5
EPOCHS = 5                # условие: минимум 5 эпох
BATCH_SIZE = 32

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
BEST_PATH = CHECKPOINT_DIR / "text_classifier_best.h5"
CONFIG_JSON = CHECKPOINT_DIR / "config.json"
HISTORY_CSV = CHECKPOINT_DIR / "training_history.csv"


def get_tokenizer(model_name: str = MODEL_NAME):
    """Загрузить токенизатор DistilBERT."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def build_classifier(num_labels: int, id2label=None, label2id=None):
    """Собрать TF-модель DistilBERT + голову классификации на num_labels классов."""
    from transformers import TFAutoModelForSequenceClassification
    return TFAutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )


def _encode(texts, tokenizer):
    """Субсловная токенизация с обрезкой/паддингом до MAX_LENGTH -> dict numpy."""
    enc = tokenizer(
        list(texts), truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="np",
    )
    return {k: v for k, v in enc.items()}


def make_dataset(texts, labels, tokenizer, batch_size=BATCH_SIZE, shuffle=False):
    """Собрать tf.data.Dataset из текстов и целочисленных меток."""
    import tensorflow as tf
    enc = _encode(texts, tokenizer)
    ds = tf.data.Dataset.from_tensor_slices((enc, np.array(labels, dtype=np.int32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train(splits, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
          per_epoch_ckpt=True):
    """
    Дообучить DistilBERT на splits['train'] (en+ru), валидация — en+ru val вместе.
    splits — из data_loader.get_train_val_test().

    Возвращает (model, tokenizer, history).
    """
    import tensorflow as tf
    import pandas as pd
    from utils.data_loader import get_category_names

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    names = get_category_names()
    num_labels = len(names)
    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in enumerate(names)}

    tr = splits["train"]
    val = pd.concat([splits["val"]["en"], splits["val"]["ru"]], ignore_index=True)

    tokenizer = get_tokenizer()
    model = build_classifier(num_labels, id2label=id2label, label2id=label2id)

    train_ds = make_dataset(tr["text"], tr["label"], tokenizer, batch_size, shuffle=True)
    val_ds = make_dataset(val["text"], val["label"], tokenizer, batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    callbacks = [
        # лучший чекпоинт по val_loss -> text_classifier_best.h5
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_PATH), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1),
        # история по эпохам -> training_history.csv
        tf.keras.callbacks.CSVLogger(str(HISTORY_CSV)),
    ]
    if per_epoch_ckpt:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(CHECKPOINT_DIR / "epoch_{epoch:02d}.h5"),
            save_weights_only=True, save_freq="epoch", verbose=0))

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, callbacks=callbacks)

    # сохранить конфиг и токенизатор (для инференса)
    model.config.to_json_file(str(CONFIG_JSON))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))
    return model, tokenizer, history


def load_trained(num_labels=18):
    """Пересобрать модель и загрузить лучшие веса из checkpoints/ (для инференса)."""
    from utils.data_loader import get_category_names
    names = get_category_names()
    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in enumerate(names)}
    model = build_classifier(num_labels, id2label=id2label, label2id=label2id)
    model.load_weights(str(BEST_PATH))
    tokenizer = get_tokenizer(str(CHECKPOINT_DIR))
    return model, tokenizer


def predict_labels(model, tokenizer, texts, batch_size=64):
    """Вернуть предсказанные ЦЕЛОЧИСЛЕННЫЕ метки (argmax логитов), батчами."""
    import tensorflow as tf
    preds = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = _encode(chunk, tokenizer)
        logits = model(enc, training=False).logits
        preds.extend(tf.argmax(logits, axis=-1).numpy().tolist())
    return preds


def evaluate_transformer(model, tokenizer, splits, languages=("en", "ru")):
    """
    accuracy и F1-macro ОТДЕЛЬНО по языкам на test (те же метрики, что у бейзлайна),
    чтобы напрямую сравнить с ним.
    """
    from sklearn.metrics import accuracy_score, f1_score
    out = {}
    all_true, all_pred = [], []
    for lang in languages:
        te = splits["test"][lang]
        y_true = list(te["label"])
        y_pred = predict_labels(model, tokenizer, te["text"])
        all_true += y_true
        all_pred += y_pred
        out[lang] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "n": len(te),
        }
    out["overall"] = {
        "accuracy": round(accuracy_score(all_true, all_pred), 4),
        "f1_macro": round(f1_score(all_true, all_pred, average="macro"), 4),
        "n": len(all_true),
    }
    return out
    