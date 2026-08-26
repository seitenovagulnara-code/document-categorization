"""
Этап 5 — Transfer learning: дообучение DistilBERT (PyTorch, HF Trainer).

Многоязычная distilbert-base-multilingual-cased дообучается на train (en+ru).
Обучение тяжёлое — запускать в Google Colab с GPU.

Почему PyTorch, а не TensorFlow: из свежего transformers (5.x) поддержку TF
убрали, TF-классов больше нет. PyTorch — текущий стандарт, работает из коробки.
(Условие требует .h5/TF — это устаревший пункт, согласуется с преподавателем.)

Стратегия («надёжно»):
  - 5 эпох, learning rate 3e-5, max_length=64;
  - оценка и сохранение ПОСЛЕ КАЖДОЙ эпохи;
  - load_best_model_at_end по F1-macro -> в конце берём лучшую модель;
  - история обучения пишется в training_history.csv.

Артефакты (в models/checkpoints/):
  model.safetensors + config.json  — лучшая модель (через trainer.save_model),
  training_history.csv             — метрики по эпохам,
  + токенизатор (для инференса).

transformers/torch/datasets импортируются ВНУТРИ функций, чтобы модуль
грузился без них (например, при импорте из дашборда).
"""

from pathlib import Path

MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 64           # по EDA: медиана 6 слов, max 61 -> 64 с запасом
LEARNING_RATE = 3e-5      # условие: 2e-5 … 5e-5
EPOCHS = 5                # условие: минимум 5 эпох
BATCH_SIZE = 32

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
HISTORY_CSV = CHECKPOINT_DIR / "training_history.csv"


def _to_hf_dataset(df, tokenizer):
    """pandas -> токенизированный HF Dataset с колонкой 'labels'."""
    from datasets import Dataset
    ds = Dataset.from_pandas(
        df[["text", "label"]].rename(columns={"label": "labels"}),
        preserve_index=False,
    )

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(tok, batched=True)
    return ds.remove_columns(["text"])


def _compute_metrics(eval_pred):
    """accuracy + F1-macro для оценки на валидации во время обучения."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def _save_history(trainer):
    """Записать историю обучения (по эпохам) в training_history.csv."""
    import pandas as pd
    pd.DataFrame(trainer.state.log_history).to_csv(HISTORY_CSV, index=False)


def train(splits, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    Дообучить DistilBERT на splits['train'] (en+ru), валидация — en+ru val вместе.
    splits — из data_loader.get_train_val_test().

    Возвращает (model, tokenizer, trainer).
    """
    import pandas as pd
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )
    from utils.data_loader import get_category_names

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    names = get_category_names()
    num_labels = len(names)
    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in enumerate(names)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )

    tr = splits["train"]
    val = pd.concat([splits["val"]["en"], splits["val"]["ru"]], ignore_index=True)
    train_ds = _to_hf_dataset(tr, tokenizer)
    val_ds = _to_hf_dataset(val, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)  # динамический паддинг (быстрее)

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",         # оценка после каждой эпохи
        save_strategy="epoch",         # сохранение после каждой эпохи
        load_best_model_at_end=True,   # в конце оставить лучшую модель
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",              # без wandb и прочих трекеров
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=_compute_metrics,
    )
    trainer.train()

    trainer.save_model(str(CHECKPOINT_DIR))       # лучшая модель: model.safetensors + config.json
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))
    _save_history(trainer)
    return model, tokenizer, trainer


def load_trained():
    """Загрузить дообученную модель и токенизатор из checkpoints/ (для инференса)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    return model, tokenizer


def predict_labels(model, tokenizer, texts, batch_size=64):
    """Вернуть предсказанные ЦЕЛОЧИСЛЕННЫЕ метки (argmax логитов), батчами."""
    import torch
    model.eval()
    device = next(model.parameters()).device
    preds = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return preds


def evaluate_transformer(model, tokenizer, splits, languages=("en", "ru")):
    """
    accuracy и F1-macro ОТДЕЛЬНО по языкам на test (те же метрики, что у бейзлайна).
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
    