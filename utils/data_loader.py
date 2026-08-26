"""
Этап 1 — Загрузка данных (MASSIVE, parquet-копия от MTEB).

Датасет: mteb/amazon_massive_scenario — короткие тексты (запросы к голосовому
помощнику), классификация по 18 «сценариям» (scenario). Это parquet-копия
датасета Amazon MASSIVE: грузится на любой версии `datasets` (в т.ч. 3.x),
без скриптов и без trust_remote_code.

Языки: конфигурации 'en' и 'ru' (английский + русский). У обоих есть train.

Почему не AmazonScience/massive напрямую: он «скриптовый» (massive.py), а в
datasets 3.x загрузку скриптов убрали. Почему не XGLUE/MLDoc: у них закрыт
доступ к исходным данным. mteb/amazon_massive_scenario — те же данные, но
в надёжном parquet-формате.

Данные в git НЕ хранятся: качаются кодом и кэшируются в data/raw_documents/.
"""

from pathlib import Path

import pandas as pd

REPO = "mteb/amazon_massive_scenario"

# Языки проекта -> конфигурации датасета
LANGUAGES = ["en", "ru"]
CONFIG = {"en": "en", "ru": "ru"}

# 18 сценариев в алфавитном порядке (= индексам, если метка придёт числом).
SCENARIO_NAMES = [
    "alarm", "audio", "calendar", "cooking", "datetime", "email", "general",
    "iot", "lists", "music", "news", "play", "qa", "recommendation",
    "social", "takeaway", "transport", "weather",
]
_NAME_TO_IDX = {n: i for i, n in enumerate(SCENARIO_NAMES)}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_documents"
CACHE_FILE = RAW_DIR / "massive.parquet"


def _reshape(raw: pd.DataFrame, language: str, split: str) -> pd.DataFrame:
    """
    Привести один сплит к единой схеме:
        text, label, label_name, language, split.
    Метка в mteb-копии приходит строкой ('alarm'); поддержим и числовой вариант.
    """
    text_col = "text" if "text" in raw.columns else "utt"
    text = raw[text_col].fillna("").astype(str).str.strip()

    if "label_text" in raw.columns:
        label_name = raw["label_text"].astype(str)
    elif raw["label"].dtype == object or pd.api.types.is_string_dtype(raw["label"]):
        label_name = raw["label"].astype(str)
    else:
        label_name = raw["label"].astype(int).map(lambda i: SCENARIO_NAMES[i])

    label = label_name.map(lambda n: _NAME_TO_IDX.get(n, -1))

    return pd.DataFrame({
        "text": text,
        "label": label,
        "label_name": label_name,
        "language": language,
        "split": split,
    })


def load_massive(languages=LANGUAGES, limit_per_split=None, cache=True) -> pd.DataFrame:
    """
    Скачать MASSIVE (en + ru) из parquet-копии и вернуть единый DataFrame
    [text, label, label_name, language, split].

    limit_per_split — взять первые N строк каждого сплита (для быстрой отладки).
    cache           — сохранить в data/raw_documents/massive.parquet.
    """
    from datasets import load_dataset  # импорт внутри — чтобы модуль грузился без datasets

    frames = []
    for lang in languages:
        ds = load_dataset(REPO, CONFIG[lang])   # parquet-нативно, без trust_remote_code
        for split_key in ds.keys():             # train / validation / test
            raw = ds[split_key].to_pandas()
            if limit_per_split:
                raw = raw.head(limit_per_split)
            frames.append(_reshape(raw, lang, split_key))

    df = pd.concat(frames, ignore_index=True)
    if cache:
        save_raw(df)
    return df


def save_raw(df: pd.DataFrame, path: Path = CACHE_FILE) -> None:
    """Сохранить в кэш (parquet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_cached(path: Path = CACHE_FILE):
    """Загрузить из кэша, если он есть; иначе None."""
    return pd.read_parquet(path) if path.exists() else None


def get_category_names():
    """Вернуть список из 18 названий сценариев."""
    return list(SCENARIO_NAMES)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Свод: сколько документов по (язык, сплит)."""
    return (df.groupby(["language", "split"]).size()
              .rename("documents").reset_index())


def get_train_val_test(df=None):
    """
    Разложить данные под обучение по нашей схеме (решения этапа 2):
      - train — английский + русский ВМЕСТЕ (кросс-язычно, надёжнее для рус),
      - val / test — по языкам РАЗДЕЛЬНО (чтобы считать метрики на каждый язык).

    Если df не передан — грузит полный датасет (load_massive()).

    Возвращает словарь:
        {
          "train": DataFrame,                       # en+ru вместе
          "val":   {"en": DataFrame, "ru": DataFrame},
          "test":  {"en": DataFrame, "ru": DataFrame},
        }
    """
    if df is None:
        df = load_massive()

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = {lang: df[(df["split"] == "validation") & (df["language"] == lang)]
                   .reset_index(drop=True)
           for lang in LANGUAGES}
    test = {lang: df[(df["split"] == "test") & (df["language"] == lang)]
                    .reset_index(drop=True)
            for lang in LANGUAGES}
    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    data = load_massive(limit_per_split=200)
    print(summarize(data).to_string(index=False))
    print("\nПример текста:", repr(data.iloc[0]["text"]))
    print("Категория:", data.iloc[0]["label_name"])
    