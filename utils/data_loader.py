"""
Этап 1 — Загрузка данных (MASSIVE).

Датасет: AmazonScience/massive — короткие тексты (запросы к голосовому
помощнику), классификация по 18 «сценариям» (scenario). Официальный датасет
Amazon Science, лицензия CC-BY-4.0, хранится на Hugging Face в формате parquet
(грузится надёжно, без внешних серверов).

Языки: локали 'en-US' и 'ru-RU' (английский + русский). В отличие от XGLUE,
у MASSIVE есть train-выборка на КАЖДОМ языке — русский не только в тесте.

Данные в git НЕ хранятся: качаются кодом и кэшируются в data/raw_documents/.

Примечание: почему не XGLUE/MLDoc — у обоих отключён/закрыт доступ к исходным
данным (Microsoft blob 409 «public access disabled» и лицензия Reuters/NIST).
"""

from pathlib import Path

import pandas as pd

# Языки проекта -> локали MASSIVE
LANGUAGES = ["en", "ru"]
LOCALE = {"en": "en-US", "ru": "ru-RU"}

# 18 сценариев MASSIVE (канонический порядок = индексам ClassLabel).
# Используется как запас, если scenario придёт строкой.
SCENARIO_NAMES = [
    "alarm", "audio", "calendar", "cooking", "datetime", "email", "general",
    "iot", "lists", "music", "news", "play", "qa", "recommendation",
    "social", "takeaway", "transport", "weather",
]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_documents"
CACHE_FILE = RAW_DIR / "massive.parquet"


def _reshape(raw: pd.DataFrame, scenario_names, language: str, split: str) -> pd.DataFrame:
    """
    Привести один сплит к единой схеме:
        text        — текст запроса (поле utt)
        label       — индекс сценария (0..17)
        label_name  — название сценария
        language    — 'en' / 'ru'
        split       — 'train' / 'validation' / 'test'

    scenario может прийти как int (ClassLabel) или как строка — обрабатываем оба.
    """
    text = raw["utt"].fillna("").astype(str).str.strip()
    s = raw["scenario"]

    if pd.api.types.is_integer_dtype(s) and scenario_names is not None:
        label = s.astype(int)
        label_name = label.map(lambda i: scenario_names[i])
    else:
        label_name = s.astype(str)
        name_to_idx = {n: i for i, n in enumerate(SCENARIO_NAMES)}
        label = label_name.map(lambda n: name_to_idx.get(n, -1))

    return pd.DataFrame({
        "text": text,
        "label": label,
        "label_name": label_name,
        "language": language,
        "split": split,
    })


def _load_one_locale(locale: str):
    """Загрузить один язык MASSIVE. Сначала пробуем штатно (parquet),
    при необходимости — с trust_remote_code (для старых скриптовых версий)."""
    from datasets import load_dataset  # импорт внутри — чтобы модуль грузился без datasets
    try:
        return load_dataset("AmazonScience/massive", locale)
    except Exception:
        return load_dataset("AmazonScience/massive", locale, trust_remote_code=True)


def load_massive(languages=LANGUAGES, limit_per_split=None, cache=True) -> pd.DataFrame:
    """
    Скачать MASSIVE (en + ru) и вернуть единый DataFrame
    [text, label, label_name, language, split].

    limit_per_split — взять первые N строк каждого сплита (для быстрой отладки;
                      для финала оставить None).
    cache           — сохранить результат в data/raw_documents/massive.parquet.
    """
    frames = []
    for lang in languages:
        ds = _load_one_locale(LOCALE[lang])
        feat = ds["train"].features.get("scenario")
        names = getattr(feat, "names", None)  # список названий, если ClassLabel
        for split_key in ds.keys():           # train / validation / test
            raw = ds[split_key].to_pandas()
            if limit_per_split:
                raw = raw.head(limit_per_split)
            frames.append(_reshape(raw, names, lang, split_key))

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


if __name__ == "__main__":
    data = load_massive(limit_per_split=200)
    print(summarize(data).to_string(index=False))
    print("\nПример текста:", repr(data.iloc[0]["text"]))
    print("Категория:", data.iloc[0]["label_name"])
