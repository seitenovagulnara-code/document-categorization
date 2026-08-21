"""
Этап 9 — Дашборд реального времени (Streamlit).

Запуск:  streamlit run app/real_time_dashboard.py

Заготовка каркаса: показывает разделы интерфейса и работает БЕЗ обученной модели
(выводит подсказку, если модели ещё нет). Логику подключим на этапе 9.
"""

import json
from pathlib import Path

import streamlit as st

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"


def load_metrics():
    """Подтянуть reports/performance_metrics.json, если он уже создан."""
    path = REPORTS_DIR / "performance_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def model_available() -> bool:
    """Есть ли обученная модель в checkpoints/."""
    return any(CHECKPOINT_DIR.glob("*.h5"))


st.set_page_config(page_title="Document Categorization & Tagging", layout="wide")
st.title("📄 Document Categorization & Tagging")

if not model_available():
    st.info("Модель ещё не обучена. Раздел классификации подключится после этапа 5 "
            "(обучение DistilBERT). Пока это каркас интерфейса.")

# --- Ввод документа ---
st.subheader("Документ")
text = st.text_area("Вставьте текст документа", height=160)
run = st.button("Классифицировать и разметить")

col1, col2 = st.columns(2)

# --- Результат классификации ---
with col1:
    st.subheader("Категория")
    if run and text:
        st.write("TODO (этап 5): предсказание категории DistilBERT")
        # категория + уверенность + определённый язык

# --- Теги ---
with col2:
    st.subheader("Теги")
    if run and text:
        st.write("TODO (этап 7): сущности и теги из spaCy + NER")

st.divider()

# --- Метрики модели ---
st.subheader("Метрики")
metrics = load_metrics()
if metrics:
    st.json(metrics)
else:
    st.caption("reports/performance_metrics.json появится после этапа 6.")

# --- Визуализации (этап 9) ---
st.subheader("Распределения")
st.caption("TODO (этап 9): распределение категорий, счётчики тегов, языки, "
           "скорость обработки.")
