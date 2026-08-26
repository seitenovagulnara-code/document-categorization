"""
Этап 9 — Дашборд (Streamlit, облегчённый).

Запуск:  streamlit run app/real_time_dashboard.py

Не требует torch/transformers: читает готовые артефакты из reports/
  - performance_metrics.json  — метрики (accuracy, F1, скорость, по языкам),
  - example_predictions.csv   — примеры обработки (категория, теги, язык).
Живое тегирование — через spaCy, если он установлен (иначе раздел скрыт).

Показывает (по условию): результаты классификации, теги, метрики, языки.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))          # чтобы работали импорты models/ и utils/
REPORTS = BASE / "reports"


@st.cache_data
def load_metrics():
    p = REPORTS / "performance_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_examples():
    p = REPORTS / "example_predictions.csv"
    return pd.read_csv(p) if p.exists() else None


st.set_page_config(page_title="Document Categorization & Tagging", layout="wide")
st.title("📄 Document Categorization & Tagging")
st.caption("Классификация (DistilBERT) + тегирование (spaCy NER) · английский и русский")

metrics = load_metrics()
examples = load_examples()

# ---------- Метрики ----------
st.header("Метрики модели")
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f'{metrics["classification_accuracy"]:.3f}')
    c2.metric("F1-macro", f'{metrics["f1_score_macro"]:.3f}')
    c3.metric("Классиф., док/с", metrics.get("processing_speed_docs_per_sec", "—"))
    c4.metric("Пайплайн, док/с", metrics.get("full_pipeline_docs_per_sec", "—"))
    pla = metrics.get("per_language_accuracy", {})
    if pla:
        st.write("**Точность по языкам:** " +
                 " · ".join(f"{k}: {v:.3f}" for k, v in pla.items()))
else:
    st.info("Нет reports/performance_metrics.json")

# ---------- Живое тегирование (spaCy, опционально) ----------
st.header("Разметить свой текст")
try:
    from models.tagger import DocumentTagger
    from utils.text_preprocessing import detect_language
    if "tagger" not in st.session_state:
        st.session_state.tagger = DocumentTagger()

    txt = st.text_area("Введите текст (en/ru)", height=100)
    if st.button("Разметить") and txt.strip():
        lang = detect_language(txt)
        res = st.session_state.tagger.generate_tags(txt, lang=lang)
        st.write(f"**Язык:** `{lang}`")
        if res["entities"]:
            st.write("**Сущности:** " +
                     ", ".join(f'{e["text"]} ({e["type"]})' for e in res["entities"]))
        st.write("**Теги:** " + (", ".join(res["tags"]) if res["tags"] else "—"))
        st.caption("Категорию присваивает DistilBERT (см. примеры ниже / прогон в Colab).")
except Exception as e:
    st.caption(f"Живое тегирование недоступно (нет spaCy): {e}")

# ---------- Примеры обработки ----------
if examples is not None:
    st.header("Примеры обработки")
    langs = ["все"] + sorted(examples["language"].unique().tolist())
    pick = st.selectbox("Язык", langs)
    view = examples if pick == "все" else examples[examples["language"] == pick]
    st.dataframe(view, use_container_width=True, height=300)

    # ---------- Распределения ----------
    st.header("Распределения")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("По категориям")
        st.bar_chart(examples["predicted_label"].value_counts())
    with col2:
        st.subheader("По языкам")
        st.bar_chart(examples["language"].value_counts())
else:
    st.info("Нет reports/example_predictions.csv — сгенерируй его в Colab (см. инструкцию).")
    