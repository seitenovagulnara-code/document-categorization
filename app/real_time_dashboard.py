"""
Этап 9 — Дашборд (Streamlit, облегчённый + живая классификация).

Запуск:  streamlit run app/real_time_dashboard.py

Не требует torch/transformers. Компоненты подключаются, если доступны:
  - метрики / примеры        — из reports/ (нужны только streamlit + pandas);
  - живая классификация      — бейзлайн TF-IDF+LogReg (scikit-learn + .joblib);
  - живое тегирование        — spaCy (+ модели en/ru).

Показывает (по условию): результаты классификации, теги, метрики, языки.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
REPORTS = BASE / "reports"


@st.cache_data
def load_metrics():
    p = REPORTS / "performance_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_examples():
    p = REPORTS / "example_predictions.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_baseline():
    """Лёгкий классификатор (TF-IDF+LogReg). None, если нет sklearn или файла."""
    try:
        from models.text_classifier import BaselineClassifier, BASELINE_PATH
        if not Path(BASELINE_PATH).exists():
            return None
        return BaselineClassifier().load()
    except Exception:
        return None


@st.cache_resource
def load_tagger():
    """Тегировщик spaCy. Объект вернётся всегда, но методы упадут без spaCy."""
    try:
        from models.tagger import DocumentTagger
        return DocumentTagger()
    except Exception:
        return None


st.set_page_config(page_title="Document Categorization & Tagging", layout="wide")
st.title("📄 Document Categorization & Tagging")
st.caption("Классификация (DistilBERT / бейзлайн) + тегирование (spaCy NER) · en/ru")

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

# ---------- Живая обработка: классификация + теги ----------
st.header("Обработать свой текст")
from utils.text_preprocessing import detect_language   # только регулярки, лёгкий

txt = st.text_area("Введите текст (en/ru)", height=100)
if st.button("Обработать") and txt.strip():
    lang = detect_language(txt)
    st.write(f"**Язык:** `{lang}`")

    # --- классификация (бейзлайн) ---
    clf = load_baseline()
    category = None
    if clf is not None:
        category = clf.predict([txt])[0]
        st.success(f"**Категория:** {category}")
    else:
        st.caption("Живая классификация недоступна: нужен baseline_tfidf_logreg.joblib "
                   "в models/checkpoints/ и scikit-learn. (Категории — в примерах ниже.)")

    # --- теги (spaCy), с учётом категории как контекста ---
    tagger = load_tagger()
    try:
        res = tagger.generate_tags(txt, lang=lang, category=category)
        if res["entities"]:
            st.write("**Сущности:** " +
                     ", ".join(f'{e["text"]} ({e["type"]})' for e in res["entities"]))
        st.write("**Теги:** " + (", ".join(res["tags"]) if res["tags"] else "—"))
    except Exception:
        st.caption("Живое тегирование недоступно (нет spaCy).")

# ---------- Примеры обработки ----------
if examples is not None:
    st.header("Примеры обработки")
    langs = ["все"] + sorted(examples["language"].unique().tolist())
    pick = st.selectbox("Язык", langs)
    view = examples if pick == "все" else examples[examples["language"] == pick]
    st.dataframe(view, use_container_width=True, height=300)

    st.header("Распределения")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("По категориям")
        st.bar_chart(examples["predicted_label"].value_counts())
    with col2:
        st.subheader("По языкам")
        st.bar_chart(examples["language"].value_counts())
else:
    st.info("Нет reports/example_predictions.csv — сгенерируй его в Colab.")
    