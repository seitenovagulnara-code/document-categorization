# Document Categorization & Tagging

Интеллектуальная система классификации документов по категориям и автоматического
тегирования на основе NLP и transfer learning. Поддержка нескольких языков
(английский + русский), обработка в реальном времени и дашборд с визуализацией.

## Задача

Для каждого входящего документа система:
1. **классифицирует** его в одну из заранее заданных категорий (дообученный DistilBERT);
2. **тегирует** — извлекает сущности и темы (spaCy + NER);
3. делает это **на нескольких языках** и **в реальном времени** (≥100 док/сек);
4. показывает результаты на **дашборде** (Streamlit).

Пороги качества по условию: accuracy ≥ 85%, F1-macro ≥ 0.80,
скорость ≥ 100 док/сек, точность ≥ 80% на каждый язык, и обгон бейзлайна
(TF-IDF + LogReg) минимум на 5%.

## Датасет

**AmazonScience/massive** — короткие тексты (запросы к голосовому помощнику),
классификация по 18 «сценариям» (alarm, music, news, weather, calendar…).
Языки: `en-US` + `ru-RU` (английский + русский). Официальный датасет Amazon
Science, лицензия CC-BY-4.0, хранится на Hugging Face в формате parquet.

> Примечание: в условии рекомендованы MLDoc и XGLUE, но у обоих сейчас закрыт
> доступ к исходным данным (лицензия Reuters/NIST у MLDoc; у XGLUE Microsoft
> отключил публичный доступ к хранилищу — ошибка 409). MASSIVE — надёжная
> открытая замена, закрывающая требования (≥5 категорий, англ+рус, ≥10 000
> документов). Отклонение согласуется с преподавателем.
>
> Минус MASSIVE: тексты короткие (фразы, не длинные статьи) — для NER легче.

Данные НЕ хранятся в репозитории — они скачиваются кодом при первом запуске
(`utils/data_loader.py`) и кэшируются в `data/raw_documents/`.

## Структура

```
document-categorization-tagging/
├── data/
│   ├── raw_documents/      # скачанный сырой датасет (кэш, не в git)
│   └── processed_data/     # очищенные train/test (кэш, не в git)
├── models/
│   ├── text_classifier.py  # классификаторы: бейзлайн + DistilBERT
│   ├── tagger.py           # тегирование: spaCy + NER
│   └── checkpoints/        # веса модели (артефакты обучения)
├── notebooks/
│   └── EDA_and_Training.ipynb   # EDA + обучение (запускать в Colab)
├── reports/
│   ├── performance_metrics.json # итоговые метрики
│   └── example_predictions.csv  # примеры предсказаний
├── utils/
│   ├── data_loader.py      # этап 1: загрузка данных
│   ├── text_preprocessing.py  # этап 2-3: очистка, язык, препроцессинг
│   └── transfer_learning.py   # этап 5: дообучение DistilBERT
├── app/
│   └── real_time_dashboard.py  # этап 9: Streamlit-дашборд
├── README.md
└── requirements.txt
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
```

## План работ (по этапам)

- [ ] **Этап 1** — загрузка данных (`utils/data_loader.py`)
- [ ] **Этап 2** — EDA + базовая очистка (`notebooks/`, `utils/text_preprocessing.py`)
- [ ] **Этап 4** — бейзлайн TF-IDF + LogReg (`models/text_classifier.py`)
- [ ] **Этап 5** — дообучение DistilBERT (`utils/transfer_learning.py`) — в Colab с GPU
- [ ] **Этап 6** — сравнение метрик, обгон бейзлайна
- [ ] **Этап 7** — тегирование spaCy + NER (`models/tagger.py`)
- [ ] **Этап 8** — пайплайн реального времени (батчинг, определение языка)
- [ ] **Этап 9** — дашборд (`app/real_time_dashboard.py`)
- [ ] **Этап 10** — артефакты и отчёты (`reports/`)

## Обучение и запуск

**Обучение** (тяжёлое, нужен GPU) — в Google Colab: открыть
`notebooks/EDA_and_Training.ipynb`, смонтировать Drive, обучить DistilBERT,
сохранить артефакт модели в `models/checkpoints/`.

**Дашборд** (локально):
```bash
streamlit run app/real_time_dashboard.py
```

> Как получить модель: при свежем `clone` весов модели в `checkpoints/` нет.
> Нужно либо запустить обучение из ноутбука, либо (позже) подтянуть готовый
> артефакт. Точный способ будет описан здесь на этапе 10.
