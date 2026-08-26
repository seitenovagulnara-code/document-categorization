"""
Этап 8 — Пайплайн реального времени + замер скорости.

Объединяет два движка над одним потоком документов:
  - классификация (дообученный DistilBERT) — батчами, ради пропускной способности;
  - тегирование (spaCy + NER) — с учётом языка и предсказанной категории.

Плюс определение языка (для маршрутизации spaCy) и бенчмарк скорости
классификации (порог условия: ≥100 док/сек; меряем батчами на GPU).

Тяжёлые импорты — внутри методов.
"""

import time


class RealTimePipeline:
    """Классификация (батчами) + тегирование одного/многих документов."""

    def __init__(self, model, tokenizer, tagger=None):
        self.model = model
        self.tokenizer = tokenizer
        if tagger is None:
            from models.tagger import DocumentTagger
            tagger = DocumentTagger()
        self.tagger = tagger
        from utils.data_loader import get_category_names
        self._names = get_category_names()

    def classify(self, texts, batch_size=64):
        """Предсказать КАТЕГОРИИ (названия) батчами."""
        from utils.transfer_learning import predict_labels
        ids = predict_labels(self.model, self.tokenizer, texts, batch_size=batch_size)
        return [self._names[i] for i in ids]

    def process(self, texts, langs=None, batch_size=64):
        """
        Полный проход: язык -> категория (батчами) -> теги.
        Возвращает список dict: {text, language, category, tags}.
        """
        texts = list(texts)
        if langs is None:
            from utils.text_preprocessing import detect_language
            langs = [detect_language(t) for t in texts]
        cats = self.classify(texts, batch_size=batch_size)

        out = []
        for text, lang, cat in zip(texts, langs, cats):
            tagged = self.tagger.generate_tags(text, lang=lang, category=cat)
            out.append({
                "text": text,
                "language": lang,
                "category": cat,
                "tags": tagged["tags"],
            })
        return out

    def process_one(self, text, lang=None):
        """Обработать один документ."""
        langs = [lang] if lang else None
        return self.process([text], langs=langs)[0]

    def benchmark_classification(self, texts, batch_size=64):
        """
        Замер пропускной способности КЛАССИФИКАЦИИ (док/сек).
        Первый прогон-разогрев (важно для GPU), затем замер.
        """
        texts = list(texts)
        self.classify(texts[:batch_size], batch_size=batch_size)  # разогрев
        t0 = time.perf_counter()
        self.classify(texts, batch_size=batch_size)
        dt = time.perf_counter() - t0
        return {
            "n": len(texts),
            "seconds": round(dt, 3),
            "docs_per_sec": round(len(texts) / dt, 1),
            "batch_size": batch_size,
        }

    def benchmark_full(self, texts, langs=None, batch_size=64):
        """
        Замер ПОЛНОГО пайплайна: классификация (BERT, батчами) + тегирование
        (spaCy, по одному). Ближе к реальной скорости обработки, чем только
        классификация. Обычно заметно ниже — spaCy работает на CPU.
        """
        texts = list(texts)
        self.process(texts[:batch_size], batch_size=batch_size)   # разогрев
        t0 = time.perf_counter()
        self.process(texts, langs=langs, batch_size=batch_size)
        dt = time.perf_counter() - t0
        return {
            "n": len(texts),
            "seconds": round(dt, 3),
            "docs_per_sec": round(len(texts) / dt, 1),
            "batch_size": batch_size,
        }


def build_pipeline():
    """Загрузить дообученную модель + тегировщик и собрать пайплайн."""
    from utils.transfer_learning import load_trained
    model, tokenizer = load_trained()
    return RealTimePipeline(model, tokenizer)
    