"""
Этап 7 — Тегирование документов (spaCy + NER).

Отдельный движок, НЕ связанный с классификатором. Для каждого документа:
  1) определяет язык (или берёт переданный),
  2) грузит нужную модель spaCy,
  3) извлекает именованные сущности (NER) — имена, места, даты, организации,
  4) добавляет ключевые слова (существительные/имена собственные),
  5) при наличии — учитывает КАТЕГОРИЮ документа как контекст.

Тексты MASSIVE короткие, поэтому сущностей мало — гибрид (NER + ключевые слова
+ категория) даёт осмысленные теги, а не пустоту. Это и есть context-aware
tagging: категория помогает уточнить теги.

spaCy импортируется внутри методов, чтобы модуль грузился без него.
"""

# Модель spaCy под каждый язык (ставятся через `python -m spacy download ...`)
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "ru": "ru_core_news_sm",
}


class DocumentTagger:
    """Извлекает сущности и генерирует теги; знает модель spaCy под каждый язык."""

    def __init__(self):
        self._nlp = {}   # кэш загруженных моделей spaCy по языку

    def _get_nlp(self, lang: str):
        """Лениво загрузить и закэшировать модель spaCy для языка."""
        import spacy
        if lang not in SPACY_MODELS:
            lang = "en"                      # запасной вариант для неизвестного языка
        if lang not in self._nlp:
            self._nlp[lang] = spacy.load(SPACY_MODELS[lang])
        return self._nlp[lang]

    def extract_entities(self, text: str, lang: str):
        """Вернуть именованные сущности [(text, label), ...] через doc.ents."""
        nlp = self._get_nlp(lang)
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def keywords(self, text: str, lang: str, max_k: int = 5):
        """Ключевые слова: существительные и имена собственные (без стоп-слов), по лемме."""
        nlp = self._get_nlp(lang)
        doc = nlp(text)
        seen, out = set(), []
        for tok in doc:
            if tok.pos_ in ("NOUN", "PROPN") and tok.is_alpha and not tok.is_stop:
                lemma = tok.lemma_.lower()
                if lemma not in seen:
                    seen.add(lemma)
                    out.append(lemma)
        return out[:max_k]

    def generate_tags(self, text: str, lang: str = None, category: str = None):
        """
        Собрать теги из сущностей + ключевых слов (+ категория как контекст).
        Если lang не задан — определяем автоматически.

        Возвращает dict:
          {
            "entities": [{"text":..., "type":...}, ...],
            "keywords": [...],
            "category": <если передана>,
            "tags": [...]                  # плоский список без дублей
          }
        """
        if lang is None:
            from utils.text_preprocessing import detect_language
            lang = detect_language(text)

        ents = self.extract_entities(text, lang)
        kws = self.keywords(text, lang)

        # убрать из ключевых слов то, что уже попало в сущности (без дублей вроде
        # 'Microsoft' + 'microsoft')
        ent_words = set()
        for t, _ in ents:
            for w in t.lower().split():
                ent_words.add(w)
        kws = [k for k in kws if k not in ent_words]

        result = {
            "entities": [{"text": t, "type": lbl} for t, lbl in ents],
            "keywords": kws,
        }
        if category:
            result["category"] = category

        flat = ([category] if category else []) + [t for t, _ in ents] + kws
        result["tags"] = list(dict.fromkeys(flat))   # убрать дубли, сохранив порядок
        return result
        