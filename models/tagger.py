"""
Этап 7 — Тегирование документов (spaCy + NER).

Отдельный движок, НЕ связанный с классификатором. Для каждого документа:
извлекает именованные сущности (NER) и превращает их в теги.
Модели spaCy зависят от языка -> сначала определяем язык, потом грузим модель.

Заготовка: интерфейс класса. Логику наполняем на этапе 7.
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
        # TODO (этап 7): spacy.load(SPACY_MODELS[lang]) с кэшированием
        raise NotImplementedError

    def extract_entities(self, text: str, lang: str):
        """Вернуть сущности [(text, label), ...] через doc.ents."""
        # TODO (этап 7)
        raise NotImplementedError

    def generate_tags(self, text: str, lang: str, category: str = None):
        """
        Собрать теги из сущностей (+ по желанию учесть категорию как контекст).
        Контекстно-зависимое тегирование: категория помогает уточнить теги.
        """
        # TODO (этап 7): сущности -> теги; взвесить по важности
        raise NotImplementedError
