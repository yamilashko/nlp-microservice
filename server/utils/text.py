import re
from typing import List

def normalize(text: str) -> str:
    """
    Очень простая нормализация:
    - lower()
    - оставляем только буквы/цифры/пробелы
    - схлопываем пробелы
    """
    text = text.lower()
    text = re.sub(r"[^a-zа-я0-9\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_simple(text: str) -> List[str]:
    """
    Простейшая токенизация для BoW/TF-IDF (по заданию пишем сами).
    """
    text = normalize(text)
    return text.split() if text else []
