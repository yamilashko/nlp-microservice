from typing import List, Tuple
import numpy as np
from server.utils.text import tokenize_simple

def build_vocabulary(texts: List[str]) -> List[str]:
    """Собираем уникальные слова по всему корпусу."""
    vocab = set()
    for t in texts:
        vocab.update(tokenize_simple(t))
    return sorted(vocab)

def bow_matrix(texts: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Возвращает:
    - vocabulary: список слов
    - matrix: numpy матрица (n_docs x n_words), где значение — количество вхождений
    """
    vocabulary = build_vocabulary(texts)
    word_to_idx = {w: i for i, w in enumerate(vocabulary)}

    mat = np.zeros((len(texts), len(vocabulary)), dtype=int)

    for doc_i, text in enumerate(texts):
        for tok in tokenize_simple(text):
            mat[doc_i, word_to_idx[tok]] += 1

    return vocabulary, mat
