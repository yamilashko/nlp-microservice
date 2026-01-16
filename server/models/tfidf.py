from typing import List, Tuple
import numpy as np
from server.models.bow import bow_matrix

def tfidf_matrix(texts: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    TF-IDF (numpy):
    TF = count(word in doc)
    IDF = log((N + 1) / (df + 1)) + 1   (сглаженная версия)
    """
    vocabulary, bow = bow_matrix(texts)  # (n_docs, n_words)
    n_docs = bow.shape[0]

    df = np.sum(bow > 0, axis=0)               # (n_words,)
    idf = np.log((n_docs + 1) / (df + 1)) + 1  # (n_words,)

    tf = bow.astype(float)
    tfidf = tf * idf

    return vocabulary, tfidf
