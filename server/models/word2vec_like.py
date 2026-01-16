from typing import List, Dict, Any
from sklearn.decomposition import TruncatedSVD
from server.models.bow import bow_matrix

def word_vectors(texts: List[str], n_components: int = 3) -> Dict[str, Any]:
    """
    'word2vec-like' (учебный вариант):
    - строим BoW (doc x word)
    - transpose => (word x doc)
    - SVD => компактные вектора слов
    """
    vocab, bow = bow_matrix(texts)  # (n_docs, n_words)
    word_doc = bow.T               # (n_words, n_docs)

    if word_doc.shape[1] <= 1:
        n_components = 1
    else:
        n_components = min(n_components, word_doc.shape[1] - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    wvecs = svd.fit_transform(word_doc)

    vectors = {vocab[i]: wvecs[i].tolist() for i in range(len(vocab))}

    return {
        "vectors": vectors,
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
    }
