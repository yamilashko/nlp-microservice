from typing import List, Dict, Any
from sklearn.decomposition import TruncatedSVD
from server.models.tfidf import tfidf_matrix

def lsa(texts: List[str], n_components: int = 2) -> Dict[str, Any]:

    if tfidf.shape[1] <= 1:
        n_components = 1
    else:
        n_components = min(n_components, tfidf.shape[1] - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    doc_vecs = svd.fit_transform(tfidf)

    return {
        "vocabulary": vocab,
        "tfidf_shape": list(tfidf.shape),
        "doc_vectors": doc_vecs.tolist(),
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
    }
