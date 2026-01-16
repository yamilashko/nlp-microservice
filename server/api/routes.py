from fastapi import APIRouter, HTTPException
from server.utils.schemas import CorpusRequest, BoWResponse, TfIdfResponse, AnyDictResponse

from server.models.bow import bow_matrix
from server.models.tfidf import tfidf_matrix
from server.models.lsa import lsa
from server.models.word2vec_like import word_vectors

from server.preprocessing.nltk_tools import tokenize, stemming, lemmatization, pos_tagging, ner, full_pipeline

router = APIRouter()

@router.get("/")
def root():
    return {
        "message": "NLP microservice is running",
        "docs": "/docs",
        "endpoints": [
            "/bag-of-words",
            "/tf-idf",
            "/lsa",
            "/word2vec",
            "/text_nltk/tokenize",
            "/text_nltk/stemming",
            "/text_nltk/lemmatize",
            "/text_nltk/pos",
            "/text_nltk/ner",
            "/text_nltk/full",
        ],
    }

@router.post("/bag-of-words", response_model=BoWResponse)
def bag_of_words(req: CorpusRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is empty")
    vocab, mat = bow_matrix(req.texts)
    return {"vocabulary": vocab, "matrix": mat.tolist()}

@router.post("/tf-idf", response_model=TfIdfResponse)
def tf_idf(req: CorpusRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is empty")
    vocab, mat = tfidf_matrix(req.texts)
    return {"vocabulary": vocab, "matrix": mat.tolist()}

@router.post("/lsa", response_model=AnyDictResponse)
def lsa_endpoint(req: CorpusRequest, n_components: int = 2):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is empty")
    return {"result": lsa(req.texts, n_components=n_components)}

@router.post("/word2vec", response_model=AnyDictResponse)
def word2vec_endpoint(req: CorpusRequest, n_components: int = 3):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is empty")
    return {"result": word_vectors(req.texts, n_components=n_components)}

# ---- NLTK endpoints ----

def join_corpus(texts):
    # Склеиваем корпус в один текст для демонстрации.
    return " ".join(texts)

@router.post("/text_nltk/tokenize", response_model=AnyDictResponse)
def nltk_tokenize(req: CorpusRequest):
    text = join_corpus(req.texts)
    return {"result": {"tokens": tokenize(text)}}

@router.post("/text_nltk/stemming", response_model=AnyDictResponse)
def nltk_stemming(req: CorpusRequest):
    text = join_corpus(req.texts)
    toks = tokenize(text)
    return {"result": {"stems": stemming(toks)}}

@router.post("/text_nltk/lemmatize", response_model=AnyDictResponse)
def nltk_lemmatize(req: CorpusRequest):
    text = join_corpus(req.texts)
    toks = tokenize(text)
    return {"result": {"lemmas": lemmatization(toks)}}

@router.post("/text_nltk/pos", response_model=AnyDictResponse)
def nltk_pos(req: CorpusRequest):
    text = join_corpus(req.texts)
    toks = tokenize(text)
    return {"result": {"pos": pos_tagging(toks)}}

@router.post("/text_nltk/ner", response_model=AnyDictResponse)
def nltk_ner(req: CorpusRequest):
    text = join_corpus(req.texts)
    toks = tokenize(text)
    return {"result": {"ner": ner(toks)}}

@router.post("/text_nltk/full", response_model=AnyDictResponse)
def nltk_full(req: CorpusRequest):
    text = join_corpus(req.texts)
    return {"result": full_pipeline(text)}
