from typing import List, Dict, Any
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def tokenize(text: str) -> List[str]:
    return word_tokenize(text)

def stemming(tokens: List[str]) -> List[str]:
    return [stemmer.stem(t) for t in tokens]

def lemmatization(tokens: List[str]) -> List[str]:
    return [lemmatizer.lemmatize(t) for t in tokens]

def pos_tagging(tokens: List[str]) -> List[List[str]]:
    return [[w, tag] for w, tag in pos_tag(tokens)]

def ner(tokens: List[str]) -> List[Dict[str, str]]:
    tree = ne_chunk(pos_tag(tokens))
    entities: List[Dict[str, str]] = []

    for node in tree:
        if hasattr(node, "label"):
            label = node.label()
            name = " ".join([leaf[0] for leaf in node.leaves()])
            entities.append({"entity": name, "label": label})

    return entities

def full_pipeline(text: str) -> Dict[str, Any]:
    toks = tokenize(text)
    return {
        "tokens": toks,
        "stems": stemming(toks),
        "lemmas": lemmatization(toks),
        "pos": pos_tagging(toks),
        "ner": ner(toks),
    }
