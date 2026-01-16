"""
Клиентский скрипт:
- генерирует корпус
- отправляет запросы на сервер
- печатает ответы

Запуск:
python client/send_corpus.py
"""

import requests

SERVER = "http://127.0.0.1:8000"

def get_corpus():
    # Можно заменить на чтение из файла — для простоты корпус прямо тут.
    return [
        "Natural language processing is fun.",
        "FastAPI makes it easy to build microservices.",
        "NLTK provides tools for tokenization and tagging.",
        "TF IDF and Bag of Words are classic text representations."
    ]

def post(endpoint: str, payload: dict):
    url = f"{SERVER}{endpoint}"
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    texts = get_corpus()
    payload = {"texts": texts}

    print("\n=== GET / ===")
    print(requests.get(SERVER + "/").json())

    print("\n=== POST /bag-of-words ===")
    print(post("/bag-of-words", payload))

    print("\n=== POST /tf-idf ===")
    print(post("/tf-idf", payload))

    print("\n=== POST /lsa?n_components=2 ===")
    r = requests.post(SERVER + "/lsa?n_components=2", json=payload, timeout=20)
    r.raise_for_status()
    print(r.json())

    print("\n=== POST /word2vec?n_components=3 ===")
    r = requests.post(SERVER + "/word2vec?n_components=3", json=payload, timeout=20)
    r.raise_for_status()
    print(r.json())

    print("\n=== POST /text_nltk/full ===")
    print(post("/text_nltk/full", payload))

if __name__ == "__main__":
    main()
