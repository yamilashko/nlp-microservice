# NLP Microservice (FastAPI)

Учебный проект: FastAPI-сервис для обработки текстов + клиент, который отправляет корпус.

## Установка (macOS)
1) Создай venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Установи зависимости:
```bash
pip install -r requirements.txt
```

3) Скачай данные NLTK (один раз):
```bash
python -m server.preprocessing.nltk_download
```

## Запуск сервера
```bash
uvicorn server.main:app --reload
```
Документация: http://127.0.0.1:8000/docs

## Запуск клиента
В другом окне терминала (с активированным venv):
```bash
python client/send_corpus.py
```

## Про word2vec
В sklearn нет настоящего word2vec (как в gensim). Здесь сделан "word2vec-like":
вектора слов получаются через SVD над матрицей word-doc.
