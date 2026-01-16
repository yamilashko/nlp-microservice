# NLP Microservice (FastAPI)

Учебный проект: микросервис на **FastAPI** для обработки текстов + клиент, который отправляет корпус на сервер.

## Возможности
- **/bag-of-words** — Bag of Words (реализация на NumPy)
- **/tf-idf** — TF-IDF (реализация на NumPy)
- **/lsa** — Latent Semantic Analysis (SVD из scikit-learn)
- **/word2vec** — упрощённый "word2vec-like" (SVD из scikit-learn)
- **/text_nltk/*** — NLTK: токенизация, стемминг, лемматизация, POS-tagging, NER

## Структура проекта
- `client/` — генерация/загрузка корпуса и запросы к API
- `server/` — FastAPI приложение
  - `api/` — роуты (эндпоинты)
  - `models/` — BoW, TF-IDF, LSA, word2vec-like
  - `preprocessing/` — функции NLTK + скачивание ресурсов
  - `utils/` — схемы и утилиты

## Установка
1) Создай виртуальное окружение:
```bash
python -m venv .venv
````

2. Активируй окружение:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

3. Установи зависимости:

```bash
pip install -r requirements.txt
```

4. Скачай ресурсы NLTK (один раз):

```bash
python -m server.preprocessing.nltk_download
```

## Запуск сервера

```bash
uvicorn server.main:app --reload
```

Документация Swagger:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Запуск клиента

В отдельном терминале (с активированным venv):

```bash
python client/send_corpus.py
```

## Примечание про "word2vec"

В scikit-learn нет классического word2vec (как в gensim).
В этом проекте сделан упрощённый вариант: **вектора слов** получаются через **SVD** над матрицей `word × document`.

