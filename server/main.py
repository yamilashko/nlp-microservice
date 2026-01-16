from fastapi import FastAPI
from server.api.routes import router

app = FastAPI(
    title="NLP Microservice",
    version="1.0.0",
    description="Учебный микросервис на FastAPI: BoW, TF-IDF, LSA, NLTK.",
)

app.include_router(router)
