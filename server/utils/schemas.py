from pydantic import BaseModel, Field
from typing import List, Dict, Any

class CorpusRequest(BaseModel):
    # texts — список строк (корпус)
    texts: List[str] = Field(..., description="Список текстов корпуса")

class BoWResponse(BaseModel):
    vocabulary: List[str]
    matrix: List[List[int]]

class TfIdfResponse(BaseModel):
    vocabulary: List[str]
    matrix: List[List[float]]

class AnyDictResponse(BaseModel):
    result: Dict[str, Any]
