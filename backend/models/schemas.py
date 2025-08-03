from pydantic import BaseModel
from typing import List, Optional

class SinglePredictionRequest(BaseModel):
    text: str
    model: str

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    model: str

class PredictionResponse(BaseModel):
    predictions: List[str]
    probabilities: Optional[List[float]] = None