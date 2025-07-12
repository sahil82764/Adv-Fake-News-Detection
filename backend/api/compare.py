from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from backend.models.loader import load_model
from ml import config
import torch
from ml.config import LABEL_MAP

# LABEL_MAP = {0: "Fake", 1: "Real"}

router = APIRouter()

# Request model for compare endpoint
class CompareRequest(BaseModel):
    text: str
    models: List[str]

@router.post("/models")
def compare_models(request: CompareRequest):
    text = request.text
    models = request.models
    """Compare predictions and probabilities from multiple models on a single input."""
    results = {}
    for model_name in models:
        try:
            model_obj = load_model(model_name)
        except Exception as e:
            results[model_name] = {"error": str(e)}
            continue
        if isinstance(model_obj, tuple):
            model, tokenizer = model_obj
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=config.TRANSFORMER_CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(config.DEVICE)
            attention_mask = encoding['attention_mask'].to(config.DEVICE)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                prob = probs[0, pred].item()
            results[model_name] = {"prediction": LABEL_MAP[pred], "probability": float(prob) if prob is not None else None}
        else:
            pipeline = model_obj
            pred = pipeline.predict([text])[0]
            if hasattr(pipeline, 'predict_proba'):
                prob = max(pipeline.predict_proba([text])[0])
            else:
                prob = None
            # Convert numpy types to Python types
            results[model_name] = {"prediction": LABEL_MAP[pred], "probability": float(prob) if prob is not None else None}
    return results

@router.get("/performance")
def get_performance():
    """Return model performance metrics from results/model_comparison.json."""
    import os, json
    results_path = os.path.join(config.RESULTS_DIR, 'model_comparison.json')
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Model comparison results not found.")
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data
