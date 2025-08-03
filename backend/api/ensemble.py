from fastapi import APIRouter, HTTPException
from backend.models.loader import load_model
from ml import config
import torch

router = APIRouter()

LABEL_MAP = {0: "Fake", 1: "Real"}

@router.post("/ensemble")
def ensemble_predict(text: str, models: list):
    """Simple majority voting ensemble for demonstration."""
    votes = []
    for model_name in models:
        try:
            model_obj = load_model(model_name)
        except Exception as e:
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
                pred = torch.argmax(outputs.logits, dim=1).item()
            votes.append(pred)
        else:
            pipeline = model_obj
            pred = pipeline.predict([text])[0]
            votes.append(pred)
    # Majority vote
    if votes:
        result = max(set(votes), key=votes.count)
    else:
        result = None
    label_votes = [LABEL_MAP[v] for v in votes]
    label_result = LABEL_MAP[result] if result is not None else None
    return {"ensemble_prediction": label_result, "votes": label_votes}
