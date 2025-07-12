from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from backend.models.schemas import SinglePredictionRequest, BatchPredictionRequest, PredictionResponse
from backend.models.loader import load_model
from ml import config
import torch
import logging
import uuid
from ml.config import LABEL_MAP

router = APIRouter()


# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend_api.log', mode='a'),
        logging.StreamHandler()
    ]
)

# LABEL_MAP = {0: "Fake", 1: "Real"}

@router.post("/single", response_model=PredictionResponse)
def predict_single(request: SinglePredictionRequest):
    model_name = request.model
    text = request.text
    try:
        model_obj = load_model(model_name)
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=400, detail=f"Model loading failed: {e}")
    try:
        if isinstance(model_obj, tuple):  # Transformer
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
            return PredictionResponse(predictions=[LABEL_MAP[pred]], probabilities=[prob])
        else:  # Traditional ML
            pipeline = model_obj
            pred = pipeline.predict([text])[0]
            if hasattr(pipeline, 'predict_proba'):
                prob = max(pipeline.predict_proba([text])[0])
            else:
                prob = None
            return PredictionResponse(predictions=[LABEL_MAP[pred]], probabilities=[prob] if prob is not None else None)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@router.post("/batch", response_model=PredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    model_name = request.model
    texts = request.texts
    try:
        model_obj = load_model(model_name)
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=400, detail=f"Model loading failed: {e}")
    try:
        if isinstance(model_obj, tuple):  # Transformer
            model, tokenizer = model_obj
            encodings = tokenizer(
                texts,
                add_special_tokens=True,
                max_length=config.TRANSFORMER_CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encodings['input_ids'].to(config.DEVICE)
            attention_mask = encodings['attention_mask'].to(config.DEVICE)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
                max_probs = probs.max(dim=1).values.cpu().numpy().tolist()
            label_preds = [LABEL_MAP[p] for p in preds]
            return PredictionResponse(predictions=label_preds, probabilities=max_probs)
        else:  # Traditional ML
            pipeline = model_obj
            preds = pipeline.predict(texts).tolist()
            if hasattr(pipeline, 'predict_proba'):
                probs = pipeline.predict_proba(texts).max(axis=1).tolist()
            else:
                probs = None
            label_preds = [LABEL_MAP[p] for p in preds]
            return PredictionResponse(predictions=label_preds, probabilities=probs if probs is not None else None)
    except Exception as e:
        logging.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

class DefaultPredictionRequest(BaseModel):
    text: str

@router.post("/default", response_model=PredictionResponse)
def predict_default(request: DefaultPredictionRequest):
    """Predict using the best general model (DistilBERT)."""
    model_name = "distilbert"
    text = request.text
    try:
        model_obj = load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
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
    return PredictionResponse(predictions=[LABEL_MAP[pred]], probabilities=[prob])

class AsyncBatchRequest(BaseModel):
    texts: List[str]
    model: str
    task_id: Optional[str] = None

BATCH_RESULTS = {}

async def process_batch_task(texts, model_name, task_id):
    try:
        model_obj = load_model(model_name)
        if isinstance(model_obj, tuple):
            model, tokenizer = model_obj
            encodings = tokenizer(
                texts,
                add_special_tokens=True,
                max_length=config.TRANSFORMER_CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encodings['input_ids'].to(config.DEVICE)
            attention_mask = encodings['attention_mask'].to(config.DEVICE)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
                max_probs = probs.max(dim=1).values.cpu().numpy().tolist()
            label_preds = [LABEL_MAP[p] for p in preds]
            BATCH_RESULTS[task_id] = {"predictions": label_preds, "probabilities": max_probs}
        else:
            pipeline = model_obj
            preds = pipeline.predict(texts).tolist()
            if hasattr(pipeline, 'predict_proba'):
                probs = pipeline.predict_proba(texts).max(axis=1).tolist()
            else:
                probs = None
            label_preds = [LABEL_MAP[p] for p in preds]
            BATCH_RESULTS[task_id] = {"predictions": label_preds, "probabilities": probs}
    except Exception as e:
        logging.error(f"Async batch job failed: {e}")
        BATCH_RESULTS[task_id] = {"error": str(e)}

@router.post("/async_batch")
def async_batch(request: AsyncBatchRequest, background_tasks: BackgroundTasks):
    task_id = request.task_id or str(uuid.uuid4())
    background_tasks.add_task(process_batch_task, request.texts, request.model, task_id)
    return {"task_id": task_id, "status": "processing"}

@router.get("/async_batch_result/{task_id}")
def async_batch_result(task_id: str):
    result = BATCH_RESULTS.get(task_id)
    if result is None:
        return {"status": "processing"}
    return result
