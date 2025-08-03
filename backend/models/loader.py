import os
import sys
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ml import config

MODEL_CACHE = {}
TOKENIZER_CACHE = {}

# Supported models
TRANSFORMER_MODELS = list(config.TRANSFORMER_CONFIG['model_names'].keys())
TRADITIONAL_MODELS = ['logistic_regression', 'naive_bayes', 'svm']

# Add deep learning and ensemble models as needed

def load_model(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    if model_name in TRANSFORMER_MODELS:
        model_path = os.path.join(config.MODELS_DIR, f'{model_name}_model')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(config.DEVICE)
        model.eval()
        MODEL_CACHE[model_name] = (model, tokenizer)
        return model, tokenizer
    elif model_name in TRADITIONAL_MODELS:
        model_path = os.path.join(config.MODELS_DIR, f'{model_name}_model.joblib')
        pipeline = joblib.load(model_path)
        MODEL_CACHE[model_name] = pipeline
        return pipeline
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
