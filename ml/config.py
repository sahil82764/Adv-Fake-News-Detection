import torch
import os

# --- General Project Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- Path Definitions ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

LABEL_MAP = {0: "Fake", 1: "Real"}

# --- Transformer Model Configuration ---
# This dictionary holds settings for models like DistilBERT and ALBERT.
# These parameters are optimized for a GPU with limited VRAM, as per the project plan.
TRANSFORMER_CONFIG = {
    'batch_size': 4,
    'gradient_accumulation_steps': 4, # Effective batch size will be 4 * 4 = 16
    'max_length': 512,
    'learning_rate': 2e-5,
    'epochs': 3,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'text_column': 'text_raw', # Use the raw text for transformers
    'model_names': {
        'distilbert': 'distilbert-base-uncased',
        'albert': 'albert-base-v2'
    }
}

# --- Traditional ML Model Configuration ---
# This dictionary holds settings for models like Naive Bayes, SVM, etc.
# These models will use the heavily processed text data.
TRADITIONAL_ML_CONFIG = {
    'text_column': 'text_processed', # Use the processed text for traditional ML
    'vectorizer_params': {
        'max_features': 10000,
        'ngram_range': (1, 2) # Consider both single words and two-word phrases
    }
}

# --- (Optional) GPT-2 Data Augmentation Configuration ---
GPT2_CONFIG = {
    'model_name': 'gpt2', # 124M parameter model, feasible to run locally
    'max_length': 200,    # Max length of generated synthetic articles
    'temperature': 0.8,   # Controls randomness. Lower is more deterministic.
    'do_sample': True,
    'num_return_sequences': 3 # Number of samples to generate per input
}

