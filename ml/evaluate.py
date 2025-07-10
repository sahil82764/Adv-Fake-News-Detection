import os
import sys
import logging
import time
import torch
import joblib
import pandas as pd
import psutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# This ensures the script can be run from anywhere by adding the project root to the path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.dataset import FakeNewsDataset
from ml import config
from ml.utils import compute_metrics, save_json_results

# --- Setup Logging ---
os.makedirs(config.LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(config.LOGS_DIR, 'evaluate.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()]
)

def get_memory_usage():
    """Returns the current process's memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def evaluate_model(model_name, model_type):
    """
    Loads a model, evaluates its performance and resource usage on the test set.
    """
    logging.info(f"--- Evaluating {model_name} ({model_type}) ---")
    
    # --- 1. Load Test Data ---
    text_column = config.TRANSFORMER_CONFIG['text_column'] if model_type == 'transformer' else config.TRADITIONAL_ML_CONFIG['text_column']
    
    test_dataset = FakeNewsDataset(
        data_path=os.path.join(config.DATA_DIR, 'test.parquet'),
        text_column=text_column
    )
    X_test = [item['text'] for item in test_dataset]
    y_test = [item['labels'].item() for item in test_dataset]
    
    # --- 2. Load Model and Measure Memory ---
    mem_before = get_memory_usage()
    
    if model_type == 'transformer':
        model_path = os.path.join(config.MODELS_DIR, f'{model_name}_model')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(config.DEVICE)
        model.eval()
    else: # traditional
        model_path = os.path.join(config.MODELS_DIR, f'{model_name}_model.joblib')
        pipeline = joblib.load(model_path)

    mem_after = get_memory_usage()
    memory_used_mb = mem_after - mem_before
    logging.info(f"Model loaded. Memory usage increased by: {memory_used_mb:.2f} MB")

    # --- 3. Run Inference and Measure Time ---
    all_preds = []
    start_time = time.time()

    if model_type == 'transformer':
        # Batch inference for transformers
        test_dataset.tokenizer = tokenizer # Attach tokenizer for on-the-fly tokenization
        test_loader = DataLoader(test_dataset, batch_size=config.TRANSFORMER_CONFIG['batch_size'])
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting with {model_name}"):
                input_ids = batch['input_ids'].to(config.DEVICE)
                attention_mask = batch['attention_mask'].to(config.DEVICE)
                with torch.cuda.amp.autocast(enabled=(config.DEVICE=='cuda')):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
    else: # traditional
        all_preds = pipeline.predict(X_test)

    end_time = time.time()
    total_time = end_time - start_time
    avg_inference_time_ms = (total_time / len(X_test)) * 1000
    logging.info(f"Inference complete in {total_time:.2f}s. Average time per sample: {avg_inference_time_ms:.4f} ms")

    # --- 4. Calculate Performance Metrics ---
    metrics = compute_metrics(y_test, all_preds)
    logging.info(f"Test Set Metrics: {metrics}")

    return {
        "performance": metrics,
        "resources": {
            "memory_mb": round(memory_used_mb, 2),
            "avg_inference_ms": round(avg_inference_time_ms, 4)
        }
    }

def main():
    """
    Main function to run the evaluation on all trained models.
    """
    logging.info("====== Starting Full Model Evaluation ======")
    
    all_model_results = {}

    # Define models to evaluate
    traditional_models = ['logistic_regression', 'naive_bayes', 'svm']
    transformer_models = list(config.TRANSFORMER_CONFIG['model_names'].keys())

    # Evaluate Traditional Models
    for model_name in traditional_models:
        all_model_results[model_name] = evaluate_model(model_name, 'traditional')

    # Evaluate Transformer Models
    for model_name in transformer_models:
        all_model_results[model_name] = evaluate_model(model_name, 'transformer')

    # Save the final consolidated report
    save_json_results(all_model_results, 'model_comparison.json', config.RESULTS_DIR)
    logging.info("====== Evaluation Complete. Report saved to results/model_comparison.json ======")

if __name__ == '__main__':
    main()