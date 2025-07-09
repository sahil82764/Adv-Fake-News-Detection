import os
import json
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(preds, labels):
    """
    Computes and returns a dictionary of classification metrics.
    Args:
        preds (list or np.array): The predicted labels.
        labels (list or np.array): The true labels.
    Returns:
        dict: A dictionary containing accuracy, f1_score, precision, and recall.
    """
    # Using zero_division=0 to avoid warnings when a class has no predictions.
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def save_json_results(results, filename, results_dir):
    """Saves a dictionary of results to a JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {path}")