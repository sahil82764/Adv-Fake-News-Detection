import os
import json
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

def compute_metrics(preds, labels):
    """
    Computes and returns a dictionary of classification metrics, including the confusion matrix.
    """
    # Ensure labels are numpy arrays for metric calculations
    labels = np.array(labels)
    preds = np.array(preds)

    # Standard metrics
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    precision = precision_score(labels, preds, average='binary', zero_division=0)
    recall = recall_score(labels, preds, average='binary', zero_division=0)
    accuracy = accuracy_score(labels, preds)

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels, preds)
    # Convert numpy array to a standard Python list of lists for JSON serialization
    cm_list = cm.tolist()

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {
            "labels": ["Fake (0)", "Real (1)"],
            "matrix": cm_list
        }
    }

def save_json_results(results, filename, results_dir):
    """Saves a dictionary of results to a JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, 'w') as f:
        # Use a custom encoder to handle numpy types if they slip through
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        json.dump(results, f, indent=4, cls=NpEncoder)
    logging.info(f"Results saved to {path}")
