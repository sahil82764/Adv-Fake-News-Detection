import os
import logging
import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib

# This ensures the script can be run from anywhere by adding the project root to the path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from our project
from ml.dataset import FakeNewsDataset
from ml.utils import compute_metrics, save_json_results
from ml import config

# --- Setup Logging ---
os.makedirs(config.LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(config.LOGS_DIR, 'train_traditional.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)

def train_and_evaluate():
    """
    Trains, evaluates, and saves traditional machine learning models based on the project config.
    """
    logging.info("--- Starting Traditional ML Model Training ---")

    # --- 1. Load Data ---
    # We don't need a tokenizer for sklearn models. The dataset class will return raw text.
    logging.info("Loading training and validation data...")
    train_dataset = FakeNewsDataset(
        data_path=os.path.join(config.DATA_DIR, 'train.parquet'),
        text_column=config.TRADITIONAL_ML_CONFIG['text_column']
    )
    val_dataset = FakeNewsDataset(
        data_path=os.path.join(config.DATA_DIR, 'validation.parquet'),
        text_column=config.TRADITIONAL_ML_CONFIG['text_column']
    )

    # The dataset __getitem__ returns a dict; we extract text and labels into simple lists
    X_train = [item['text'] for item in train_dataset]
    y_train = [item['labels'].item() for item in train_dataset]

    X_val = [item['text'] for item in val_dataset]
    y_val = [item['labels'].item() for item in val_dataset]

    logging.info(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples.")

    # --- 2. Define Models and Pipelines ---
    models = {
        'logistic_regression': LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000),
        'naive_bayes': MultinomialNB(),
        'svm': LinearSVC(random_state=config.RANDOM_SEED, max_iter=2000, dual=True)
    }

    vectorizer = TfidfVectorizer(**config.TRADITIONAL_ML_CONFIG['vectorizer_params'])

    results = {}

    # --- 3. Train and Evaluate Each Model ---
    for name, model in models.items():
        logging.info(f"--- Training {name} ---")

        # Create a pipeline that first vectorizes the text and then applies the classifier
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)

        logging.info(f"Evaluating {name} on validation data...")
        y_pred = pipeline.predict(X_val)

        metrics = compute_metrics(y_val, y_pred)
        results[name] = metrics
        logging.info(f"Validation Metrics for {name}: {metrics}")

        os.makedirs(config.MODELS_DIR, exist_ok=True)
        model_path = os.path.join(config.MODELS_DIR, f'{name}_model.joblib')
        joblib.dump(pipeline, model_path)
        logging.info(f"Saved trained model pipeline to {model_path}")

    results_filename = 'traditional_ml_results.json'
    save_json_results(results, results_filename, config.RESULTS_DIR)

    logging.info(f"--- All models trained. ---")

if __name__ == '__main__':
    train_and_evaluate()