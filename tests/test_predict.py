
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import pytest
from fastapi.testclient import TestClient
from backend.main import app

# Setup logging
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, 'test_predict.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)

client = TestClient(app)


def test_predict_default():
    logging.info("Running test_predict_default...")
    response = client.post("/api/predict/default", json={"text": "NASA confirms the discovery of water on Mars."})
    logging.info(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["predictions"][0] in ["Fake", "Real"]
    logging.info("test_predict_default passed.")


def test_predict_single():
    logging.info("Running test_predict_single...")
    response = client.post("/api/predict/single", json={"text": "NASA confirms the discovery of water on Mars.", "model": "distilbert"})
    logging.info(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["predictions"][0] in ["Fake", "Real"]
    logging.info("test_predict_single passed.")


def test_predict_batch():
    logging.info("Running test_predict_batch...")
    response = client.post("/api/predict/batch", json={"texts": ["NASA confirms the discovery of water on Mars."], "model": "distilbert"})
    logging.info(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["predictions"][0] in ["Fake", "Real"]
    logging.info("test_predict_batch passed.")


def test_invalid_model():
    logging.info("Running test_invalid_model...")
    response = client.post("/api/predict/single", json={"text": "Test", "model": "invalid_model"})
    logging.info(f"Response: {response.json()}")
    assert response.status_code == 400
    assert "Model loading failed" in response.json()["detail"]
    logging.info("test_invalid_model passed.")


def test_missing_text():
    logging.info("Running test_missing_text...")
    response = client.post("/api/predict/default", json={})
    logging.info(f"Response: {response.json()}")
    assert response.status_code == 422
    logging.info("test_missing_text passed.")


def test_async_batch():
    logging.info("Running test_async_batch...")
    # Start async batch
    response = client.post("/api/predict/async_batch", json={"texts": ["NASA confirms the discovery of water on Mars."], "model": "distilbert"})
    logging.info(f"Async batch start response: {response.json()}")
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    # Poll for result
    import time
    for _ in range(10):
        result = client.get(f"/api/predict/async_batch_result/{task_id}")
        logging.info(f"Polling async batch result: {result.json()}")
        if result.status_code == 200 and "predictions" in result.json():
            assert result.json()["predictions"][0] in ["Fake", "Real"]
            logging.info("test_async_batch passed.")
            break
        time.sleep(0.5)
    else:
        logging.error("Async batch result not ready in time")
        pytest.fail("Async batch result not ready in time")
