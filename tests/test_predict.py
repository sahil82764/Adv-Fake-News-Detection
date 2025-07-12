import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)

def test_predict_default():
    response = client.post("/api/predict/default", json={"text": "NASA confirms the discovery of water on Mars."})
    assert response.status_code == 200
    assert response.json()["predictions"][0] in ["Fake", "Real"]

def test_predict_single():
    response = client.post("/api/predict/single", json={"text": "NASA confirms the discovery of water on Mars.", "model": "distilbert"})
    assert response.status_code == 200
    assert response.json()["predictions"][0] in ["Fake", "Real"]

def test_predict_batch():
    response = client.post("/api/predict/batch", json={"texts": ["NASA confirms the discovery of water on Mars."], "model": "distilbert"})
    assert response.status_code == 200
    assert response.json()["predictions"][0] in ["Fake", "Real"]

def test_invalid_model():
    response = client.post("/api/predict/single", json={"text": "Test", "model": "invalid_model"})
    assert response.status_code == 400
    assert "Model loading failed" in response.json()["detail"]

def test_missing_text():
    response = client.post("/api/predict/default", json={})
    assert response.status_code == 422

def test_async_batch():
    # Start async batch
    response = client.post("/api/predict/async_batch", json={"texts": ["NASA confirms the discovery of water on Mars."], "model": "distilbert"})
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    # Poll for result
    import time
    for _ in range(10):
        result = client.get(f"/api/predict/async_batch_result/{task_id}")
        if result.status_code == 200 and "predictions" in result.json():
            assert result.json()["predictions"][0] in ["Fake", "Real"]
            break
        time.sleep(0.5)
    else:
        pytest.fail("Async batch result not ready in time")
