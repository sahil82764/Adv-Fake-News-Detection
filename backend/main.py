import os, sys
from fastapi import FastAPI
from backend.api import predict, compare, ensemble

app = FastAPI(title="Fake News Detection API", version="1.0")

# Include prediction, comparison, and ensemble endpoints
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(compare.router, prefix="/api/compare", tags=["Comparison"])
app.include_router(ensemble.router, prefix="/api/ensemble", tags=["Ensemble"])

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
