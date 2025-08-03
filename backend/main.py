import os, sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import predict, compare, ensemble

app = FastAPI(title="Fake News Detection API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include prediction, comparison, and ensemble endpoints
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(compare.router, prefix="/api/compare", tags=["Comparison"])
app.include_router(ensemble.router, prefix="/api/ensemble", tags=["Ensemble"])

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
