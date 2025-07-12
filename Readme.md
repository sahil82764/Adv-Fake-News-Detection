# Advanced Fake News Detection System

## Overview
This project is an academic implementation of a robust Fake News Detection system using both traditional machine learning and modern transformer-based models. It features a FastAPI backend for model inference and analytics, and a Next.js frontend for interactive exploration, comparison, and visualization of model performance.

## Features
- **Multiple Model Support:**
  - Traditional ML: Logistic Regression, Naive Bayes, SVM
  - Transformers: DistilBERT, ALBERT
- **Interactive Web UI:**
  - Single and batch prediction
  - Model comparison and analytics
  - Confusion matrix visualization
  - Educational help page
- **REST API:**
  - Predict, compare, and benchmark models via HTTP endpoints
- **Extensible:**
  - Easily add new models or metrics

## Project Structure
```
Adv-Fake-News-Detection/
├── backend/           # FastAPI backend (API, model loading, inference)
├── frontend/          # Next.js frontend (UI, analytics, help)
├── ml/                # ML utilities, config, preprocessing
├── models/            # Saved model weights and tokenizers
├── data/              # Raw and processed datasets
├── results/           # Model performance metrics and logs
├── notebooks/         # Jupyter notebooks for exploration and benchmarking
├── requirements.txt   # Python dependencies
├── pyproject.toml     # Python project config
└── README.md          # Project documentation
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-org-or-username>/Adv-Fake-News-Detection.git
cd Adv-Fake-News-Detection
```

### 2. Backend Setup (FastAPI)
- Install Python 3.8+
- Create a virtual environment and activate it:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Run the backend locally:
  ```bash
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
  ```
- API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Frontend Setup (Next.js)
- Go to the frontend directory:
  ```bash
  cd frontend
  ```
- Install dependencies:
  ```bash
  npm install
  ```
- Run the frontend locally:
  ```bash
  npm run dev
  ```
- Access the UI at: [http://localhost:3000](http://localhost:3000)

### 4. Deployment
- **Backend:** Deploy FastAPI to Render, Railway, or any cloud platform. Set the `PYTHONPATH` to `.` and ensure all `__init__.py` files exist in backend folders.
- **Frontend:** Deploy Next.js to Vercel. Set the `BACKEND_URL` environment variable in Vercel to your deployed backend URL.

## API Endpoints
- `POST /api/predict/single` — Predict label for a single news article
- `POST /api/predict/batch` — Predict labels for multiple articles
- `POST /api/compare/models` — Compare predictions from multiple models
- `GET /api/compare/performance` — Get model performance metrics
- `GET /api/health` — Health check

See `/docs` on the backend for full OpenAPI documentation.

## Data
- Uses Kaggle Fake News and LIAR datasets (see `data/` folder)
- Preprocessing scripts in `ml/preprocess.py` and `scripts.py/prepare_data.py`

## Notebooks
- Data exploration, model training, and benchmarking in `notebooks/`

## Educational Content
- Help page in the frontend explains model types and evaluation metrics
- Tooltips and popups throughout the UI

## Acknowledgements
- HuggingFace Transformers
- Scikit-learn
- FastAPI, Uvicorn
- Next.js, React
- Kaggle and LIAR datasets

## License
This project is for academic and educational purposes only. Please cite appropriately if used in research or coursework.

---

For questions or suggestions, please open an issue or pull request on GitHub.
