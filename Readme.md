# Advanced Fake News Detection System

This project is a comprehensive system for detecting fake news using a variety of machine learning and deep learning models. It allows for the training, evaluation, and comparison of multiple architectures, from traditional ML baselines to advanced transformer-based hybrid models.

## Project Overview

The core of this project is a comparative framework designed to analyze the effectiveness of different NLP techniques for misinformation detection under specific hardware constraints (8GB RAM, 2GB GPU). The system is served via a FastAPI backend and includes a multi-feature Next.js frontend for interaction and analysis.

### Key Features:

- **Multi-Model Architecture:** Implements and compares:
    - [cite_start]**Transformer Models:** DistilBERT and ALBERT [cite: 230]
    - [cite_start]**Hybrid Models:** Transformer + BiGRU/CNN layers [cite: 234]
    - [cite_start]**Traditional ML:** Naive Bayes, SVM, Logistic Regression [cite: 237]
    - [cite_start]**Deep Learning:** Standalone LSTM and XGBoost models [cite: 241]
- [cite_start]**Advanced Data Pipeline:** Includes optional data augmentation with GPT-2, advanced text preprocessing, and memory-efficient data loaders[cite: 212, 222, 223].
- [cite_start]**Comprehensive Evaluation:** Provides detailed analysis of model performance (F1-score, ROC-AUC), inference speed, and memory usage[cite: 260, 265].
- [cite_start]**Interactive Frontend:** A web interface built with Next.js for single-instance prediction, side-by-side model comparison, and performance analytics[cite: 292].
- [cite_start]**Zero-Cost Deployment:** Designed to be deployed entirely on free-tier services like Vercel and Railway[cite: 15].

## Project Structure

The project is organized into a modular structure to separate concerns for machine learning, backend, and frontend development. See the `docs/` directory for detailed architecture diagrams.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18.0+
- `pip` and `virtualenv`
- `npm` or `yarn`

### Installation & Setup

1.  **Clone the repository and create the virtual environment.**
2.  **Install PyTorch with CUDA support** (see official PyTorch website for instructions).
3.  **Install Python and Node.js dependencies** using `requirements.txt` and `package.json`.
4.  **Download Datasets:** Place the Kaggle and LIAR datasets into the `data/raw/` directory as specified in the documentation.

*For detailed instructions, please refer to the `docs/TRAINING.md` and `docs/DEPLOYMENT.md` files.*

---
*This project was developed by Sahil Khan.*
