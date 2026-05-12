---
title: Customer Support Intelligence
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.57.0"
python_version: "3.10"
app_file: app/main.py
pinned: false
---

# AI-Powered Customer Support Intelligence Platform

This repository contains an end-to-end customer support intelligence application.
It includes synthetic dataset generation, model training pipelines for ticket classification, priority prediction, and resolution time estimation, and a Streamlit inference application.

## Features
- Synthetic customer support ticket dataset generator
- Text preprocessing with TF-IDF
- Ticket type classification with Logistic Regression
- Ticket priority prediction with Random Forest
- Resolution time regression with Gradient Boosting
- Streamlit web application for interactive predictions
- Git LFS support for large model files
- Docker containerization for deployment

## Project Structure
- `data/sample_data_generator.py` — create sample ticket dataset for development
- `train.py` — train models and save artifacts into `models/`
- `predict.py` — local CLI prediction example
- `app/main.py` — Streamlit application for inference
- `src/customer_support_intelligence/` — reusable preprocessing, model, and inference modules

## Getting Started
### Install dependencies
```bash
python -m pip install -r requirements.txt
```

### Generate sample data
```bash
python data/sample_data_generator.py --output data/sample_tickets.csv --rows 5000
```

### Train models
```bash
python train.py --data-path data/sample_tickets.csv --model-dir models
```

### Run inference locally
```bash
python predict.py --model-dir models
```

### Start the API
```bash
uvicorn app.main:app --reload --port 8000
```

Then POST to `http://127.0.0.1:8000/predict` with a JSON payload.

## API Example
```json
{
  "ticket_subject": "Unable to access billing page",
  "ticket_description": "I keep getting an error when I try to update my credit card info.",
  "ticket_channel": "Email",
  "product_purchased": "Cloud CRM",
  "customer_age": 34,
  "customer_gender": "Female",
  "date_of_purchase": "2025-08-15",
  "first_response_time": 1.2
}
```

## Docker
Build the container:
```bash
docker build -t support-intelligence-api .
```
Run it:
```bash
docker run -p 8000:8000 support-intelligence-api
```

## Notes
- The repository is designed as a production-grade scaffold that can be extended with real Kaggle data.
- The training pipeline is modular and reproducible with fixed random seeds.
- Advanced transformer fine-tuning and MLflow experiment tracking are included as optional extension points.
