# AI Keyword Prediction System

An NLP project that predicts and suggests relevant keywords from user queries using a BERT-based model.

Built as an end-to-end personal project with:
- model training pipeline
- FastAPI inference service
- lightweight web frontend
- reproducible development and release workflow

## Project Snapshot

- Problem: Users type short, ambiguous queries and need better keyword suggestions.
- Solution: Train a BERT classifier on query-keyword pairs, then expose suggestions through an API.
- Value: Improves discoverability and search assistance with a deployable workflow.

## Key Highlights

- End-to-end ML engineering workflow from dataset to serving.
- Production-style API layer with health checks.
- Reproducible setup using pinned dependencies and Python 3.11.
- Clear Git and GitHub publishing process for personal project delivery.

## Tech Stack

- Python 3.11
- FastAPI
- Transformers
- Datasets
- PyTorch
- Pandas

## Architecture

User Query
   |
   v
FastAPI Service (app.py)
   |
   v
BERT Model (final_model/)
   |
   v
Suggested Keywords (JSON response)

## Repository Structure

~~~text
.
├── app.py                # FastAPI backend
├── training.py           # Main model training pipeline
├── pre_training.py       # Supporting data prep/training script
├── dataset.py            # Dataset utilities
├── frontend.html         # Quick test UI
├── data.csv              # Input dataset (query, keyword)
├── requirements.txt      # Dependencies
├── final_model/          # Saved model artifacts
└── results/              # Training checkpoints and outputs
~~~

## Quick Start

### 1) Create environment and install dependencies

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### 2) Verify core libraries

~~~powershell
python -c "import fastapi, numpy, datasets, transformers; print('ok')"
~~~

Expected output:
- ok

## Run Full Workflow

### 1) Train model

~~~powershell
python training.py
~~~

Expected artifacts:
- final_model
- final_model/label_mapping.json
- results

### 2) Start API server

~~~powershell
python -m uvicorn app:app --reload
~~~

Default local endpoints:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/suggest

### 3) Open frontend

- Open frontend.html in browser (or with Live Server).
- Ensure API is running on http://127.0.0.1:8000.

### 4) Smoke test API

~~~powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/suggest" -ContentType "application/json" -Body '{"query":"example"}'
~~~

Success condition:
- response contains suggested_keywords.

## API Reference

### POST /suggest

Request:

~~~json
{
  "query": "python course"
}
~~~

Example response:

~~~json
{
  "query": "python course",
  "suggested_keywords": [
    "python beginner",
    "python tutorial"
  ]
}
~~~

### GET /health

Example response:

~~~json
{
  "status": "healthy",
  "data_loaded": true
}
~~~

## Model Training Details

Main script: training.py

Pipeline:
- load and validate data.csv columns
- split train and test
- tokenize queries and encode labels
- train with Transformers Trainer
- save model and label mappings

## Results Section Template

Use this section to showcase impact when publishing:

- Dataset size: <add value>
- Number of labels: <add value>
- Best validation accuracy: <add value>
- Inference latency (local): <add value>

## Troubleshooting

Import errors in VS Code:
- select .venv/Scripts/python.exe as interpreter
- run python -m pip install -r requirements.txt
- reload VS Code window or restart language server

Python 3.13 installation issues:
- recreate environment with Python 3.11

API returns empty or poor suggestions:
- verify data.csv has query and keyword columns
- check dataset quality and label balance

## Roadmap

- add precision, recall, f1 metrics reporting
- add automated API tests
- move runtime configs to env file
- add Dockerfile for containerized deployment

## Author

Tran Minh Quan - tmqthedev

LinkedIn: https://www.linkedin.com/in/tmqthedev

GitHub: https://github.com/tmqthedev

## License

MIT License - see LICENSE file for details.

