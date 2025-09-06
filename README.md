# Iris Neural Network Predictor (FastAPI + scikit-learn)

This project demonstrates how to train and serve a simple neural network model (MLP classifier from scikit-learn) as a REST API using **FastAPI**.  
It also includes a minimal web UI (HTML form) to make predictions interactively.

## Project Structure

```plaintext
neuronowy_model_predykcyjny/
├── artifacts/           # trained model and scaler are saved here
│   ├── model.joblib
│   └── scaler.joblib
├── templates/           # HTML templates for the web UI
│   └── index.html
├── app.py               # FastAPI app serving predictions and the UI
├── train.py             # script to train the model and save artifacts
├── pyproject.toml       # project dependencies (for uv)
├── uv.lock              # lock file with pinned dependencies
└── .gitignore

## Model

- **Dataset:** [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) (from scikit-learn).  
- **Preprocessing:** features are standardized using `StandardScaler`.  
- **Model:** `MLPClassifier` from scikit-learn (a simple feedforward neural network).  
  - hidden layer size: 16 neurons  
  - max iterations: 1000  
  - random_state: 42  

The trained model (`model.joblib`) and scaler (`scaler.joblib`) are stored in the `artifacts/` folder.

## Installation and Usage

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### 1. Create a virtual environment and install dependencies:

    uv venv  
    uv sync  

### 2. Train the model:

    uv run python train.py  

This will generate the artifacts/ folder with the trained model and scaler. The basic model is already trained, so you can skip this step.

### 3. Run the FastAPI app:

    uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000

## Endpoints

- GET / → redirects to the web UI (/ui)  
- GET /ui → interactive HTML form for predictions  
- POST /predict → predict class for given flower measurements  

Example payload:

    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }

- GET /info → returns available classes in JSON

## Web UI

Accessible at http://localhost:8000/ui  
It provides a simple form where you can input the 4 Iris features and get predictions with class probabilities.
