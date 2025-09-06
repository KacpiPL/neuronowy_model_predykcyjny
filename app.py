from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path

ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "model.joblib"
SCALER_PATH = ART_DIR / "scaler.joblib"

app = FastAPI(title="Iris NN Predictor (MLPClassifier)")
templates = Jinja2Templates(directory="templates")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Prediction(BaseModel):
    predicted_class: int
    probabilities: List[float]

# --- ENDPOINTY ---

# Redirect z / do /ui (formularz)
@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/ui")

# Formularz HTML
@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Informacyjny JSON
@app.get("/info")
def info():
    classes = [int(c) for c in model.classes_]
    return {"message": "Iris predictor działa!", "classes": classes}

# Predykcja
@app.post("/predict", response_model=Prediction)
def predict(item: IrisRequest):
    x = np.array([[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]])
    x_sc = scaler.transform(x)
    probs = model.predict_proba(x_sc)[0]
    clazz = int(model.predict(x_sc)[0])
    return Prediction(predicted_class=clazz, probabilities=[float(p) for p in probs])