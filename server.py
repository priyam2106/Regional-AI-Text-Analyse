"""API and static web server for a locally trained Verity model."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
app = FastAPI(title="Verity Detector API", version="1.0.0")

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=30, max_length=5000)
    language: str = "auto"

@lru_cache
def model_bundle():
    if (MODEL_DIR / "detector.joblib").exists():
        return "baseline", joblib.load(MODEL_DIR / "detector.joblib")
    if (MODEL_DIR / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        return "transformer", (tokenizer, model)
    return None

@app.get("/")
def homepage():
    return FileResponse(ROOT / "index.html")

@app.get("/{asset_name}")
def asset(asset_name: str):
    if asset_name not in {"app.js", "styles.css"}:
        raise HTTPException(404, "Not found")
    return FileResponse(ROOT / asset_name)

@app.get("/api/status")
def status():
    return {"trained_model_available": model_bundle() is not None}

@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    bundle = model_bundle()
    if bundle is None:
        raise HTTPException(503, "No trained model found. Run train.py first.")
    kind, artifact = bundle
    if kind == "baseline":
        probability = artifact["pipeline"].predict_proba([request.text])[0, 1]
        model_name = artifact["metadata"]["model_type"]
    else:
        tokenizer, model = artifact
        encoded = tokenizer(request.text, truncation=True, max_length=384, return_tensors="pt")
        with torch.no_grad():
            probability = torch.softmax(model(**encoded).logits, dim=-1)[0, 1].item()
        model_name = "xlm-roberta"
    return {"ai_probability": round(probability * 100, 1), "mode": "trained", "model": model_name, "notice": "This is a model estimate, not proof of authorship."}
