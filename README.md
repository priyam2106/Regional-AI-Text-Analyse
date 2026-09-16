<<<<<<< HEAD
# Verity: Hindi AI-text detector

This project now includes a real training pipeline and a local prediction API. It starts with Hindi because CT² AG_hi contains paired human and AI-generated Hindi news text. The classifier uses XLM-RoBERTa, a multilingual encoder that can later be extended with balanced, labelled data for additional Indic languages.

## Train locally

1. Create and activate a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Train the CPU-friendly, real Hindi baseline: `python train_baseline.py`
4. Optionally run a small transformer validation: `python train.py --max-examples 500 --epochs 1`
5. Fine-tune the transformer on a CUDA GPU: `python train.py --epochs 2 --batch-size 8`
6. Start the website and API: `python -m uvicorn server:app --reload --port 8000`

The first training run downloads the public CT² AG_hi dataset. The baseline uses TF-IDF plus logistic regression and runs on a CPU; transformer fine-tuning also downloads `FacebookAI/xlm-roberta-base` and needs a CUDA GPU. Trained artifacts are saved to `model/` and excluded from git.

## Scope

The pipeline separates original articles between training and test data, so an article and its generated variants cannot leak across the split. It is a research starting point, not universal proof of authorship: it may be unreliable for short passages, newer generators, edited text, non-news domains, and languages absent from its training data.

After training, the site calls `POST /api/analyze`. Until then, it clearly falls back to the local demo estimate.

## Deploy to Render

The repository includes `render.yaml`, so you can deploy it as a Blueprint:

1. Push this project, including `model/detector.joblib`, to a GitHub repository.
2. In Render, choose **New** → **Blueprint**, connect the repository, and approve the detected `verity-hindi-detector` service.
3. Render installs `requirements-runtime.txt` and starts the FastAPI app using the platform-provided port.

The deployed site serves both the interface and `/api/analyze` from the same service. Do not use the free service for sensitive text: it is a public, shared runtime and may spin down when idle.
=======
# Regional-AI-Text-Analyse
>>>>>>> 75e3bf09509e2c491acd802f3121cfc79c65146d
