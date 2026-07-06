# Multimodal Sports Prediction

A PyTorch research pipeline for comparing football match-outcome prediction from rolling match statistics with a multimodal model that also consumes text embeddings.

> **Project status:** experimental. The repository contains a runnable synthetic-data demo and a real-data mode. It is not presented as a production forecasting service, and no performance claim should be interpreted without reproducing the experiments.

## What is implemented

- Three-class prediction: home win, draw, or away win.
- Rolling statistical features derived from historical match results.
- A statistical sequence encoder and a 768-dimensional text-embedding input.
- Multimodal late fusion in PyTorch.
- Synthetic demo-data generation for an end-to-end sanity check.
- Real-data ingestion using historical Premier League data and `football-data.org`.
- Chronological splitting and time-series cross-validation in real-data mode.
- Confusion-matrix and prediction-distribution visualizations.

## Reproduce the demo

```bash
git clone https://github.com/chaoui-ahmed/multimodal-sports-prediction.git
cd multimodal-sports-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

The default command uses generated data. Its purpose is to verify that the pipeline executes end to end. Synthetic-demo scores are **not** evidence of real predictive performance.

## Run with real match data

```bash
cp .env.example .env
python src/main.py --real-api
```

## Evaluation

Recruiter-facing conclusions should be based on a committed experiment containing:

- dataset period and number of matches;
- chronological train/validation/test protocol;
- majority-class and statistical-only baselines;
- accuracy and macro-F1;
- per-class confusion matrix;
- statistical-only versus multimodal comparison;
- random seeds and hardware.

No headline performance number is stated until those results are committed reproducibly.

## Limitations

- Match outcomes are noisy and draws are difficult to model.
- Text timing and coverage may introduce missingness or leakage.
- Team-name matching across sources requires validation.
- The demo uses synthetic data and is only a software check.
- External APIs and websites may change.

## Author

**Ahmed Taha Chaoui** — Engineering student at EURECOM, focused on data science and machine learning.
