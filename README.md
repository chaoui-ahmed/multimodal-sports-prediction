# ⚽ Multimodal Sports Prediction Pipeline

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)]()
[![Model](https://img.shields.io/badge/NLP_Encoder-RoBERTa-brightgreen.svg)]()
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-blueviolet.svg)]()

A multimodal deep learning architecture designed to predict association football match outcomes (Home Win, Draw, Away Win). The pipeline fuses tabular statistics (historical form, rolling averages) with text embeddings extracted from sports media reports to capture both quantitative team performance and qualitative sentiment.

---

## 🏗️ Architecture Overview

The framework combines two data modalities:

1. **Statistical Modality**:
   - Rolling team statistics (goals scored, goals conceded, points per match over a sliding window).
   - Historical Premier League match outcomes scraped or downloaded from Kaggle.
   - Processed via fully connected layers and recurrent networks (LSTMs).

2. **Textual Modality (NLP)**:
   - Scraped pre-match news articles and analyst reports.
   - Embeddings generated using a fine-tuned **RoBERTa** model to encode public sentiment, injuries, and team momentum.

3. **Late Fusion**:
   - The features from the statistical encoder and NLP encoder are fused via dense layers to predict a final softmax distribution over match outcomes (Win, Draw, Loss).

---

## 📁 Repository Structure

```
├── requirements.txt            # Python environment dependencies
├── predict.py / graph.py       # Quick prediction and visualization scripts
│
├── src/                        # Main source code directory
│   ├── main.py                 # Orchestrates the entire training pipeline
│   │
│   ├── stats/                  # Modality 1: Tabular stats extraction & processing
│   │   ├── fetch_historic.py   # Downloads historical matches (Kaggle)
│   │   ├── fetch_data.py       # Fetches active season matches from football APIs
│   │   ├── preprocessing.py    # Standardizing tabular data and rolling stats
│   │   └── generate_demo_stats.py
│   │
│   ├── nlp/                    # Modality 2: Sports news NLP processing
│   │   ├── scraper.py          # Pre-match sports report web scraper
│   │   ├── roberta_model.py    # Custom RoBERTa classifier for embedding extraction
│   │   └── nlp_train.py        # Sentiment model training script
│   │
│   └── models/                 # Deep Learning models (PyTorch)
│       ├── fusion_model.py     # Multimodal Fusion Neural Network architecture
│       ├── fusion_dataloader.py# Standard dataloaders for multi-modal formats
│       ├── train.py            # Model training & optimization loops
│       └── evaluate_and_plot.py# Custom confusion matrix & loss curves generator
│
└── reports/                    # Generated charts and performance metrics
    └── figures/                # Confusion matrices, distributions, and loss curves
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chaoui-ahmed/multimodal-sports-prediction-main.git
   cd multimodal-sports-prediction-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

You can run the pipeline in two modes:

1. **Demo Mode** (runs instantly with simulated/mock statistical data for verification):
   ```bash
   python src/main.py
   ```

2. **Real Production Mode** (downloads real Premier League data, calculates rolling forms, crawls pre-match articles, and runs training/evaluation):
   ```bash
   # Add your FOOTBALL_DATA_API_KEY environment variable to a .env file or export it:
   export FOOTBALL_DATA_API_KEY="your_api_key"
   python src/main.py --real-api
   ```

---

## 📊 Evaluation Results
Training outputs automatically generate visual analysis reports under `reports/figures/`, including:
- **Loss curves comparison** between unimodal statistical and multimodal architectures.
- **Confusion matrices** indicating performance across Win/Draw/Loss classes.
- **Probability distribution graphs** highlighting decision confidence.

---

## ✍️ Author
- **Ahmed Chaoui** — Engineering Student at Eurecom
