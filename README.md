# Multimodal Sports Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)]()

A late-fusion PyTorch pipeline for three-way football match prediction (home win / draw / away win), combining a sequential encoder over rolling match statistics with sentence-level text embeddings from `xlm-roberta-base`.

Two evaluation paths are included: a real-data pipeline benchmarked against Bet365 closing odds ([Results](#results)), and a synthetic-data smoke test that runs end-to-end with no credentials ([Demo mode](#demo-mode)).

## Architecture

```
rolling match stats  ──▶  LSTM encoder  ──┐
  [seq_len=5, 8 features]                 ├──▶  concat ──▶ MLP head ──▶ 3-way softmax
match-adjacent news  ──▶  xlm-roberta ────┘
  [CLS, 768-dim]           Linear 768→256→128
```

The 8 statistical features per match, computed **only from matches before kickoff**:
`home_form`, `home_goals_scored`, `home_goals_conceded`, `home_elo`, and the four away equivalents. Form and goal averages use a 5-match rolling window; ELO is computed sequentially in date order with a home-advantage term.

Each match is paired with the text embedding of the nearest news article within a configurable window (default 30 days). Matches with no article in range receive a zero vector.

## Methodology notes

- **Chronological splits.** Train/validation/test are split by date, never randomly, so no future information leaks into training. ELO and rolling form are computed sequentially in match order for the same reason.
- **Zero-vector fallback is a real risk.** If the embedding files are missing, every NLP vector is zero and the model silently degrades to stats-only while still calling itself multimodal. The pipeline now prints a loud warning in that case — do not report such a run as a multimodal result.

## Results

Held-out season 2024/25 (504 Premier League matches), trained on 2015/16–2023/24.
Every number below is produced by `src/models/evaluate_strategy.py` and stored in
`results/strategy_evaluation.json`.

### Model quality

Feature matrix is 2,326-dimensional: 20 tabular features (rolling form, head-to-head,
season points) + 2,304 historical team-embedding features + 2 news-sentiment features.
Model selection is by log-loss rather than accuracy, since calibrated probabilities are
what the staking rule consumes.

| Model | Accuracy | Log-loss |
|---|---|---|
| **MLP (128, 64)** | **54.17%** | **0.9723** |
| Logistic regression | 52.98% | 0.9807 |
| Random forest | 52.78% | 0.9981 |
| Gradient boosting | 51.79% | 1.0052 |

Benchmarked against the market on the same 504 matches:

| | Log-loss | Accuracy |
|---|---|---|
| Uniform prior | 1.0986 | — |
| **This model (MLP)** | **0.9723** | 54.17% |
| Bet365 closing odds, de-vigged | 0.9427 | 55.95% |

The model is well clear of the uniform baseline and lands within 0.030 nats of the
de-vigged closing line — but the bookmaker is still better on both metrics. Mean
overround across the season is 5.44%.

### Betting simulation

Value bets are placed when `P_model > 1/Odds`, staked by fractional Kelly.

| Kelly | Min edge | Bets | ROI | Yield | Win rate | Max drawdown |
|---|---|---|---|---|---|---|
| 0.05 | 0.00 | 389 | −42.70% | −24.82% | 32.4% | 42.7% |
| 0.05 | 0.05 | 361 | −43.06% | −26.06% | 29.9% | 43.1% |
| 0.05 | 0.10 | 299 | −43.80% | −31.25% | 25.1% | 43.8% |
| 0.05 | 0.20 | 194 | −44.05% | −41.92% | 17.5% | 44.0% |
| 0.10 | 0.00 | 417 | −67.60% | −26.44% | 33.6% | 67.6% |
| 0.25 | 0.00 | 379 | −94.50% | −33.23% | 33.2% | 94.5% |

![Betting performance](results/figures/betting_performance.png)

**The strategy does not beat the market.** Two things drive this, and the second is the
more interesting one:

1. The model's probabilities are worse-calibrated than the de-vigged closing line
   (0.9723 vs 0.9427). Betting into a better-informed counterparty loses the overround
   by construction.
2. **Yield gets monotonically worse as the edge threshold rises** — from −24.8% at no
   floor to −41.9% at a 20% floor. If the model held genuine edge, filtering to its
   highest-confidence disagreements with the market should *improve* yield. It does the
   opposite, which says those large apparent edges are miscalibration, not signal.

Being close to the closing line on log-loss is a respectable result for a model built
from public match statistics and news text. Being close is not the same as being ahead,
and only being ahead is profitable.

## Demo mode

`python src/main.py` runs the full pipeline end-to-end on synthetic data with no
credentials or downloads required. Note that `src/stats/generate_demo_stats.py` draws
match scores independently of the teams involved, so the demo data carries no signal and
the model correctly collapses to the majority class (46% accuracy = the base rate). It is
a smoke test for the pipeline, not an experiment — use the real-data path above for
results.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your football-data.org API key
```

A free key is available at [football-data.org](https://www.football-data.org/client/register). Scripts that need it now fail with a clear message rather than falling back to a bundled key.

## Running

Demo pipeline, no credentials or downloads required — synthetic data, end to end:

```bash
python src/main.py
```

Generate text embeddings (downloads ~5,900 articles from the Transfermarkt news archive on Hugging Face and runs `xlm-roberta-base` over them; several minutes on CPU):

```bash
python src/nlp/nlp_train.py
```

Fetch real match data (requires `FOOTBALL_DATA_API_KEY`):

```bash
python src/stats/fetch_data.py
```

Interactive checks:

```bash
python test.py --all
```

Full real-data evaluation (downloads results + odds, trains, backtests):

```bash
python src/models/train_model.py --matches matches_PL.csv \
    --historical-embeddings data/historical_embeddings.pt \
    --news-embeddings data/news_embeddings.pt \
    --model auto --test-season 2024 \
    --output-predictions predictions.csv

python src/models/evaluate_strategy.py --test-season 2024 --predictions predictions.csv
```

## Repository structure

```text
src/
  main.py                      end-to-end demo pipeline
  models/
    fusion_model.py            late-fusion architecture, training loop, predict_match()
    fusion_dataloader.py       dataset, date-based article matching, chronological splits
    train_model.py             tabular + embedding features, model comparison, exports predictions
    backtesting.py             value-bet detection, fractional Kelly, ROI/yield/drawdown metrics
    evaluate_strategy.py       model vs. de-vigged market + staking grid -> results/
  nlp/
    nlp_train.py               generates data/historical_embeddings.pt
    roberta_model.py           embedding utilities
    scraper.py                 RSS headline collection
  stats/
    generate_demo_stats.py     synthetic demo data + feature engineering (ELO, form)
    fetch_data.py              football-data.org ingestion
    preprocessing.py           feature assembly
    merge_sources.py           joins statistical and text sources
predict.py                     single-match inference
test.py                        interactive pipeline checks
```

Datasets and generated embeddings live in `data/` and are git-ignored.

## Limitations

- Demo data is synthetic noise; no real predictive result is committed to this repository.
- Results are from a single held-out season (504 matches) with one seed — no cross-season variance estimate or confidence intervals.
- The simulation assumes every bet is placed at the listed closing price with unlimited liquidity and no account limits, which overstates achievable returns.
- The text branch matches articles to matches by date proximity only — the nearest article is not necessarily *about* either team.
- Single train/val/test split, single seed; no cross-validation or confidence intervals.
- `src/models/multimodal_v1.pth` and `lstm_v1.pth` are checkpoints from demo runs and carry no predictive value.

## License

MIT — see [LICENSE](LICENSE).
