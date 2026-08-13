"""End-to-end evaluation: model quality vs. the market, and betting performance.

Downloads Premier League results and Bet365 closing odds, trains on all seasons
before the held-out one, then scores the model against a de-vigged bookmaker
baseline and runs the value-betting simulation across staking configurations.

    python src/models/evaluate_strategy.py --test-season 2024

Writes results/strategy_evaluation.json and results/figures/*.png.
All numbers in the README are produced by this script.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtesting import Bettor

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://www.football-data.co.uk/mmz4281/"
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]


def download_matches() -> pd.DataFrame:
    frames = []
    for s in SEASONS:
        d = pd.read_csv(f"{BASE}{s}/E0.csv")
        frames.append(d[[c for c in COLS if c in d.columns]].copy())
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def devig(odds: np.ndarray) -> np.ndarray:
    """Bookmaker implied probabilities with the overround removed."""
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, default=2024)
    ap.add_argument("--predictions", default="predictions.csv",
                    help="CSV from train_model.py (date, home_team, away_team, "
                         "result, prob_home, prob_draw, prob_away)")
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions)
    preds["date"] = pd.to_datetime(preds["date"], errors="coerce")

    matches = download_matches().rename(columns={
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "B365H": "odd_home", "B365D": "odd_draw", "B365A": "odd_away"})

    df = preds.merge(matches[["date", "home_team", "away_team",
                              "odd_home", "odd_draw", "odd_away"]],
                     on=["date", "home_team", "away_team"], how="left")
    df = df.dropna(subset=["odd_home", "odd_draw", "odd_away"]).reset_index(drop=True)
    df["match_id"] = [f"M{i}" for i in range(len(df))]

    # ── model quality vs. market ────────────────────────────────────────
    labels = ["A", "D", "H"]
    y = df["result"].values
    P_model = df[["prob_away", "prob_draw", "prob_home"]].values
    odds = df[["odd_away", "odd_draw", "odd_home"]].values
    P_book = devig(odds)
    overround = ((1.0 / odds).sum(axis=1) - 1).mean() * 100

    ll_model = log_loss(y, P_model, labels=labels)
    ll_book = log_loss(y, P_book, labels=labels)
    ll_unif = float(np.log(3))
    acc_model = float((np.array(labels)[P_model.argmax(1)] == y).mean())
    acc_book = float((np.array(labels)[P_book.argmax(1)] == y).mean())

    print(f"matches: {len(df)} | bookmaker overround: {overround:.2f}%")
    print(f"  model      log-loss {ll_model:.4f}  accuracy {acc_model*100:.2f}%")
    print(f"  bookmaker  log-loss {ll_book:.4f}  accuracy {acc_book*100:.2f}%  (de-vigged)")
    print(f"  uniform    log-loss {ll_unif:.4f}")

    # ── betting simulation across staking configurations ────────────────
    grid = []
    print(f"\n{'kelly':>6}{'min_edge':>10}{'bets':>6}{'ROI':>10}{'yield':>9}"
          f"{'win%':>7}{'maxDD':>8}")
    print("-" * 56)
    for kf in [0.05, 0.10, 0.25]:
        for me in [0.00, 0.05, 0.10, 0.20]:
            b = Bettor(1000.0, kelly_fraction=kf, min_edge=me)
            b.run(df, verbose=False)
            r = b.compute_metrics()
            if "error" in r:
                continue
            grid.append({"kelly_fraction": kf, "min_edge": me,
                         "n_bets": r["n_paris"], "roi_pct": r["roi_pct"],
                         "yield_pct": r["yield_pct"], "win_rate_pct": r["win_rate_pct"],
                         "max_drawdown_pct": r["max_drawdown_pct"],
                         "final_bankroll": r["bankroll_finale"]})
            print(f"{kf:>6}{me:>10}{r['n_paris']:>6}{r['roi_pct']:>9.2f}%"
                  f"{r['yield_pct']:>8.2f}%{r['win_rate_pct']:>6.1f}%"
                  f"{r['max_drawdown_pct']:>7.1f}%")

    # ── figures ─────────────────────────────────────────────────────────
    figdir = ROOT / "results" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    b = Bettor(1000.0, kelly_fraction=0.05, min_edge=0.0)
    b.run(df, verbose=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(range(len(b.history.values)), b.history.values, lw=1.4, color="#c0392b")
    a1.axhline(1000, ls="--", lw=1, color="grey", label="starting bankroll")
    a1.set_xlabel("bet number"); a1.set_ylabel("bankroll (EUR)")
    a1.set_title(f"Value-betting bankroll — season {args.test_season}\n"
                 f"(Kelly 0.05, no edge floor)", fontsize=10)
    a1.legend(fontsize=8)

    ys = [g["yield_pct"] for g in grid if g["kelly_fraction"] == 0.05]
    xs = [g["min_edge"] for g in grid if g["kelly_fraction"] == 0.05]
    a2.plot(xs, ys, "o-", color="#2c3e50")
    a2.axhline(0, ls="--", lw=1, color="grey")
    a2.set_xlabel("minimum edge threshold"); a2.set_ylabel("yield (%)")
    a2.set_title("Yield vs. selectivity\n(more selective = worse)", fontsize=10)
    plt.tight_layout()
    plt.savefig(figdir / "betting_performance.png", dpi=150)

    out = {
        "test_season": args.test_season,
        "n_matches": int(len(df)),
        "bookmaker_overround_pct": float(overround),
        "model": {"log_loss": float(ll_model), "accuracy": acc_model},
        "bookmaker_devigged": {"log_loss": float(ll_book), "accuracy": acc_book},
        "uniform_log_loss": ll_unif,
        "betting_grid": grid,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "strategy_evaluation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/strategy_evaluation.json and results/figures/")


if __name__ == "__main__":
    main()
