"""Closing Line Value test.

The industry edge test: place bets at the OPENING price, then check whether the
market moved toward you by kickoff. Beating the closing line is the standard
evidence of genuine predictive edge — it needs no profitable season, because it
measures information rather than luck.

    CLV = opening_odds / closing_odds - 1

Positive CLV = you took a better price than the market settled at.
Benchmarked against Pinnacle's close (the sharpest available line).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE = "https://www.football-data.co.uk/mmz4281/"
OUTS = ["H", "D", "A"]


def devig(o):
    inv = 1.0 / o
    return inv / inv.sum(1, keepdims=True)


def load_season(season, div="E0"):
    d = pd.read_csv(f"{BASE}{season}/{div}.csv")
    need = ["Date", "HomeTeam", "AwayTeam", "FTR",
            "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
            "PSCH", "PSCD", "PSCA"]
    if not all(c in d.columns for c in need):
        return None
    d = d[need].dropna()
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
    return d.dropna(subset=["Date"]).reset_index(drop=True)


def clv_report(name, sel_idx, sel_out, open_o, close_o, sharp_o, results):
    """sel_out: array of chosen outcome index (0=H,1=D,2=A) per selected match."""
    if len(sel_idx) == 0:
        print(f"  {name:<34} no selections")
        return None
    oo = open_o[sel_idx, sel_out]
    cc = close_o[sel_idx, sel_out]
    ss = sharp_o[sel_idx, sel_out]
    clv_b365 = (oo / cc - 1) * 100
    clv_sharp = (oo / ss - 1) * 100
    won = (results[sel_idx] == np.array(OUTS)[sel_out])
    t, p = stats.ttest_1samp(clv_sharp, 0.0)
    print(f"  {name:<34}{len(sel_idx):>6}{clv_b365.mean():>+11.2f}%"
          f"{clv_sharp.mean():>+13.2f}%{p:>10.3f}{won.mean()*100:>9.1f}%")
    return {"n": int(len(sel_idx)), "clv_b365": float(clv_b365.mean()),
            "clv_sharp": float(clv_sharp.mean()), "p_value": float(p),
            "hit_rate": float(won.mean())}


if __name__ == "__main__":
    preds = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "predictions_2024.csv")
    preds["date"] = pd.to_datetime(preds["date"], errors="coerce")

    frames = [f for f in (load_season(s) for s in ["2324", "2425"]) if f is not None]
    mk = pd.concat(frames, ignore_index=True).rename(
        columns={"Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team"})

    df = preds.merge(mk, on=["date", "home_team", "away_team"], how="inner")
    print(f"matches with opening + closing + Pinnacle odds: {len(df)}\n")

    open_o = df[["B365H", "B365D", "B365A"]].values
    close_o = df[["B365CH", "B365CD", "B365CA"]].values
    sharp_o = df[["PSCH", "PSCD", "PSCA"]].values
    results = df["FTR"].values
    P = df[["prob_home", "prob_draw", "prob_away"]].values
    P_open = devig(open_o)

    edge = P * open_o - 1.0          # expected value per unit at the opening price
    n = len(df)

    print(f"  {'selection rule':<34}{'bets':>6}{'CLV vs B365':>12}"
          f"{'CLV vs Pinn':>13}{'p':>10}{'hit':>10}")
    print("  " + "-" * 85)

    rng = np.random.default_rng(0)
    clv_report("random outcome (control)", np.arange(n),
               rng.integers(0, 3, n), open_o, close_o, sharp_o, results)
    clv_report("always home (control)", np.arange(n),
               np.zeros(n, int), open_o, close_o, sharp_o, results)
    clv_report("model best-EV pick, all matches", np.arange(n),
               edge.argmax(1), open_o, close_o, sharp_o, results)

    for thr in [0.0, 0.05, 0.10, 0.20]:
        best = edge.argmax(1)
        idx = np.where(edge.max(1) >= thr)[0]
        clv_report(f"model EV >= {thr:.2f}", idx, best[idx],
                   open_o, close_o, sharp_o, results)

    # model disagrees with the opening favourite
    fav = P_open.argmax(1)
    pick = P.argmax(1)
    idx = np.where(pick != fav)[0]
    clv_report("model disagrees with fav", idx, pick[idx],
               open_o, close_o, sharp_o, results)
