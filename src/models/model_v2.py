"""Improved match-outcome model.

Three changes over the baseline:
  1. Shot-based features (shots, shots on target, corners) instead of goals only.
     Goals are a noisy, low-count signal; shot volume stabilises much faster.
  2. Venue-split form — teams perform differently home vs away, so rolling windows
     are kept separately for each venue.
  3. Market anchoring — the de-vigged opening price enters as a feature, so the
     model starts from everything the market knows and can only add to it.

(3) makes the test clean: if the model beats the market's log-loss on held-out
data, the improvement is information the price did not contain.

Protocol: train <= 2022/23, validate 2023/24, test 2024/25. Reported once.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

BASE = "https://www.football-data.co.uk/mmz4281/"
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
LEAGUES = ["E0", "E1", "E2", "E3", "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "N1", "B1", "P1"]
NEED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HC", "AC", "B365H", "B365D", "B365A"]
W = 6  # rolling window


def load():
    frames = []
    for lg in LEAGUES:
        for s in SEASONS:
            try:
                d = pd.read_csv(f"{BASE}{s}/{lg}.csv")
            except Exception:
                continue
            if not all(c in d.columns for c in NEED):
                continue
            d = d[NEED].dropna().copy()
            d["league"] = lg
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def devig(o):
    inv = 1.0 / o
    return inv / inv.sum(1, keepdims=True)


def build(df):
    """Rolling venue-split form. Every feature uses only prior matches."""
    home_hist, away_hist = defaultdict(list), defaultdict(list)
    elo, last_played = {}, {}
    rows = []

    for r in df.itertuples(index=False):
        h, a = r.HomeTeam, r.AwayTeam
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)

        def agg(hist, key):
            x = hist[key][-W:]
            if len(x) < 3:
                return None
            m = np.mean(x, axis=0)
            return m  # pts, gf, ga, sf, sa, stf, sta, cf, ca

        fh, fa = agg(home_hist, h), agg(away_hist, a)
        if fh is None or fa is None:
            ok = False
            fh = fh if fh is not None else np.zeros(9)
            fa = fa if fa is not None else np.zeros(9)
        else:
            ok = True

        rest_h = (r.Date - last_played.get(h, r.Date)).days
        rest_a = (r.Date - last_played.get(a, r.Date)).days

        rows.append(dict(
            date=r.Date, league=r.league, result=r.FTR,
            odd_home=r.B365H, odd_draw=r.B365D, odd_away=r.B365A, usable=ok,
            h_pts=fh[0], h_gf=fh[1], h_ga=fh[2], h_sf=fh[3], h_sa=fh[4],
            h_stf=fh[5], h_sta=fh[6], h_cf=fh[7], h_ca=fh[8],
            a_pts=fa[0], a_gf=fa[1], a_ga=fa[2], a_sf=fa[3], a_sa=fa[4],
            a_stf=fa[5], a_sta=fa[6], a_cf=fa[7], a_ca=fa[8],
            elo_diff=eh + 60 - ea,
            st_diff=(fh[5] - fh[6]) - (fa[5] - fa[6]),
            sh_diff=(fh[3] - fh[4]) - (fa[3] - fa[4]),
            rest_diff=np.clip(rest_h - rest_a, -14, 14),
        ))

        gh, ga_, hs, as_, hst, ast, hc, ac = (
            r.FTHG, r.FTAG, r.HS, r.AS, r.HST, r.AST, r.HC, r.AC)
        ph = 3 if gh > ga_ else (1 if gh == ga_ else 0)
        pa = 3 if ga_ > gh else (1 if gh == ga_ else 0)
        home_hist[h].append([ph, gh, ga_, hs, as_, hst, ast, hc, ac])
        away_hist[a].append([pa, ga_, gh, as_, hs, ast, hst, ac, hc])
        last_played[h] = last_played[a] = r.Date

        e = 1 / (1 + 10 ** (-((eh + 60 - ea) / 400)))
        s = 1.0 if gh > ga_ else (0.5 if gh == ga_ else 0.0)
        elo[h], elo[a] = eh + 20 * (s - e), ea + 20 * ((1 - s) - (1 - e))

    out = pd.DataFrame(rows)
    out = out[out.usable].reset_index(drop=True)
    for c in ["odd_home", "odd_draw", "odd_away"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[(out[["odd_home", "odd_draw", "odd_away"]] > 1.0).all(axis=1)]
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    B = devig(out[["odd_home", "odd_draw", "odd_away"]].values)
    out["mkt_h"], out["mkt_d"], out["mkt_a"] = B[:, 0], B[:, 1], B[:, 2]
    out["mkt_lh"] = np.log(B[:, 0] / B[:, 2])
    return out


FEATS = [c for c in
         ["h_pts", "h_gf", "h_ga", "h_sf", "h_sa", "h_stf", "h_sta", "h_cf", "h_ca",
          "a_pts", "a_gf", "a_ga", "a_sf", "a_sa", "a_stf", "a_sta", "a_cf", "a_ca",
          "elo_diff", "st_diff", "sh_diff", "rest_diff",
          "mkt_h", "mkt_d", "mkt_a", "mkt_lh"]]
LBL = ["A", "D", "H"]


def score(name, y, P, B):
    ll = log_loss(y, P, labels=LBL)
    llb = log_loss(y, B, labels=LBL)
    print(f"  {name:<22} model {ll:.4f}   market {llb:.4f}   diff {ll-llb:+.4f}"
          f"   {'MODEL BETTER' if ll < llb else ''}")
    return ll, llb


if __name__ == "__main__":
    print("downloading…")
    raw = load()
    print(f"raw matches: {len(raw)}  leagues: {raw.league.nunique()}")
    X = build(raw)
    print(f"usable after warm-up: {len(X)}")

    tr = X[X.date < "2023-07-01"]
    va = X[(X.date >= "2023-07-01") & (X.date < "2024-07-01")]
    te = X[X.date >= "2024-07-01"]
    print(f"train {len(tr)} | validation {len(va)} | test {len(te)}\n")

    clf = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(LogisticRegression(max_iter=4000, C=0.05), cv=3))
    clf.fit(tr[FEATS], tr.result)
    cls = list(clf.classes_)
    assert cls == LBL, cls

    for nm, part in [("VALIDATION 2023/24", va), ("TEST 2024/25", te)]:
        P = clf.predict_proba(part[FEATS])
        B = part[["mkt_a", "mkt_d", "mkt_h"]].values
        score(nm, part.result.values, P, B)
        np.save(f"P_{nm.split()[0]}.npy", P)
    va.to_csv("v2_val.csv", index=False)
    te.to_csv("v2_test.csv", index=False)
