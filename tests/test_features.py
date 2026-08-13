"""Tests for feature building and odds handling.

Main thing checked here: no feature can see the match it describes or any
later one. Look-ahead bugs are silent and make results look better than they
are, so they get tested rather than assumed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from model_v2 import build, devig  # noqa: E402


def _synthetic(n=60):
    """Fake fixtures, with a blowout in the last match so it stands out."""
    teams = ["A", "B", "C", "D"]
    rows = []
    for i in range(n):
        h, a = teams[i % 4], teams[(i + 1) % 4]
        gh, ga = (1, 0) if i % 2 else (0, 1)
        if i == n - 1:                       # extreme result, last match only
            gh, ga = 9, 0
        rows.append({
            "Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=3 * i),
            "HomeTeam": h, "AwayTeam": a, "FTHG": gh, "FTAG": ga,
            "FTR": "H" if gh > ga else ("A" if ga > gh else "D"),
            "HS": 10 + gh, "AS": 10 + ga, "HST": 4 + gh, "AST": 4 + ga,
            "HC": 5, "AC": 5,
            "B365H": 2.0, "B365D": 3.4, "B365A": 3.8, "league": "T1"})
    return pd.DataFrame(rows)


def test_devig_removes_the_overround():
    odds = np.array([[2.0, 3.4, 3.8], [1.5, 4.0, 7.0]])
    p = devig(odds)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()
    # raw implied probabilities must have summed to more than 1
    assert ((1 / odds).sum(axis=1) > 1.0).all()


def test_devig_preserves_ordering():
    odds = np.array([[1.5, 4.0, 7.0]])
    p = devig(odds)[0]
    assert p[0] > p[1] > p[2]


def test_features_do_not_use_the_current_match():
    """Changing the last match's score must not change its own features."""
    df = _synthetic()
    base = build(df.copy())

    tampered = df.copy()
    tampered.loc[tampered.index[-1], ["FTHG", "FTAG"]] = [0, 9]   # flip the blowout
    tampered.loc[tampered.index[-1], "FTR"] = "A"
    after = build(tampered)

    feat_cols = [c for c in base.columns
                 if c.startswith(("h_", "a_")) or c in ("elo_diff", "st_diff", "sh_diff")]
    assert len(base) == len(after)
    pd.testing.assert_frame_equal(
        base[feat_cols].reset_index(drop=True),
        after[feat_cols].reset_index(drop=True),
        check_exact=False, rtol=1e-9)


def test_features_do_not_use_future_matches():
    """Cutting off later matches must not change the earlier ones."""
    df = _synthetic()
    full = build(df.copy())
    part = build(df.iloc[:-5].copy())

    feat_cols = [c for c in full.columns
                 if c.startswith(("h_", "a_")) or c in ("elo_diff", "st_diff", "sh_diff")]
    n = len(part)
    pd.testing.assert_frame_equal(
        full[feat_cols].iloc[:n].reset_index(drop=True),
        part[feat_cols].reset_index(drop=True),
        check_exact=False, rtol=1e-9)


def test_warmup_rows_are_dropped():
    """Teams need history first, so early rows get dropped."""
    out = build(_synthetic(20))
    assert len(out) < 20
    assert out["usable"].all()


def test_market_probabilities_are_attached_and_normalised():
    out = build(_synthetic())
    tot = out[["mkt_h", "mkt_d", "mkt_a"]].sum(axis=1)
    assert np.allclose(tot, 1.0)


def test_rows_with_impossible_odds_are_removed():
    df = _synthetic()
    df.loc[df.index[10], "B365H"] = 0.0          # invalid price
    out = build(df)
    assert (out[["odd_home", "odd_draw", "odd_away"]] > 1.0).all().all()
    assert np.isfinite(out[["mkt_h", "mkt_d", "mkt_a"]].values).all()
