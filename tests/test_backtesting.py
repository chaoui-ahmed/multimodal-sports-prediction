"""Unit tests for the staking and value-detection maths.

These are the functions that decide how much money is risked, so they are the
ones worth pinning down. Everything here is a closed-form check against the
Kelly criterion rather than a regression snapshot.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from backtesting import Bettor, generate_demo_data  # noqa: E402


# ── value detection ──────────────────────────────────────────────────────────

def test_edge_formula_matches_definition():
    """edge = p*odds - 1, and a fair bet has zero edge."""
    b = Bettor(min_edge=0.0)
    is_value, edge = b.is_value_bet(0.50, 2.00)      # exactly fair
    assert edge == pytest.approx(0.0)
    assert not is_value                              # not > implied prob

    _, edge = b.is_value_bet(0.60, 2.00)
    assert edge == pytest.approx(0.20)

    _, edge = b.is_value_bet(0.40, 2.00)
    assert edge == pytest.approx(-0.20)


def test_min_edge_threshold_is_respected():
    b = Bettor(min_edge=0.10)
    assert b.is_value_bet(0.55, 2.00)[0]             # edge 0.10, boundary is inclusive
    assert b.is_value_bet(0.56, 2.00)[0]             # edge 0.12 > 0.10
    assert not b.is_value_bet(0.52, 2.00)[0]         # edge 0.04 < 0.10


def test_no_value_when_below_implied_probability():
    b = Bettor(min_edge=0.0)
    assert not b.is_value_bet(0.30, 2.00)[0]


# ── Kelly staking ────────────────────────────────────────────────────────────

def test_kelly_matches_closed_form():
    """f* = (b*p - q)/b, scaled by the Kelly fraction and the bankroll."""
    b = Bettor(initial_bankroll=1000.0, kelly_fraction=1.0, max_bet_pct=1.0)
    p, odd = 0.60, 2.00
    expected_f = (1.0 * 0.60 - 0.40) / 1.0           # = 0.20
    assert b.kelly_stake(p, odd) == pytest.approx(expected_f * 1000.0)


def test_kelly_fraction_scales_linearly():
    full = Bettor(1000.0, kelly_fraction=1.0, max_bet_pct=1.0).kelly_stake(0.60, 2.00)
    half = Bettor(1000.0, kelly_fraction=0.5, max_bet_pct=1.0).kelly_stake(0.60, 2.00)
    assert half == pytest.approx(full * 0.5)


def test_no_stake_when_kelly_is_negative():
    """A bet with no edge must never be sized."""
    b = Bettor(1000.0, kelly_fraction=1.0)
    assert b.kelly_stake(0.40, 2.00) == 0.0          # negative Kelly
    assert b.kelly_stake(0.50, 2.00) == 0.0          # exactly zero Kelly


def test_max_bet_pct_caps_the_stake():
    b = Bettor(1000.0, kelly_fraction=1.0, max_bet_pct=0.05)
    # unconstrained Kelly here is 0.20 of bankroll; the cap must bind at 0.05
    assert b.kelly_stake(0.60, 2.00) == pytest.approx(50.0)


def test_stake_scales_with_current_bankroll():
    b = Bettor(1000.0, kelly_fraction=1.0, max_bet_pct=1.0)
    first = b.kelly_stake(0.60, 2.00)
    b.bankroll = 500.0
    assert b.kelly_stake(0.60, 2.00) == pytest.approx(first * 0.5)


def test_longshot_amplification_is_real():
    """A fixed absolute error in p implies a much larger edge at long odds.

    This is the mechanism behind the README's odds-bucket table: selecting on
    maximum edge preferentially selects the model's most inflated estimates.
    """
    b = Bettor(min_edge=0.0)
    _, edge_long = b.is_value_bet(0.08, 15.0)        # true 0.05, 3pt overestimate
    _, edge_short = b.is_value_bet(0.68, 1.5)        # true 0.65, same 3pt error
    assert edge_long > 8 * edge_short


# ── settlement ───────────────────────────────────────────────────────────────

def test_winning_and_losing_pnl():
    import pandas as pd
    b = Bettor(1000.0, kelly_fraction=1.0, min_edge=0.0, max_bet_pct=1.0)
    row = pd.Series({"prob_home": 0.60, "prob_draw": 0.01, "prob_away": 0.01,
                     "odd_home": 2.00, "odd_draw": 50.0, "odd_away": 50.0,
                     "result": "H", "date": "2025-01-01", "match_id": "M1",
                     "home_team": "A", "away_team": "B"})
    bet = b.process_match(row)
    assert bet.won and bet.pnl == pytest.approx(bet.stake * (bet.odd - 1))
    assert b.bankroll == pytest.approx(1000.0 + bet.pnl)

    row["result"] = "A"
    b2 = Bettor(1000.0, kelly_fraction=1.0, min_edge=0.0, max_bet_pct=1.0)
    bet2 = b2.process_match(row)
    assert not bet2.won and bet2.pnl == pytest.approx(-bet2.stake)


def test_metrics_are_internally_consistent():
    b = Bettor(1000.0, kelly_fraction=0.05, min_edge=0.0)
    b.run(generate_demo_data(120, seed=7), verbose=False)
    m = b.compute_metrics()
    assert m["n_paris"] == m["n_gagnes"] + m["n_perdus"]
    assert m["bankroll_finale"] == pytest.approx(
        m["bankroll_initiale"] + m["profit_net"])
    assert m["roi_pct"] == pytest.approx(
        (m["bankroll_finale"] - 1000.0) / 1000.0 * 100)
    assert m["yield_pct"] == pytest.approx(
        m["profit_net"] / m["volume_total_mise"] * 100)
    assert 0.0 <= m["win_rate_pct"] <= 100.0


def test_run_is_deterministic():
    df = generate_demo_data(80, seed=3)
    a = Bettor(1000.0, kelly_fraction=0.05); a.run(df, verbose=False)
    c = Bettor(1000.0, kelly_fraction=0.05); c.run(df, verbose=False)
    assert a.bankroll == pytest.approx(c.bankroll)
