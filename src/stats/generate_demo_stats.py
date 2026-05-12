"""
generate_demo_stats.py
----------------------
Feature engineering pour les données de matchs de football.
Produit 8 features par match :
  home_form, home_goals_scored, home_goals_conceded, home_elo,
  away_form, away_goals_scored, away_goals_conceded, away_elo

Utilisé aussi bien en mode démo (données synthétiques)
qu'en mode réel (données Kaggle + API).
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# Features utilisées par le modèle (ordre important — doit correspondre à stats_input_size)
STAT_FEATURES = [
    'home_form',            # moyenne des points sur les 5 derniers matchs
    'home_goals_scored',    # moyenne buts marqués sur les 5 derniers
    'home_goals_conceded',  # moyenne buts concédés sur les 5 derniers
    'home_elo',             # score ELO normalisé
    'away_form',
    'away_goals_scored',
    'away_goals_conceded',
    'away_elo',
]

TEAMS = [
    "Manchester City", "Liverpool", "Chelsea", "Arsenal",
    "Manchester United", "Tottenham Hotspur", "Newcastle United",
    "Aston Villa", "Brighton & Hove Albion", "West Ham United",
    "Wolverhampton Wanderers", "Brentford", "Crystal Palace",
    "Fulham", "Leicester City", "Everton", "Nottingham Forest",
    "Bournemouth", "Luton Town", "Sheffield United"
]


# ─────────────────────────────────────────────────────
# 1. ELO
# ─────────────────────────────────────────────────────

def compute_elo(df, k=20, home_advantage=100, initial=1500):
    """
    Calcule les scores ELO pour chaque équipe match par match.
    Ajoute les colonnes home_elo et away_elo (ELO AVANT le match).
    """
    df = df.sort_values('date').reset_index(drop=True)
    elo = {}  # team -> rating courant

    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h, a = row['home_team'], row['away_team']
        r_h = elo.get(h, initial)
        r_a = elo.get(a, initial)

        home_elos.append(r_h)
        away_elos.append(r_a)

        # Résultat réel
        hs, as_ = row['home_score'], row['away_score']
        if hs > as_:
            s_h, s_a = 1.0, 0.0
        elif hs == as_:
            s_h = s_a = 0.5
        else:
            s_h, s_a = 0.0, 1.0

        # Score attendu (avec avantage domicile)
        e_h = 1 / (1 + 10 ** ((r_a - (r_h + home_advantage)) / 400))
        e_a = 1 - e_h

        elo[h] = r_h + k * (s_h - e_h)
        elo[a] = r_a + k * (s_a - e_a)

    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    return df


# ─────────────────────────────────────────────────────
# 2. Feature engineering complet (8 features)
# ─────────────────────────────────────────────────────

def calculate_rolling_form(df, window=5):
    """
    Calcule pour chaque match les 8 features stats sur fenêtre glissante.
    Corrige le bug d'indexation (mapping par index de ligne, pas par position).
    """
    df = df.sort_values('date').reset_index(drop=True)

    def pts(hs, as_, side):
        if hs == as_: return 1
        return 3 if (side == 'home' and hs > as_) or (side == 'away' and as_ > hs) else 0

    # Pré-calcul des pts / buts par match
    df['_h_pts'] = df.apply(lambda r: pts(r['home_score'], r['away_score'], 'home'), axis=1)
    df['_a_pts'] = df.apply(lambda r: pts(r['home_score'], r['away_score'], 'away'), axis=1)

    # Initialisation
    for col in STAT_FEATURES[:4]:   # home features
        df[col] = 0.0
    for col in STAT_FEATURES[4:]:   # away features
        df[col] = 0.0

    print("  Calcul des features par équipe (form, buts, ELO)...")
    teams = pd.concat([df['home_team'], df['away_team']]).unique()

    for team in teams:
        home_mask = df['home_team'] == team
        away_mask = df['away_team'] == team

        home_idx = df.index[home_mask].tolist()
        away_idx = df.index[away_mask].tolist()

        # Tous les matchs de l'équipe (chronologique)
        team_df = df[home_mask | away_mask].copy()
        team_df['_pts']   = team_df.apply(
            lambda r: r['_h_pts'] if r['home_team'] == team else r['_a_pts'], axis=1)
        team_df['_scored']    = team_df.apply(
            lambda r: r['home_score'] if r['home_team'] == team else r['away_score'], axis=1)
        team_df['_conceded']  = team_df.apply(
            lambda r: r['away_score'] if r['home_team'] == team else r['home_score'], axis=1)

        # Rolling sur les 5 matchs PRÉCÉDENTS (shift(1) pour éviter la fuite)
        roll_form      = team_df['_pts'].shift(1).rolling(window, min_periods=1).mean()
        roll_scored    = team_df['_scored'].shift(1).rolling(window, min_periods=1).mean()
        roll_conceded  = team_df['_conceded'].shift(1).rolling(window, min_periods=1).mean()

        # Mapping par index de ligne (corrige le bug d'indexation)
        for df_idx, val_form, val_scored, val_conc in zip(
                team_df.index, roll_form, roll_scored, roll_conceded):
            if df_idx in home_idx:
                df.loc[df_idx, 'home_form']           = val_form
                df.loc[df_idx, 'home_goals_scored']   = val_scored
                df.loc[df_idx, 'home_goals_conceded'] = val_conc
            else:
                df.loc[df_idx, 'away_form']           = val_form
                df.loc[df_idx, 'away_goals_scored']   = val_scored
                df.loc[df_idx, 'away_goals_conceded'] = val_conc

    # ELO (calculé sur l'ensemble)
    df = compute_elo(df)

    # Nettoyage colonnes temporaires
    df.drop(columns=['_h_pts', '_a_pts'], inplace=True, errors='ignore')

    print(f"  ✅ Features calculées : {STAT_FEATURES}")
    return df


# ─────────────────────────────────────────────────────
# 3. Génération de données synthétiques (mode démo)
# ─────────────────────────────────────────────────────

def generate_season(start_date, nb_matchdays=38):
    """Génère une saison de Premier League (matchs aller-retour round-robin)."""
    matches = []
    teams = TEAMS[:]
    random.shuffle(teams)
    current_date = start_date

    for matchday in range(nb_matchdays):
        round_teams = teams[:]
        random.shuffle(round_teams)
        paired = [(round_teams[i], round_teams[i + 1])
                  for i in range(0, len(round_teams) - 1, 2)]

        match_date = current_date + timedelta(days=matchday * 7 + random.randint(0, 2))

        for home, away in paired:
            home_score = np.random.choice([0, 1, 2, 3, 4, 5],
                                          p=[0.18, 0.30, 0.28, 0.15, 0.07, 0.02])
            away_score = np.random.choice([0, 1, 2, 3, 4, 5],
                                          p=[0.25, 0.33, 0.24, 0.12, 0.05, 0.01])
            matches.append({
                'date': match_date.strftime("%Y-%m-%d"),
                'home_team': home, 'away_team': away,
                'home_score': int(home_score), 'away_score': int(away_score)
            })

    return matches


def build_demo_dataset():
    print("🏗️  Génération des données de démo (Premier League synthétique 2020-2025)...")
    all_matches = []
    for year in range(2020, 2025):
        all_matches.extend(generate_season(datetime(year, 8, 15)))

    df = pd.DataFrame(all_matches)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/matches_PL.csv", index=False)
    print(f"  ✅ {len(df)} matchs générés → data/raw/matches_PL.csv")
    return df


def generate_processed_stats():
    """Pipeline complet en mode démo : génère les CSV train/test prêts à l'emploi."""
    df = build_demo_dataset()
    df = calculate_rolling_form(df)

    train_df = df[df['date'] < '2025-01-01'].copy()
    test_df  = df[df['date'] >= '2025-01-01'].copy()

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train_stats.csv", index=False)
    test_df.to_csv("data/processed/test_stats.csv",   index=False)

    print(f"  ✅ Train (2020-2024) : {len(train_df)} matchs")
    print(f"  ✅ Test  (2025+)     : {len(test_df)} matchs")
    return train_df, test_df


if __name__ == "__main__":
    generate_processed_stats()
