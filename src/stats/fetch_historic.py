"""
fetch_historic.py
-----------------
Télécharge la base european_database.sqlite depuis Kaggle
et extrait les matchs de Premier League (Div='E0').
Retourne un DataFrame avec les colonnes standardisées.
"""

import kagglehub
import sqlite3
import pandas as pd
import os

KAGGLE_DATASET = "groleo/european-football-database"
DB_DEST        = "data/raw/european_database.sqlite"


def download_and_extract(league="E0") -> pd.DataFrame:
    """
    Télécharge (ou réutilise le cache) le dataset Kaggle,
    extrait les matchs de la ligue demandée, renvoie un DataFrame propre.
    """
    # 1. Téléchargement (kagglehub gère le cache automatiquement)
    print(f"📦 Téléchargement du dataset Kaggle ({KAGGLE_DATASET})...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    sqlite_file = next(f for f in os.listdir(path) if f.endswith(".sqlite"))
    src_path = os.path.join(path, sqlite_file)

    os.makedirs("data/raw", exist_ok=True)
    import shutil
    shutil.copy(src_path, DB_DEST)
    print(f"  ✅ Base copiée → {DB_DEST}")

    # 2. Extraction
    conn = sqlite3.connect(DB_DEST)
    df = pd.read_sql_query(
        f"SELECT Date, HomeTeam, AwayTeam, FTHG, FTAG, season FROM matchs WHERE Div = '{league}'",
        conn
    )
    conn.close()

    # 3. Renommage + nettoyage
    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_score",
        "FTAG": "away_score",
    })
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    seasons = sorted(df["season"].unique())
    print(f"  ✅ {len(df)} matchs PL extraits | saisons : {seasons[0]}→{seasons[-1]}")
    return df


if __name__ == "__main__":
    df = download_and_extract("E0")
    print(df.head())