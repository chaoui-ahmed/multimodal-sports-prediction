import sqlite3
import pandas as pd
import os

def extract_and_merge():
    # 1. Connexion à la base SQLite de Kaggle
    db_path = "data/raw/european_database.sqlite"
    if not os.path.exists(db_path):
        print(f"❌ Erreur : Le fichier {db_path} est introuvable.")
        return

    conn = sqlite3.connect(db_path)
    print("Extraction des données historiques (SQLite)...")
    
    # On récupère les matchs de la Premier League (Code 'E0')
    query = "SELECT * FROM matchs WHERE div = 'E0'"
    df_historic = pd.read_sql_query(query, conn)
    conn.close()
    
    # 2. Renommage des colonnes pour correspondre au format de l'API
    rename_cols = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_score',
        'FTAG': 'away_score'
    }
    df_historic = df_historic.rename(columns=rename_cols)
    
    # 3. Chargement des données récentes de l'API
    api_path = "data/raw/matches_PL.csv"
    if not os.path.exists(api_path):
        print(f"❌ Erreur : Le fichier {api_path} est introuvable. Lance fetch_data.py d'abord.")
        return
        
    print("Chargement des données récentes (API)...")
    df_recent = pd.read_csv(api_path)
    
    # --- ÉTAPE 3 MODIFIÉE : Harmonisation des fuseaux horaires ---
    print("Harmonisation des formats de date (tz-naive)...")
    
    # On convertit en datetime ET on supprime toute information de fuseau horaire (.dt.tz_localize(None))
    # Cela permet de comparer les dates de l'API et du SQLite sans erreur
    df_historic['date'] = pd.to_datetime(df_historic['date']).dt.tz_localize(None)
    df_recent['date'] = pd.to_datetime(df_recent['date']).dt.tz_localize(None)
    
    # 4. Fusion et Nettoyage
    cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    
    # On concatène les deux DataFrames
    df_final = pd.concat([df_historic[cols], df_recent[cols]])
    
    # On trie par date et on supprime les doublons éventuels
    df_final = df_final.sort_values('date').drop_duplicates(subset=['date', 'home_team', 'away_team'])

    # 5. Sauvegarde du fichier final
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/full_premier_league.csv"
    df_final.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"✅ FUSION RÉUSSIE !")
    print(f"📍 Fichier créé : {output_path}")
    print(f"📊 Nombre total de matchs : {len(df_final)}")
    print(f"📅 Premier match : {df_final['date'].min().strftime('%d/%m/%Y')}")
    print(f"📅 Dernier match : {df_final['date'].max().strftime('%d/%m/%Y')}")
    print("-" * 30)

if __name__ == "__main__":
    extract_and_merge()