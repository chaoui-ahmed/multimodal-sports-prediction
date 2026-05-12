import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "5fdadf02cbbb4060a089cee644997086")
HEADERS = {'X-Auth-Token': API_KEY}

def save_matches_to_csv(league_code="PL"):
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    print(f"Téléchargement des données pour {league_code}...")
    
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        matches = data['matches']
        
        # Extraction des colonnes importantes
        df_list = []
        for m in matches:
            df_list.append({
                'match_id': m['id'],
                'date': m['utcDate'],
                'home_team': m['homeTeam']['name'],
                'away_team': m['awayTeam']['name'],
                'home_score': m['score']['fullTime']['home'],
                'away_score': m['score']['fullTime']['away'],
                'winner': m['score']['winner']
            })
        
        df = pd.DataFrame(df_list)
        
        # Créer le dossier data/raw s'il n'existe pas
        os.makedirs("data/raw", exist_ok=True)
        
        # Sauvegarde
        file_path = f"data/raw/matches_{league_code}.csv"
        df.to_csv(file_path, index=False)
        print(f"✅ Fichier sauvegardé : {file_path}")
    else:
        print(f"❌ Erreur {response.status_code}")

if __name__ == "__main__":
    save_matches_to_csv("PL")
    