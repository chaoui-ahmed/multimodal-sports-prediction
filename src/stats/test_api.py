import requests
import os
from dotenv import load_dotenv

# 1. On charge la clé depuis le .env
load_dotenv()
api_key = os.getenv("FOOTBALL_DATA_API_KEY")
headers = {'X-Auth-Token': api_key}

# 2. On définit l'URL (ici on demande les matchs de la Premier League)
url = "https://api.football-data.org/v4/competitions/PL/matches"

# 3. On fait l'appel
print("Connexion à l'API en cours...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    nb_matchs = len(data['matches'])
    print(f"Succès ! On a récupéré {nb_matchs} matchs.")
    # On affiche le premier match pour voir à quoi ça ressemble
    premier_match = data['matches'][0]
    print(f"Exemple : {premier_match['homeTeam']['name']} vs {premier_match['awayTeam']['name']}")
else:
    print(f"Erreur {response.status_code} : {response.text}")