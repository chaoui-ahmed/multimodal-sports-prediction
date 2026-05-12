import sqlite3
import pandas as pd

def load_from_sqlite(db_path):
    # Connexion à la base de données que tu as téléchargée
    conn = sqlite3.connect(db_path)
    
    # On liste les tables disponibles pour trouver celle des matchs
    # Généralement, elle s'appelle 'Match' ou 'Matches'
    query = "SELECT * FROM Match WHERE league_id = (SELECT id FROM League WHERE name = 'England Premier League')"
    
    df_historic = pd.read_sql_query(query, conn)
    conn.close()
    
    # Nettoyage rapide pour correspondre à ton API
    df_historic = df_historic[['date', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal']]
    # Note : Il faudra transformer les IDs en noms d'équipes (Man City, etc.)
    
    return df_historic