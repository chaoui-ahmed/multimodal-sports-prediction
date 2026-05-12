import pandas as pd
import numpy as np
import os

def calculate_rolling_form(df):
    """ Calcule la forme des équipes (points sur les 5 derniers matchs) """
    df = df.sort_values('date')
    
    # Calcul des points par match
    def get_pts(score_h, score_a, side):
        if score_h == score_a: return 1
        if side == 'home':
            return 3 if score_h > score_a else 0
        else:
            return 3 if score_a > score_h else 0

    df['home_pts'] = df.apply(lambda r: get_pts(r['home_score'], r['away_score'], 'home'), axis=1)
    df['away_pts'] = df.apply(lambda r: get_pts(r['home_score'], r['away_score'], 'away'), axis=1)

    # Calcul de la moyenne glissante (Rolling average) pour chaque équipe
    # C'est ici que l'IA apprend la "dynamique"
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    team_forms = {}

    print("Calcul de la forme glissante pour chaque équipe...")
    
    # On initialise des colonnes vides
    df['home_form'] = 0.0
    df['away_form'] = 0.0

    for team in teams:
        # On récupère tous les matchs de l'équipe
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        
        # On calcule ses points au fil du temps
        points = team_matches.apply(lambda r: r['home_pts'] if r['home_team'] == team else r['away_pts'], axis=1)
        
        # Forme = moyenne des 5 matchs PRÉCÉDENTS (shift(1) pour ne pas tricher avec le futur)
        rolling_form = points.shift(1).rolling(window=5, min_periods=1).mean()
        
        # On réinjecte dans le DF principal
        df.loc[df['home_team'] == team, 'home_form'] = rolling_form
        df.loc[df['away_team'] == team, 'away_form'] = rolling_form

    return df

if __name__ == "__main__":
    input_file = "data/processed/full_premier_league.csv"
    
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Feature Engineering
        df_with_form = calculate_rolling_form(df)
        
        # 2. Split Chronologique (2025 pour le test)
        train_df = df_with_form[df_with_form['date'] < '2025-01-01']
        test_df = df_with_form[df_with_form['date'] >= '2025-01-01']
        
        # 3. Sauvegarde
        train_df.to_csv("data/processed/train_stats.csv", index=False)
        test_df.to_csv("data/processed/test_stats.csv", index=False)
        
        print(f"✅ Preprocessing terminé !")
        print(f"🏠 Train (2005-2024) : {len(train_df)} matchs")
        print(f"🚀 Test (2025+) : {len(test_df)} matchs")
    else:
        print("❌ Fichier full_premier_league.csv introuvable.")