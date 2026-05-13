import sys
import os
import requests
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "src", "models"))
from fusion_model import MultimodalSportsModel

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "5fdadf02cbbb4060a089cee644997086")

def custom_predict_match(model, stats_seq_np, nlp_vec_tensor, device):
    model.eval()
    with torch.no_grad():
        stats = torch.tensor(stats_seq_np, dtype=torch.float32).unsqueeze(0).to(device)
        nlp   = nlp_vec_tensor.unsqueeze(0).to(device)
        logits = model(stats, nlp)
        probs  = torch.softmax(logits, dim=1)[0]

    labels = ["Win (domicile)", "Draw", "Loss (domicile)"]
    p_win, p_draw, p_loss = probs[0].item(), probs[1].item(), probs[2].item()
    
    if p_win >= 0.45:
        pred_idx = 0 
    elif p_loss >= 0.38 and p_loss > p_draw:
        pred_idx = 2 
    elif p_draw >= 0.32:
        pred_idx = 1
    else:
        pred_idx = torch.argmax(probs).item()

    return {
        "prediction": labels[pred_idx],
        "probabilités": {labels[i]: f"{probs[i].item()*100:.1f}%" for i in range(3)}
    }

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultimodalSportsModel().to(device)
    model.load_state_dict(torch.load("src/models/multimodal_v1.pth", map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv("data/processed/test_stats.csv")
    feat_cols = ['home_form', 'home_goals_scored', 'home_goals_conceded', 'home_elo',
                 'away_form', 'away_goals_scored', 'away_goals_conceded', 'away_elo']
    
    min_vals = df[feat_cols].min().values
    max_vals = df[feat_cols].max().values

    matches = requests.get(
        "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED",
        headers={'X-Auth-Token': API_KEY}
    ).json().get('matches', [])[:10]

    for m in matches:
        h_team, a_team = m['homeTeam']['name'], m['awayTeam']['name']
        h_data = df[(df['home_team'] == h_team) | (df['away_team'] == h_team)].tail(1)
        a_data = df[(df['home_team'] == a_team) | (df['away_team'] == a_team)].tail(1)
        
        if h_data.empty or a_data.empty:
            continue

        h_feat = h_data[['home_form', 'home_goals_scored', 'home_goals_conceded', 'home_elo']].values[0] if h_data['home_team'].values[0] == h_team else h_data[['away_form', 'away_goals_scored', 'away_goals_conceded', 'away_elo']].values[0]
        a_feat = a_data[['away_form', 'away_goals_scored', 'away_goals_conceded', 'away_elo']].values[0] if a_data['away_team'].values[0] == a_team else a_data[['home_form', 'home_goals_scored', 'home_goals_conceded', 'home_elo']].values[0]

        stats = np.concatenate([h_feat, a_feat]).astype(np.float32)
        stats_scaled = (stats - min_vals) / (max_vals - min_vals + 1e-8)
        stats_seq = np.tile(stats_scaled, (5, 1))
        
        nlp_vec = torch.zeros(768).to(device)
        pred = custom_predict_match(model, stats_seq, nlp_vec, device)
        
        print(f"{h_team} vs {a_team}\n{pred['prediction']} | {pred['probabilités']}\n")

if __name__ == "__main__":
    main()