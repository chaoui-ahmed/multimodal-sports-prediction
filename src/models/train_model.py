"""
train_model.py
──────────────────────────────────────────────────────────────────
Pipeline complet d'entraînement pour la prédiction de matchs de foot.

Ce script intègre TROIS sources de features :
  1. Features tabulaires classiques (forme, H2H, points de saison)
  2. historical_embeddings.pt → représentation vectorielle STATIQUE
     de chaque équipe, apprise sur l'historique des matchs.
     → "quelle équipe est-ce, quel est son profil de jeu ?"
  3. news_embeddings.pt → représentation vectorielle DYNAMIQUE
     des actualités récentes avant chaque match.
     → "quoi de neuf avant ce match ?" (blessures, moral, suspensions)

Architecture des features finale par match :
    [tabular (20) | embed_home (768) | embed_away (768) | embed_diff (768)
     | news_home_sentiment (1) | news_away_sentiment (1)]
    = ~2 330 dimensions

Étapes :
  1. Charge et nettoie les matchs
  2. Construit les features tabulaires (forme, H2H, saison)
  3. Charge historical_embeddings.pt → features équipe par équipe
  4. Charge news_embeddings.pt → features sentiment pré-match
  5. Concatène tout → matrice X finale
  6. Entraîne + compare les modèles (log-loss comme métrique clé)
  7. Sauvegarde le meilleur → models/model.pkl
  8. Exporte les prédictions → data/processed/predictions.csv

Usage :
    python train_model.py --matches data/raw/matches_PL.csv
    python train_model.py \\
        --matches data/raw/matches_PL.csv \\
        --historical-embeddings data/embeddings/historical_embeddings.pt \\
        --news-embeddings       news_embeddings.pt \\
        --model xgboost \\
        --test-season 2025

Format CSV attendu :
    match_id, date, home_team, away_team,
    home_score, away_score, result (H/D/A)
"""

import os
import re
import argparse
import warnings
import zipfile
import struct
import io
import math
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Sklearn ───────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    log_loss, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

# XGBoost optionnel
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost non installé (pip install xgboost). Utilisation de GradientBoosting.")


# ─────────────────────────────────────────────────────────────────
#  CHARGEUR DE FICHIERS .pt (sans torch)
# ─────────────────────────────────────────────────────────────────

def _load_pt_raw(pt_path: str) -> tuple[np.ndarray, list, list]:
    """
    Charge un fichier .pt PyTorch sans avoir besoin d'installer torch.
    Retourne (embeddings: np.ndarray, labels: list[str], dates: list[str|None])

    Format interne PyTorch : ZIP contenant data.pkl + data/0 (float32 LE).
    On lit les métadonnées via une extraction de chaînes ASCII depuis le pickle,
    et le tenseur directement depuis les octets bruts.
    """
    with open(pt_path, 'rb') as f:
        raw = f.read()
    z = zipfile.ZipFile(io.BytesIO(raw))

    # Nom du dossier racine dans le zip (= nom de la variable sauvegardée)
    root = z.namelist()[0].split('/')[0]
    pkl_bytes    = z.read(f'{root}/data.pkl')
    tensor_bytes = z.read(f'{root}/data/0')

    # ── Extraction des chaînes lisibles ──────────────────────────
    strings = re.findall(rb'[\x20-\x7e]{4,}', pkl_bytes)
    decoded = [s.decode('ascii', errors='replace').strip() for s in strings]

    # ── Reconstruction du tenseur ─────────────────────────────────
    n_floats = len(tensor_bytes) // 4
    floats = struct.unpack(f'<{n_floats}f', tensor_bytes)
    flat = np.array(floats, dtype=np.float32)

    # Détermination dimension : normes stables → bonne dim
    best_dim, best_std = 768, 1e9
    for dim in [768, 384, 512, 256]:
        if n_floats % dim == 0:
            n = n_floats // dim
            mat = flat[:n * dim].reshape(n, dim)
            norms = np.linalg.norm(mat, axis=1)
            std = float(np.std(norms))
            if std < best_std:
                best_dim, best_std = dim, std

    embed_dim = best_dim
    n_vecs = n_floats // embed_dim
    embeddings = flat[:n_vecs * embed_dim].reshape(n_vecs, embed_dim)

    # ── Extraction des labels et dates ────────────────────────────
    labels, dates = [], []
    current_section = None
    skip_tokens = {'titres', 'sources', 'dates', 'embeddings',
                   'Foot Mercato (France)', 'BBC Sport (UK)',
                   'ctorch._utils', '_rebuild_tensor_v2', 'storage',
                   'ctorch', 'FloatStorage', 'ccollections', 'OrderedDict'}
    for tok in decoded:
        if tok in ('titres', 'sources', 'dates', 'embeddings', 'teams'):
            current_section = tok
            continue
        if tok in skip_tokens:
            continue
        if current_section in ('titres', 'teams'):
            if len(tok) > 3 and not re.match(
                r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),', tok
            ):
                labels.append(tok)
        elif current_section == 'dates':
            if re.match(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),', tok):
                try:
                    dt = datetime.strptime(tok.strip(), '%a, %d %b %Y %H:%M:%S %z')
                    dates.append(dt.strftime('%Y-%m-%d'))
                except Exception:
                    dates.append(None)

    # Padding
    while len(labels) < n_vecs:
        labels.append(f'item_{len(labels)}')
    while len(dates) < n_vecs:
        dates.append(None)

    return embeddings, labels[:n_vecs], dates[:n_vecs]


# ─────────────────────────────────────────────────────────────────
#  MODULE A : HistoricalEmbeddingFeatures
#  Charge historical_embeddings.pt → vecteur fixe par équipe
# ─────────────────────────────────────────────────────────────────

class HistoricalEmbeddingFeatures:
    """
    Fournit des features d'embedding STATIQUES par équipe.

    historical_embeddings.pt contient un vecteur par équipe, appris
    sur l'historique complet (style de jeu, niveau, profil).

    Pour chaque match, on concatène :
        [embed_home (D) | embed_away (D) | embed_diff (D)]
    = 3×D features supplémentaires (D = dimension du modèle d'embedding).

    Si le fichier n'est pas fourni, renvoie des zéros (mode dégradé).
    """

    def __init__(self, pt_path: str = None):
        self.available = False
        self.embed_dim = 0
        self.team_vectors = {}

        if pt_path is None or not os.path.exists(pt_path):
            print("  ⚠️  historical_embeddings.pt absent → features historiques désactivées")
            return

        print(f"  📂 Chargement historical_embeddings.pt...")
        embeddings, labels, _ = _load_pt_raw(pt_path)

        self.embed_dim = embeddings.shape[1]

        # Normalisation L2 pour stabiliser les distances cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms

        # Mapping label → vecteur normalisé
        for label, vec in zip(labels, embeddings_norm):
            self.team_vectors[label.lower().strip()] = vec

        self.available = True
        print(f"  ✅ {len(self.team_vectors)} équipes chargées (dim={self.embed_dim})")

    def n_features(self) -> int:
        """Nombre de features générées par match (3 × embed_dim)."""
        return self.embed_dim * 3 if self.available else 0

    def _get_team_vector(self, team_name: str) -> np.ndarray:
        """
        Cherche le vecteur d'une équipe avec matching flou.
        Si non trouvée, retourne un vecteur nul (équipe inconnue).
        """
        key = team_name.lower().strip()

        # 1. Match exact
        if key in self.team_vectors:
            return self.team_vectors[key]

        # 2. Match partiel (ex: "Manchester City FC" → "man city")
        for stored_key, vec in self.team_vectors.items():
            if stored_key in key or key in stored_key:
                return vec
            # Matching sur tokens (ex: "psg" dans "paris saint-germain")
            stored_tokens = set(stored_key.split())
            key_tokens = set(key.split())
            if stored_tokens & key_tokens:  # intersection non vide
                return vec

        # 3. Non trouvé → vecteur nul
        return np.zeros(self.embed_dim, dtype=np.float32)

    def get_match_features(self, home_team: str, away_team: str) -> np.ndarray:
        """
        Retourne [embed_home | embed_away | embed_diff] pour un match.
        embed_diff = embed_home - embed_away capture l'avantage relatif.
        """
        if not self.available:
            return np.array([], dtype=np.float32)

        h = self._get_team_vector(home_team)
        a = self._get_team_vector(away_team)
        diff = h - a
        return np.concatenate([h, a, diff])

    def build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Construit la matrice de features historiques pour tout un DataFrame."""
        if not self.available:
            return np.zeros((len(df), 0), dtype=np.float32)

        rows = [
            self.get_match_features(row['home_team'], row['away_team'])
            for _, row in df.iterrows()
        ]
        return np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
#  MODULE B : NewsEmbeddingFeatures
#  Charge news_embeddings.pt → sentiment pré-match par équipe
# ─────────────────────────────────────────────────────────────────

# Mots-clés pour le scoring de sentiment (identiques à news_odds_adjuster.py)
_POS_KW = {
    'retour': .08, 'revient': .07, 'fit': .06, 'returns': .06, 'reprend': .07,
    'victoire': .06, 'gagne': .06, 'wins': .06, 'imbattu': .08, 'unbeaten': .08,
    'finale': .07, 'qualifi': .06, 'prolonge': .04, 'optimiste': .05,
}
_NEG_KW = {
    'bless': -.10, 'injury': -.10, 'absent': -.09, 'forfait': -.10,
    'suspendu': -.08, 'suspension': -.08, 'ban': -.08, 'doute': -.06,
    'crise': -.08, 'tension': -.07, 'clash': -.07, 'insulte': -.09,
    'licenci': -.08, 'sacked': -.08, 'défaite': -.05, 'sombr': -.06,
    'dilemme': -.05, 'failles': -.06, 'col\u00e8re': -.06, 'furieux': -.06,
}
_TEAM_ALIASES = {
    'psg': ['paris', 'parisien'], 'bayern': ['munich', 'bavaria'],
    'real madrid': ['madrid', 'real'], 'barcelona': ['barça', 'barca'],
    'arsenal': ['gunners'], 'manchester city': ['man city'],
    'manchester united': ['man utd', 'man united'],
    'liverpool': ['reds'], 'chelsea': ['blues'],
    'marseille': ['om', 'phoc'], 'lyon': ['ol'],
}


class NewsEmbeddingFeatures:
    """
    Fournit des features de sentiment DYNAMIQUES depuis news_embeddings.pt.

    Pour chaque match, on cherche les actualités récentes mentionnant
    les équipes concernées (fenêtre de N jours avant le match) et on
    calcule deux scores de sentiment : home_sentiment, away_sentiment.

    Ces 2 scalaires s'ajoutent aux features tabulaires et historiques.
    Ils capturent les signaux courts-terme (blessures, suspensions, moral).

    Si le fichier n'est pas fourni, renvoie [0, 0] (mode dégradé).
    """

    def __init__(self, pt_path: str = None, news_window_days: int = 7):
        self.available = False
        self.news_window_days = news_window_days
        self.articles = []
        self.date_index = defaultdict(list)

        if pt_path is None or not os.path.exists(pt_path):
            print("  ⚠️  news_embeddings.pt absent → features news désactivées")
            return

        print(f"  📂 Chargement news_embeddings.pt...")
        embeddings, titres, dates = _load_pt_raw(pt_path)

        for i, (titre, date) in enumerate(zip(titres, dates)):
            art = {'titre': titre, 'date': date, 'embedding': embeddings[i]}
            self.articles.append(art)
            if date:
                self.date_index[date].append(art)

        self.available = True
        print(f"  ✅ {len(self.articles)} articles chargés")

    def n_features(self) -> int:
        """2 features : home_sentiment, away_sentiment."""
        return 2 if self.available else 0

    def _mentions_team(self, titre: str, team: str) -> bool:
        key = team.lower()
        titre_l = titre.lower()
        if key in titre_l:
            return True
        for canonical, aliases in _TEAM_ALIASES.items():
            if key in (canonical,) + tuple(aliases):
                all_variants = [canonical] + aliases
                return any(v in titre_l for v in all_variants)
        # Matching token
        for tok in key.split():
            if len(tok) > 3 and tok in titre_l:
                return True
        return False

    def _sentiment(self, titre: str) -> float:
        tl = titre.lower()
        score = sum(w for kw, w in _POS_KW.items() if kw in tl)
        score += sum(w for kw, w in _NEG_KW.items() if kw in tl)
        return float(np.clip(score, -1.0, 1.0))

    def get_match_features(
        self, home_team: str, away_team: str, match_date: str
    ) -> np.ndarray:
        """
        Retourne [home_sentiment, away_sentiment] pour un match.
        Calcul sur les articles publiés dans les N jours précédant le match.
        """
        if not self.available:
            return np.zeros(2, dtype=np.float32)

        try:
            match_dt = datetime.strptime(str(match_date)[:10], '%Y-%m-%d')
        except Exception:
            return np.zeros(2, dtype=np.float32)

        # Collecte des articles dans la fenêtre temporelle
        articles_window = []
        for delta in range(self.news_window_days + 1):
            d = (match_dt - timedelta(days=delta)).strftime('%Y-%m-%d')
            articles_window.extend(self.date_index.get(d, []))

        if not articles_window:
            return np.zeros(2, dtype=np.float32)

        home_scores, away_scores = [], []
        for art in articles_window:
            titre = art['titre']
            if self._mentions_team(titre, home_team):
                home_scores.append(self._sentiment(titre))
            if self._mentions_team(titre, away_team):
                away_scores.append(self._sentiment(titre))

        h_sent = float(np.mean(home_scores)) if home_scores else 0.0
        a_sent = float(np.mean(away_scores)) if away_scores else 0.0
        return np.array([h_sent, a_sent], dtype=np.float32)

    def build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Construit la matrice de features news pour tout un DataFrame."""
        if not self.available:
            return np.zeros((len(df), 0), dtype=np.float32)

        rows = [
            self.get_match_features(
                row['home_team'], row['away_team'],
                str(row['date'])[:10]
            )
            for _, row in df.iterrows()
        ]
        return np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
#  ÉTAPE 1 : CHARGEMENT ET NETTOYAGE
# ─────────────────────────────────────────────────────────────────

def load_and_clean(matches_path: str) -> pd.DataFrame:
    """
    Charge le CSV de matchs et normalise les colonnes.
    """
    df = pd.read_csv(matches_path)

    # 1. On force tous les noms de colonnes en minuscules immédiatement
    # Cela règle définitivement le problème Date vs date, FTR vs ftr, etc.
    df.columns = [c.lower() for c in df.columns]

    # 2. Normalisation du champ résultat
    # football-data.org -> 'winner'
    if 'winner' in df.columns and 'result' not in df.columns:
        winner_map = {'HOME_TEAM': 'H', 'AWAY_TEAM': 'A', 'DRAW': 'D'}
        df['result'] = df['winner'].map(winner_map)

    # Kaggle/Football-data.co.uk -> 'ftr', 'fthg', 'ftag', 'hometeam'
    # (Note: ils sont en minuscules grâce à l'étape 1)
    if 'ftr' in df.columns and 'result' not in df.columns:
        df['result'] = df['ftr']
    
    if 'fthg' in df.columns and 'home_score' not in df.columns:
        df['home_score'] = df['fthg']
        df['away_score'] = df['ftag']
        
    if 'hometeam' in df.columns and 'home_team' not in df.columns:
        df['home_team'] = df['hometeam']
        df['away_team'] = df['awayteam']

    # 3. Normalisation de la date (maintenant c'est 'date' en minuscule)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # 4. Nettoyage des données manquantes
    # On vérifie que les colonnes indispensables existent avant le dropna
    cols_to_check = ['date', 'result', 'home_score', 'away_score']
    # On ne garde que les colonnes qui sont réellement présentes dans le df
    existing_cols = [c for c in cols_to_check if c in df.columns]
    
    df = df.dropna(subset=existing_cols)
    df = df[df['result'].isin(['H', 'D', 'A'])]
    
    # Tri par date
    df = df.sort_values('date').reset_index(drop=True)

    # 5. Conversion numérique
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(int)
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(int)
    df['goal_diff'] = df['home_score'] - df['away_score']

    print(f"  ✅ {len(df)} matchs chargés ({df['date'].min().date()} → {df['date'].max().date()})")
    print(f"     Résultats : H={sum(df.result=='H')} D={sum(df.result=='D')} A={sum(df.result=='A')}")
    
    return df
# ─────────────────────────────────────────────────────────────────
#  ÉTAPE 2 : FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

def compute_form(df: pd.DataFrame, n_games: int = 5) -> pd.DataFrame:
    """
    Calcule pour chaque match les features de forme récente des deux équipes.

    Features générées (pour home et away) :
      - form_pts_X      : points sur les N derniers matchs (victoire=3, nul=1, défaite=0)
      - form_gf_X       : buts marqués en moyenne sur N derniers
      - form_ga_X       : buts encaissés en moyenne sur N derniers
      - form_wins_X     : nb de victoires sur N derniers
      - home_advantage  : différentiel de performance domicile vs extérieur
    """
    df = df.copy()

    # Initialisation des colonnes
    for prefix in ['home_', 'away_']:
        for col in [f'form_pts', f'form_gf', f'form_ga', f'form_wins', f'form_clean_sheets']:
            df[f'{prefix}{col}'] = np.nan

    # Pour chaque équipe, on garde un historique glissant
    team_history = {}  # team → liste de (date, pts, gf, ga)

    for idx, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        date = row['date']

        # ── Points du match ──
        if row['result'] == 'H':
            home_pts, away_pts = 3, 0
        elif row['result'] == 'D':
            home_pts, away_pts = 1, 1
        else:
            home_pts, away_pts = 0, 3

        # ── Calcul des features AVANT ce match ──
        for team, prefix, pts, gf, ga in [
            (home, 'home_', home_pts, row['home_score'], row['away_score']),
            (away, 'away_', away_pts, row['away_score'], row['home_score']),
        ]:
            if team in team_history and len(team_history[team]) > 0:
                hist = team_history[team][-n_games:]
                h_pts   = [h['pts'] for h in hist]
                h_gf    = [h['gf'] for h in hist]
                h_ga    = [h['ga'] for h in hist]

                df.at[idx, f'{prefix}form_pts']          = np.mean(h_pts)
                df.at[idx, f'{prefix}form_gf']           = np.mean(h_gf)
                df.at[idx, f'{prefix}form_ga']           = np.mean(h_ga)
                df.at[idx, f'{prefix}form_wins']         = sum(p == 3 for p in h_pts) / len(h_pts)
                df.at[idx, f'{prefix}form_clean_sheets'] = sum(g == 0 for g in h_ga) / len(h_ga)

            # Mise à jour de l'historique APRÈS utilisation
            if team not in team_history:
                team_history[team] = []
            team_history[team].append({'date': date, 'pts': pts, 'gf': gf, 'ga': ga})

    return df


def compute_head_to_head(df: pd.DataFrame, n_games: int = 5) -> pd.DataFrame:
    """
    Calcule les statistiques de confrontations directes (head-to-head).

    Features :
      - h2h_home_wins  : taux de victoire de l'équipe domicile en H2H
      - h2h_draw_rate  : taux de nuls en H2H
      - h2h_goals_diff : différentiel de buts moyen en H2H
    """
    df = df.copy()
    df['h2h_home_wins'] = np.nan
    df['h2h_draw_rate'] = np.nan
    df['h2h_goals_diff'] = np.nan

    h2h_history = {}  # (home, away) ou (away, home) → liste de résultats

    for idx, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        key1 = (home, away)
        key2 = (away, home)

        past = h2h_history.get(key1, []) + [
            {'result': r, 'gd': g}
            for r, g in zip(
                h2h_history.get(key2, [{'result': None}])[-n_games:],
                h2h_history.get(key2, [{'gd': 0}])[-n_games:]
            )
            if isinstance(r, dict)
        ] if not h2h_history.get(key2) else []

        # Simplification : on cherche juste les matchs de cette paire
        past_direct = h2h_history.get(key1, [])[-n_games:]

        if past_direct:
            results = [p['result'] for p in past_direct]
            diffs   = [p['gd'] for p in past_direct]
            df.at[idx, 'h2h_home_wins']  = results.count('H') / len(results)
            df.at[idx, 'h2h_draw_rate']  = results.count('D') / len(results)
            df.at[idx, 'h2h_goals_diff'] = np.mean(diffs)

        # Enregistrement
        if key1 not in h2h_history:
            h2h_history[key1] = []
        h2h_history[key1].append({
            'result': row['result'],
            'gd': row['goal_diff'],
        })

    return df


def compute_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la position au classement et les statistiques de saison.

    Features :
      - home_pts_season  : total de points de l'équipe domicile dans la saison
      - away_pts_season  : total de points de l'équipe extérieure dans la saison
      - pts_diff         : différence de points entre les deux équipes
    """
    df = df.copy()
    df['season'] = df['date'].dt.year.astype(str) + '-' + (
        df['date'].dt.year + 1).astype(str)

    # Saison au sens football (juillet → juin)
    df['season'] = df['date'].apply(
        lambda d: f"{d.year}-{d.year+1}" if d.month >= 7 else f"{d.year-1}-{d.year}"
    )

    df['home_pts_season'] = np.nan
    df['away_pts_season'] = np.nan

    # Points cumulés PAR saison
    season_pts = {}  # (saison, team) → pts cumulés

    for idx, row in df.iterrows():
        home, away, season = row['home_team'], row['away_team'], row['season']

        key_h = (season, home)
        key_a = (season, away)

        df.at[idx, 'home_pts_season'] = season_pts.get(key_h, 0)
        df.at[idx, 'away_pts_season'] = season_pts.get(key_a, 0)

        # Mise à jour des points
        if row['result'] == 'H':
            season_pts[key_h] = season_pts.get(key_h, 0) + 3
        elif row['result'] == 'D':
            season_pts[key_h] = season_pts.get(key_h, 0) + 1
            season_pts[key_a] = season_pts.get(key_a, 0) + 1
        else:
            season_pts[key_a] = season_pts.get(key_a, 0) + 3

    df['pts_diff'] = df['home_pts_season'] - df['away_pts_season']
    return df


def build_features(df: pd.DataFrame, n_form: int = 5) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering (features tabulaires uniquement).
    Les embeddings sont ajoutés séparément via build_full_feature_matrix().
    """
    print("  🔧 Calcul des features de forme...")
    df = compute_form(df, n_games=n_form)

    print("  🔧 Calcul des head-to-head...")
    df = compute_head_to_head(df, n_games=5)

    print("  🔧 Calcul des stats de saison...")
    df = compute_season_stats(df)

    # ── Features dérivées ──────────────────────────────────────
    df['form_pts_diff']  = df['home_form_pts'] - df['away_form_pts']
    df['form_gf_diff']   = df['home_form_gf']  - df['away_form_gf']
    df['form_ga_diff']   = df['home_form_ga']  - df['away_form_ga']
    df['form_gd_diff']   = df['form_gf_diff']  - df['form_ga_diff']

    feature_cols = get_tabular_feature_columns()
    df_clean = df.dropna(subset=feature_cols).copy()

    print(f"  ✅ Features tabulaires : {len(feature_cols)} colonnes")
    print(f"     Matchs conservés après dropna : {len(df_clean)}/{len(df)}")
    return df_clean


def get_tabular_feature_columns() -> list:
    """Features tabulaires classiques (sans embeddings)."""
    return [
        'home_form_pts', 'home_form_gf', 'home_form_ga',
        'home_form_wins', 'home_form_clean_sheets',
        'away_form_pts', 'away_form_gf', 'away_form_ga',
        'away_form_wins', 'away_form_clean_sheets',
        'form_pts_diff', 'form_gf_diff', 'form_ga_diff', 'form_gd_diff',
        'h2h_home_wins', 'h2h_draw_rate', 'h2h_goals_diff',
        'home_pts_season', 'away_pts_season', 'pts_diff',
    ]


def build_full_feature_matrix(
    df: pd.DataFrame,
    hist_emb: 'HistoricalEmbeddingFeatures',
    news_emb: 'NewsEmbeddingFeatures',
) -> np.ndarray:
    """
    Construit la matrice X complète en concaténant 3 sources :

        [tabular (20) | hist_home (D) | hist_away (D) | hist_diff (D) | news (2)]

    Pourquoi cette concaténation ?
    ─────────────────────────────
    • Tabulaire   → signaux à court terme (forme, classement, H2H)
    • Historique  → identité et profil de l'équipe (style de jeu, niveau)
    • News        → signaux exogènes (blessures, crises, moral)

    Les trois sources sont complémentaires et capturent des horizons
    temporels différents. Le scaler dans le Pipeline sklearn les
    normalisera ensemble avant l'entraînement.
    """
    tabular_cols = get_tabular_feature_columns()

    # Bloc 1 : features tabulaires
    X_tab = df[tabular_cols].fillna(0).values.astype(np.float32)

    # Bloc 2 : embeddings historiques [embed_home | embed_away | embed_diff]
    X_hist = hist_emb.build_feature_matrix(df)          # shape (n, 3*D) ou (n, 0)

    # Bloc 3 : sentiment news [home_sent, away_sent]
    X_news = news_emb.build_feature_matrix(df)          # shape (n, 2) ou (n, 0)

    # Concaténation finale
    parts = [X_tab]
    if X_hist.shape[1] > 0:
        parts.append(X_hist)
    if X_news.shape[1] > 0:
        parts.append(X_news)

    X = np.concatenate(parts, axis=1)

    n_tab  = X_tab.shape[1]
    n_hist = X_hist.shape[1]
    n_news = X_news.shape[1]
    print(f"  📐 Matrice X finale : {X.shape[0]} matchs × {X.shape[1]} features")
    print(f"     └─ tabulaire={n_tab} | historique={n_hist} | news={n_news}")

    return X


# ─────────────────────────────────────────────────────────────────
#  ÉTAPE 3 : SPLIT TEMPOREL
# ─────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, test_season: int) -> tuple:
    """
    Split TEMPOREL — JAMAIS aléatoire pour des données de séries temporelles.

    On entraîne sur toutes les saisons avant test_season,
    et on teste sur test_season uniquement.

    Exemple : test_season=2025 → train sur 2020-2024, test sur 2025.
    """
    test_mask = df['date'].dt.year >= test_season
    train_df = df[~test_mask].copy()
    test_df  = df[test_mask].copy()

    print(f"  📅 Train : {len(train_df)} matchs (jusqu'en {test_season-1})")
    print(f"  📅 Test  : {len(test_df)} matchs (saison {test_season})")

    if len(train_df) < 50:
        raise ValueError(
            f"Pas assez de données d'entraînement ({len(train_df)} matchs). "
            f"Essaie --test-season {test_season + 1} ou fournis plus de données."
        )

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────
#  ÉTAPE 4 : ENTRAÎNEMENT ET ÉVALUATION
# ─────────────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Catalogue des modèles à comparer.

    Chaque modèle est un Pipeline sklearn (scaler + modèle) pour
    garantir que le preprocessing est inclus dans le .pkl.
    La calibration (CalibratedClassifierCV) est essentielle pour
    obtenir des probabilités fiables → indispensable pour Kelly.
    """
    models = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', CalibratedClassifierCV(
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,

                    solver='lbfgs',
                ),
                cv=3, method='isotonic'
            )),
        ]),

        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
                cv=3, method='isotonic'
            )),
        ]),

        'gradient_boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', CalibratedClassifierCV(
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                ),
                cv=3, method='isotonic'
            )),
        ]),

        'mlp': MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.005,        # Augmente cette valeur (ex: 0.01) si le modèle surapprend
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True, # Très efficace : s'arrête si le score de test ne progresse plus
    validation_fraction=0.1,
    random_state=42
),
    }

    if HAS_XGBOOST:
        models['xgboost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', CalibratedClassifierCV(
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    
                    eval_metric='mlogloss',
                    random_state=42,
                    n_jobs=-1,
                ),
                cv=3, method='isotonic'
            )),
        ])

    return models


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    X_train:  np.ndarray,
    X_test:   np.ndarray,
    model_name: str = 'auto',
) -> tuple:
    """
    Entraîne les modèles, compare, retourne le meilleur.

    Reçoit X_train / X_test déjà construits par build_full_feature_matrix()
    (tabulaire + embeddings historiques + sentiment news).

    La métrique de sélection est le log-loss (plus basses = meilleures probabilités)
    plutôt que l'accuracy, car des probabilités bien calibrées sont cruciales pour Kelly.
    """
    # Encodage des labels : A=0, D=1, H=2 (ordre alphabétique sklearn)
    le = LabelEncoder()
    le.fit(['A', 'D', 'H'])
    y_train = le.transform(train_df['result'].values)
    y_test  = le.transform(test_df['result'].values)

    all_models = build_models()

    # ── Sélection des modèles à entraîner ─────────────────────────
    if model_name == 'auto':
        models_to_train = all_models
    elif model_name in all_models:
        models_to_train = {model_name: all_models[model_name]}
    else:
        print(f"⚠️  Modèle '{model_name}' inconnu. Utilisation de 'auto'.")
        models_to_train = all_models

    print(f"\n  🏋️  Entraînement de {len(models_to_train)} modèle(s)...")

    results = {}
    for name, pipeline in models_to_train.items():
        print(f"\n  ── {name.upper()} ──")
        try:
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)

            acc   = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_proba)

            print(f"     Accuracy  : {acc:.4f}")
            print(f"     Log-loss  : {logloss:.4f}  ← métrique principale")

            results[name] = {
                'pipeline': pipeline,
                'accuracy': acc,
                'log_loss': logloss,
                'y_pred': y_pred,
                'y_proba': y_proba,
            }
        except Exception as e:
            print(f"     ❌ Erreur : {e}")

    if not results:
        raise RuntimeError("Aucun modèle n'a pu être entraîné.")

    # ── Sélection du meilleur (log-loss minimal) ──────────────────
    best_name = min(results, key=lambda k: results[k]['log_loss'])
    best = results[best_name]

    print(f"\n{'═'*50}")
    print(f"  🏆 Meilleur modèle : {best_name.upper()}")
    print(f"     Accuracy  : {best['accuracy']:.4f}")
    print(f"     Log-loss  : {best['log_loss']:.4f}")
    print(f"{'═'*50}")

    # ── Rapport détaillé ──────────────────────────────────────────
    print("\n  📊 Rapport de classification :")
    print(classification_report(
        y_test, best['y_pred'],
        target_names=le.classes_,
        digits=3,
    ))

    # ── Matrice de confusion ──────────────────────────────────────
    cm = confusion_matrix(y_test, best['y_pred'])
    print("  Matrice de confusion (A / D / H) :")
    print(f"     {cm}")

    return best['pipeline'], le, best_name, results


# ─────────────────────────────────────────────────────────────────
#  ÉTAPE 5 : EXPORT DES PRÉDICTIONS
# ─────────────────────────────────────────────────────────────────

def export_predictions(
    pipeline,
    label_encoder: LabelEncoder,
    test_df: pd.DataFrame,
    X_test: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """
    Génère le fichier CSV de prédictions pour backtesting.py.

    Colonnes exportées :
        match_id, date, home_team, away_team,
        prob_home, prob_draw, prob_away,   ← sorties du modèle
        result                             ← vérité terrain
    """
    y_proba = pipeline.predict_proba(X_test)

    # Ordre des classes sklearn : alphabétique A, D, H
    classes = list(label_encoder.classes_)
    idx_a = classes.index('A')
    idx_d = classes.index('D')
    idx_h = classes.index('H')
    # On ne prend que les colonnes qui existent vraiment
    cols_to_keep = [c for c in ['date', 'home_team', 'away_team', 'result'] if c in test_df.columns]
    out = test_df[cols_to_keep].copy()
    out['prob_home'] = y_proba[:, idx_h].round(4)
    out['prob_draw'] = y_proba[:, idx_d].round(4)
    out['prob_away'] = y_proba[:, idx_a].round(4)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\n  💾 Prédictions exportées : {output_path}")
    print(f"     {len(out)} matchs | colonnes : {list(out.columns)}")
    return out


# ─────────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle de prédiction de matchs et génère le .pkl"
    )
    parser.add_argument('--matches', required=True,
                        help='CSV des matchs historiques')
    parser.add_argument('--historical-embeddings', default=None,
                        help='Chemin vers historical_embeddings.pt (optionnel)')
    parser.add_argument('--news-embeddings', default=None,
                        help='Chemin vers news_embeddings.pt (optionnel)')
    parser.add_argument('--model', default='auto',
                        choices=['auto', 'logistic_regression', 'random_forest',
                                 'gradient_boosting', 'mlp', 'xgboost'],
                        help='Modèle à entraîner (auto = compare tous)')
    parser.add_argument('--test-season', type=int, default=2025,
                        help='Année de début de la saison de test (défaut: 2025)')
    parser.add_argument('--n-form', type=int, default=5,
                        help='Nb de matchs pour la forme récente (défaut: 5)')
    parser.add_argument('--output-model', default='models/model.pkl',
                        help='Chemin de sauvegarde du modèle')
    parser.add_argument('--output-predictions', default='data/processed/predictions.csv',
                        help='Chemin des prédictions CSV pour backtesting')
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION")
    print("═" * 55)

    # ── 1. Chargement ──────────────────────────────────────────────
    print("\n📂 Étape 1 : Chargement et nettoyage...")
    df = load_and_clean(args.matches)

    # ── 2. Feature engineering tabulaire ─────────────────────────
    print("\n🔧 Étape 2 : Feature engineering tabulaire...")
    df = build_features(df, n_form=args.n_form)

    # ── 3. Chargement des embeddings ──────────────────────────────
    print("\n🧠 Étape 3 : Chargement des embeddings...")
    hist_emb = HistoricalEmbeddingFeatures(
        pt_path=args.historical_embeddings
    )
    news_emb = NewsEmbeddingFeatures(
        pt_path=args.news_embeddings,
        news_window_days=7,
    )

    # ── 4. Split temporel ─────────────────────────────────────────
    print(f"\n📅 Étape 4 : Split temporel (test = saison {args.test_season})...")
    train_df, test_df = temporal_split(df, test_season=args.test_season)

    # ── 5. Construction des matrices X (tabular + embeddings) ─────
    print("\n📐 Étape 5 : Construction des matrices de features...")
    print("  Train :")
    X_train = build_full_feature_matrix(train_df, hist_emb, news_emb)
    print("  Test :")
    X_test  = build_full_feature_matrix(test_df,  hist_emb, news_emb)

    # ── 6. Entraînement ───────────────────────────────────────────
    print("\n🏋️  Étape 6 : Entraînement et évaluation...")
    pipeline, le, best_name, all_results = train_and_evaluate(
        train_df, test_df, X_train, X_test, model_name=args.model
    )

    # ── 7. Sauvegarde du modèle ────────────────────────────────────
    print("\n💾 Étape 7 : Sauvegarde du modèle...")
    os.makedirs(os.path.dirname(args.output_model) or '.', exist_ok=True)

    model_bundle = {
        'pipeline':       pipeline,
        'label_encoder':  le,
        'tabular_cols':   get_tabular_feature_columns(),
        'hist_emb':       hist_emb,   # inclus pour predict_proba ultérieure
        'news_emb':       news_emb,   # inclus pour predict_proba ultérieure
        'model_name':     best_name,
        'trained_on':     datetime.now().isoformat(),
        'test_season':    args.test_season,
        'n_form':         args.n_form,
        'n_features':     X_train.shape[1],
        'feature_breakdown': {
            'tabular':    len(get_tabular_feature_columns()),
            'historical': hist_emb.n_features(),
            'news':       news_emb.n_features(),
        },
        'metrics': {
            name: {'accuracy': r['accuracy'], 'log_loss': r['log_loss']}
            for name, r in all_results.items()
        },
    }
    joblib.dump(model_bundle, args.output_model)
    print(f"  ✅ Modèle sauvegardé : {args.output_model}")
    print(f"     Features : {X_train.shape[1]} total "
          f"(tabulaire={model_bundle['feature_breakdown']['tabular']} "
          f"| historique={model_bundle['feature_breakdown']['historical']} "
          f"| news={model_bundle['feature_breakdown']['news']})")

    # ── 8. Export des prédictions ──────────────────────────────────
    print("\n📤 Étape 8 : Export des prédictions pour backtesting...")
    export_predictions(pipeline, le, test_df, X_test, args.output_predictions)

    print("\n" + "═" * 55)
    print("  ✅ ENTRAÎNEMENT TERMINÉ")
    print(f"\n  Étape suivante — ajouter les cotes et lancer le backtesting :")
    print(f"    python prepare_backtesting_data.py \\")
    print(f"        --matches {args.matches} \\")
    print(f"        --odds data/raw/odds_PL.csv \\")
    print(f"        --model {args.output_model} \\")
    print(f"        --output data/processed/backtesting_ready.csv")
    print(f"\n    python backtesting.py \\")
    print(f"        --csv data/processed/backtesting_ready.csv \\")
    print(f"        --bankroll 1000")
    print("═" * 55 + "\n")


if __name__ == '__main__':
    main()
