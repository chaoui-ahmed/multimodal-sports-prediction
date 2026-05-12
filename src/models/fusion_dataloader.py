"""
fusion_dataloader.py
--------------------
Combine les vecteurs NLP d'Archi (news_embeddings.pt / historical_embeddings.pt)
avec les séquences de stats d'Estelle (train_stats.csv / test_stats.csv)
pour créer un Dataset PyTorch multimodal.

Structure d'un sample :
  - stats_seq   : Tensor [5, 2]    → 5 matchs × (home_form, away_form)
  - nlp_vec     : Tensor [768]     → vecteur RoBERTa de l'article le plus proche
  - label       : int              → 0=Win, 1=Draw, 2=Loss
"""

import sys, os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import de la liste des features stats depuis le module de feature engineering
# → garantit que dataloader et modèle restent synchronisés
try:
    _stats_dir = os.path.join(os.path.dirname(__file__), '..', 'stats')
    sys.path.insert(0, os.path.abspath(_stats_dir))
    from generate_demo_stats import STAT_FEATURES
except ImportError:
    # Fallback si on est lancé depuis un répertoire différent
    STAT_FEATURES = [
        'home_form', 'home_goals_scored', 'home_goals_conceded', 'home_elo',
        'away_form', 'away_goals_scored', 'away_goals_conceded', 'away_elo',
    ]

# ──────────────────────────────────────────────
# 1. Chargement des embeddings NLP (fichiers .pt)
# ──────────────────────────────────────────────

def load_nlp_embeddings(recent_path="data/news_embeddings.pt",
                         historical_path="data/historical_embeddings.pt"):
    """
    Charge et fusionne les embeddings récents + historiques d'Archi.
    Retourne un dict { date_str -> tenseur [768] } pour chaque article.
    """
    all_dates = []
    all_embeddings = []

    # -- Articles récents (scraper d'Archi)
    try:
        recent = torch.load(recent_path, weights_only=False)
        all_dates.extend(recent["dates"])
        # recent["embeddings"] : shape [N, 768]
        for i in range(recent["embeddings"].shape[0]):
            all_embeddings.append(recent["embeddings"][i])
        print(f"✅ Embeddings récents chargés : {len(recent['dates'])} articles")
    except FileNotFoundError:
        print(f"⚠️  {recent_path} introuvable, on continue sans.")

    # -- Articles historiques (Transfermarkt dataset)
    try:
        historical = torch.load(historical_path, weights_only=False)
        all_dates.extend(historical["dates"])
        for i in range(historical["embeddings"].shape[0]):
            all_embeddings.append(historical["embeddings"][i])
        print(f"✅ Embeddings historiques chargés : {len(historical['dates'])} articles")
    except FileNotFoundError:
        print(f"⚠️  {historical_path} introuvable, on continue sans.")

    if not all_embeddings:
        print("❌ Aucun embedding chargé !")
        return {}, None

    # On empile tout dans une matrice [total_articles, 768]
    embedding_matrix = torch.stack(all_embeddings, dim=0)
    print(f"📐 Matrice NLP totale : {embedding_matrix.shape}")

    return all_dates, embedding_matrix


# ──────────────────────────────────────────────
# 2. Trouver le vecteur NLP le plus proche d'une date
# ──────────────────────────────────────────────

def parse_date_flexible(date_str):
    """Essaie plusieurs formats de date courants."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",   # RSS : "Tue, 05 May 2026 16:25:53 +0200"
        "%Y-%m-%d %H:%M",              # historical_embeddings : "2020-06-01 23:32"
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
    # Dernière chance : pandas
    try:
        return pd.to_datetime(str(date_str), utc=False).to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


def find_closest_nlp_vector(match_date, all_dates, embedding_matrix,
                             window_days=30, zero_vec_size=768):
    """
    Pour un match donné, cherche les articles dans une fenêtre de `window_days`
    jours autour du match (avant ET après) et retourne la moyenne de leurs embeddings.
    Si rien n'est trouvé dans la fenêtre, prend les N articles temporellement
    les plus proches comme fallback.
    """
    if not all_dates or embedding_matrix is None:
        return torch.zeros(zero_vec_size)

    if not isinstance(match_date, datetime):
        match_date = parse_date_flexible(match_date)
    if match_date is None:
        return torch.zeros(zero_vec_size)

    window_start = match_date - timedelta(days=window_days)
    window_end   = match_date + timedelta(days=window_days)
    selected_indices = []
    parsed_dates = []

    for i, d_str in enumerate(all_dates):
        article_date = parse_date_flexible(d_str)
        if article_date is None:
            continue
        parsed_dates.append((i, article_date))
        if window_start <= article_date <= window_end:
            selected_indices.append(i)

    if selected_indices:
        selected = embedding_matrix[selected_indices]
        return selected.mean(dim=0)

    # Fallback : les 5 articles les plus proches en temps
    if parsed_dates:
        parsed_dates.sort(key=lambda x: abs((x[1] - match_date).total_seconds()))
        closest = [idx for idx, _ in parsed_dates[:5]]
        return embedding_matrix[closest].mean(dim=0)

    return torch.zeros(zero_vec_size)


# ──────────────────────────────────────────────
# 3. Dataset multimodal
# ──────────────────────────────────────────────

class MultimodalSportsDataset(Dataset):
    """
    Chaque sample contient :
      - stats_seq (Tensor [5, 2])   : séquence de 5 matchs (home_form, away_form)
      - nlp_vec   (Tensor [768])    : vecteur RoBERTa contextualisé
      - label     (int)             : 0=Win, 1=Draw, 2=Loss
    """

    def __init__(self,
                 csv_path,
                 recent_emb_path="data/news_embeddings.pt",
                 historical_emb_path="data/historical_embeddings.pt",
                 sequence_length=5,
                 nlp_window_days=7):

        self.sequence_length = sequence_length

        # -- Chargement du CSV stats (Estelle)
        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

        # -- Normalisation Min-Max des 8 features stats
        available = [c for c in STAT_FEATURES if c in self.df.columns]
        if len(available) < len(STAT_FEATURES):
            missing = set(STAT_FEATURES) - set(available)
            print(f"  ⚠️  Features manquantes dans le CSV : {missing}")
            print(f"       → Utilisation des features disponibles : {available}")
        features_df = self.df[available].copy()
        denom = features_df.max() - features_df.min()
        denom = denom.replace(0, 1)  # évite division par zéro
        self.features = ((features_df - features_df.min()) / denom).fillna(0).values

        # -- Labels : Win=0, Draw=1, Loss=2
        def map_label(row):
            if row['home_score'] > row['away_score']: return 0
            if row['home_score'] == row['away_score']: return 1
            return 2
        self.labels = self.df.apply(map_label, axis=1).values

        # -- Chargement des embeddings NLP (Archi)
        print("\nChargement des embeddings NLP...")
        self.all_dates, self.embedding_matrix = load_nlp_embeddings(
            recent_emb_path, historical_emb_path
        )
        self.nlp_window_days = nlp_window_days

        # -- Précalcul des vecteurs NLP pour chaque match (accélère l'entraînement)
        print(f"Précalcul des vecteurs NLP pour {len(self.df)} matchs...")
        self.nlp_vectors = []
        for idx, row in self.df.iterrows():
            vec = find_closest_nlp_vector(
                row['date'], self.all_dates, self.embedding_matrix,
                window_days=self.nlp_window_days
            )
            self.nlp_vectors.append(vec)
        print(f"✅ Dataset multimodal prêt : {len(self)} samples")

    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        # Séquence de 5 matchs passés → shape [5, 2]
        stats_seq = torch.tensor(
            self.features[idx : idx + self.sequence_length],
            dtype=torch.float32
        )
        # Vecteur NLP du match cible → shape [768]
        nlp_vec = self.nlp_vectors[idx + self.sequence_length]

        # Label du match cible
        label = torch.tensor(self.labels[idx + self.sequence_length], dtype=torch.long)

        return stats_seq, nlp_vec, label


# ──────────────────────────────────────────────
# 4. Calcul des poids de classes (anti-biais)
# ──────────────────────────────────────────────

def compute_class_weights(dataset):
    """
    Retourne un tenseur de poids inversement proportionnel à la fréquence
    de chaque classe — à passer à nn.CrossEntropyLoss(weight=...).
    """
    import numpy as np
    from torch import tensor
    labels = [dataset[i][2].item() for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=3).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 3
    print(f"  ⚖️  Poids classes → Win: {weights[0]:.2f} | Draw: {weights[1]:.2f} | Loss: {weights[2]:.2f}")
    return tensor(weights, dtype=torch.float32)


# ──────────────────────────────────────────────
# 5. Dataset depuis DataFrame (pour la CV)
# ──────────────────────────────────────────────

class MultimodalSportsDatasetFromDF(Dataset):
    """
    Variante de MultimodalSportsDataset qui prend un DataFrame directement
    au lieu d'un chemin CSV — utilisée en cross-validation pour éviter
    d'écrire des fichiers temporaires.
    """

    def __init__(self, df, all_dates, embedding_matrix,
                 sequence_length=5, nlp_window_days=30):
        self.sequence_length = sequence_length

        df = df.copy().reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        available = [c for c in STAT_FEATURES if c in df.columns]
        features_df = df[available].copy()
        denom = features_df.max() - features_df.min()
        denom = denom.replace(0, 1)
        self.features = ((features_df - features_df.min()) / denom).fillna(0).values

        def map_label(row):
            if row['home_score'] > row['away_score']: return 0
            if row['home_score'] == row['away_score']: return 1
            return 2
        self.labels = df.apply(map_label, axis=1).values

        self.nlp_vectors = []
        for _, row in df.iterrows():
            vec = find_closest_nlp_vector(
                row['date'], all_dates, embedding_matrix,
                window_days=nlp_window_days
            )
            self.nlp_vectors.append(vec)

    def __len__(self):
        return len(self.labels) - self.sequence_length

    def __getitem__(self, idx):
        stats_seq = torch.tensor(
            self.features[idx: idx + self.sequence_length],
            dtype=torch.float32
        )
        nlp_vec = self.nlp_vectors[idx + self.sequence_length]
        label   = torch.tensor(self.labels[idx + self.sequence_length], dtype=torch.long)
        return stats_seq, nlp_vec, label


# ──────────────────────────────────────────────
# 6. Test rapide
# ──────────────────────────────────────────────

if __name__ == "__main__":
    dataset = MultimodalSportsDataset(
        csv_path="data/processed/train_stats.csv",
        recent_emb_path="data/news_embeddings.pt",
        historical_emb_path="data/historical_embeddings.pt",
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    stats_batch, nlp_batch, labels_batch = next(iter(loader))

    print("\n--- Vérification des shapes ---")
    print(f"Stats  : {stats_batch.shape}")   # [32, 5, 2]
    print(f"NLP    : {nlp_batch.shape}")     # [32, 768]
    print(f"Labels : {labels_batch.shape}")  # [32]
    print(f"Exemple label : {labels_batch[0].item()} (0=Win, 1=Draw, 2=Loss)")

