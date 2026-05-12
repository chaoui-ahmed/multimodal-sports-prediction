"""
main.py — Pipeline multimodal de prédiction de matchs de football
==================================================================
Orchestrateur du projet. Lance les étapes dans l'ordre :
  1. Génération / récupération des données stats
  2. Preprocessing + feature engineering
  3. Chargement des embeddings NLP (déjà présents)
  4. Entraînement du modèle de fusion
  5. Évaluation + génération des graphiques

Usage :
    python src/main.py              # mode démo (données synthétiques)
    python src/main.py --real-api   # mode réel (requiert .env FOOTBALL_DATA_API_KEY)
"""

import sys
import os
import argparse

# S'assurer que les modules du projet sont accessibles
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "stats"))
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "models"))
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "nlp"))

os.chdir(ROOT_DIR)  # Toujours exécuter depuis la racine du projet


def _fetch_and_prepare_real_data(train_csv, test_csv):
    """
    Mode réel :
      1. Télécharge les données historiques PL depuis Kaggle (2020→2024)
      2. Télécharge la saison en cours 2024/25 via l'API
      3. Fusionne et calcule la forme glissante
      4. Sauvegarde train/test pour compatibilité (split 80/20)
      5. Retourne le DataFrame complet (pour la cross-validation)
    """
    import pandas as pd

    # -- 1. Données historiques Kaggle
    print("📦 Chargement des données historiques (Kaggle)...")
    from fetch_historic import download_and_extract
    df_hist = download_and_extract("E0")
    print(f"  → {len(df_hist)} matchs historiques PL (Kaggle)")

    # -- 2. Données récentes via l'API
    print("📡 Récupération de la saison 2024/25 (API)...")
    from fetch_data import save_matches_to_csv
    save_matches_to_csv("PL")

    df_api = pd.read_csv("data/raw/matches_PL.csv")
    df_api['date'] = pd.to_datetime(df_api['date'], utc=True).dt.tz_localize(None)
    df_api = df_api.dropna(subset=['home_score', 'away_score']).copy()
    df_api['home_score'] = df_api['home_score'].astype(int)
    df_api['away_score'] = df_api['away_score'].astype(int)
    print(f"  → {len(df_api)} matchs joués (API 2024/25)")

    # -- 3. Fusion
    cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    df_all = pd.concat([df_hist[cols], df_api[cols]], ignore_index=True)
    df_all = (df_all
              .sort_values('date')
              .drop_duplicates(subset=['date', 'home_team', 'away_team'])
              .reset_index(drop=True))
    print(f"  → {len(df_all)} matchs total après fusion")
    print(f"     Du {df_all['date'].min().strftime('%d/%m/%Y')} "
          f"au {df_all['date'].max().strftime('%d/%m/%Y')}")

    # -- 4. Rolling form
    print("  ⚙️  Calcul de la forme glissante...")
    from generate_demo_stats import calculate_rolling_form
    df_all = calculate_rolling_form(df_all)

    # -- 5. Sauvegarde train/test (80/20) pour compatibilité avec les scripts existants
    split_idx = int(len(df_all) * 0.80)
    os.makedirs("data/processed", exist_ok=True)
    df_all.iloc[:split_idx].to_csv(train_csv, index=False)
    df_all.iloc[split_idx:].to_csv(test_csv,  index=False)
    print(f"  ✅ train_stats.csv ({split_idx} matchs) + test_stats.csv ({len(df_all)-split_idx} matchs)")

    return df_all






def step_banner(step_num, title):
    print(f"\n{'═' * 55}")
    print(f"  ÉTAPE {step_num} — {title}")
    print(f"{'═' * 55}")


def run_pipeline(use_demo_data=True):
    """Exécute le pipeline complet d'entraînement multimodal."""

    print("╔══════════════════════════════════════════════════════╗")
    print("║    ⚽  Multimodal Sports Prediction Pipeline  ⚽      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n📁 Répertoire de travail : {ROOT_DIR}")
    mode = "DÉMO (données synthétiques)" if use_demo_data else "RÉEL (API football-data.org)"
    print(f"🔧 Mode : {mode}\n")

    # ─────────────────────────────────────────────────
    # ÉTAPE 1 : Génération/récupération des données stats
    # ─────────────────────────────────────────────────
    step_banner(1, "Données statistiques")

    train_csv = "data/processed/train_stats.csv"
    test_csv  = "data/processed/test_stats.csv"

    if use_demo_data:
        print("Mode démo : génération de données synthétiques...")
        from generate_demo_stats import generate_processed_stats
        generate_processed_stats()
    else:
        df_all = _fetch_and_prepare_real_data(train_csv, test_csv)

    # ─────────────────────────────────────────────────
    # ÉTAPE 2 : Chargement des embeddings NLP
    # ─────────────────────────────────────────────────
    step_banner(2, "Embeddings NLP (RoBERTa)")

    import torch
    from fusion_dataloader import load_nlp_embeddings

    recent_emb     = "data/news_embeddings.pt"
    historical_emb = "data/historical_embeddings.pt"

    for emb_path in [recent_emb, historical_emb]:
        if os.path.exists(emb_path):
            d = torch.load(emb_path, weights_only=False)
            print(f"  ✅ {emb_path} → {d['embeddings'].shape[0]} articles")
        else:
            print(f"  ⚠️  {emb_path} introuvable")

    all_dates, embedding_matrix = load_nlp_embeddings(recent_emb, historical_emb)

    # ─────────────────────────────────────────────────
    # ÉTAPE 3 : Sanity check architecture
    # ─────────────────────────────────────────────────
    step_banner(3, "Test de l'architecture du modèle")

    from fusion_model import MultimodalSportsModel, STATS_INPUT_SIZE

    model_test  = MultimodalSportsModel()
    out         = model_test(torch.randn(4, 5, STATS_INPUT_SIZE), torch.randn(4, 768))
    nb_params   = sum(p.numel() for p in model_test.parameters())
    print(f"  ✅ Forward pass OK → shape sortie : {out.shape}  (attendu : [4, 3])")
    print(f"  🧠 Paramètres totaux : {nb_params:,}")
    print(f"  📐 Features stats : {STATS_INPUT_SIZE} (form, buts marqués, buts concédés, ELO) x 2")
    assert out.shape == (4, 3)


    # ─────────────────────────────────────────────────
    # ÉTAPE 4 : Entraînement (CV temporelle ou simple)
    # ─────────────────────────────────────────────────
    step_banner(4, "Entraînement du modèle de fusion")

    SAVE_PATH = "src/models/multimodal_v1.pth"

    if use_demo_data:
        # Mode démo : train/test split simple
        from fusion_model import train as train_model
        trained_model = train_model(
            csv_train      = train_csv,
            csv_test       = test_csv,
            recent_emb     = recent_emb,
            historical_emb = historical_emb,
            epochs         = 15,
            batch_size     = 32,
            lr             = 5e-4,
            save_path      = SAVE_PATH,
        )
    else:
        # Mode réel : cross-validation temporelle (TimeSeriesSplit)
        from fusion_model import cross_validate
        trained_model = cross_validate(
            df              = df_all,
            all_dates       = all_dates,
            embedding_matrix= embedding_matrix,
            n_splits        = 5,
            epochs          = 15,
            batch_size      = 32,
            lr              = 5e-4,
            save_path       = SAVE_PATH,
        )

    # ─────────────────────────────────────────────────
    # ÉTAPE 5 : Inférence de démonstration
    # ─────────────────────────────────────────────────
    step_banner(5, "Inférence de démonstration")

    from fusion_model import predict_match
    import numpy as np

    demo_stats = np.random.rand(5, 2).astype(np.float32)
    demo_nlp   = torch.zeros(768)
    result     = predict_match(trained_model, demo_stats, demo_nlp)

    print(f"\n  🔮 Match fictif :")
    print(f"  ┌────────────────────────────────────────┐")
    print(f"  │  Prédit : {result['prediction']:<30s}│")
    print(f"  ├────────────────────────────────────────┤")
    for label, prob in result["probabilités"].items():
        bar = "█" * int(float(prob.strip("%")) / 5)
        print(f"  │  {label:<22s} {prob:>6s}  {bar}")
    print(f"  └────────────────────────────────────────┘")

    # ─────────────────────────────────────────────────
    # ÉTAPE 6 : Graphiques (mode démo seulement)
    # ─────────────────────────────────────────────────
    if use_demo_data:
        step_banner(6, "Génération des graphiques")
        _generate_plots(trained_model, test_csv, recent_emb, historical_emb)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  ✅  Pipeline terminé avec succès !                   ║")
    print("║  💾  Modèle    → src/models/multimodal_v1.pth         ║")
    print("╚══════════════════════════════════════════════════════╝\n")



def _generate_plots(model, test_csv, recent_emb, historical_emb):
    """Génère la matrice de confusion et la courbe de loss."""
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False

    from sklearn.metrics import confusion_matrix, classification_report
    from fusion_dataloader import MultimodalSportsDataset
    from torch.utils.data import DataLoader

    os.makedirs("reports/figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Chargement du jeu de test
    print("  Chargement du dataset de test pour évaluation...")
    test_dataset = MultimodalSportsDataset(test_csv, recent_emb, historical_emb)

    if len(test_dataset) == 0:
        print("  ⚠️  Dataset de test vide, graphiques ignorés.")
        return

    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for stats_seq, nlp_vec, labels in test_loader:
            stats_seq, nlp_vec = stats_seq.to(device), nlp_vec.to(device)
            logits = model(stats_seq, nlp_vec)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # ── Graph 1 : Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    if has_seaborn:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Win", "Draw", "Loss"],
                    yticklabels=["Win", "Draw", "Loss"])
    else:
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Win", "Draw", "Loss"])
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Win", "Draw", "Loss"])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax.set_title("Matrice de Confusion — Modèle Multimodal", fontsize=14)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    fig.tight_layout()
    fig.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("  ✅ reports/figures/confusion_matrix.png")

    # ── Graph 2 : Distribution des prédictions
    fig, ax = plt.subplots(figsize=(8, 5))
    labels_str = ["Win (domicile)", "Draw", "Loss (domicile)"]
    counts_true = [np.sum(y_true == i) for i in range(3)]
    counts_pred = [np.sum(y_pred == i) for i in range(3)]
    x = np.arange(3)
    ax.bar(x - 0.2, counts_true, 0.4, label="Réel", color="#4A90D9", alpha=0.85)
    ax.bar(x + 0.2, counts_pred, 0.4, label="Prédit", color="#E84545", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels_str)
    ax.set_ylabel("Nombre de matchs")
    ax.set_title("Distribution des résultats — Réel vs Prédit", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig("reports/figures/distribution.png", dpi=150)
    plt.close(fig)
    print("  ✅ reports/figures/distribution.png")

    # ── Rapport texte
    report = classification_report(y_true, y_pred,
                                   target_names=["Win", "Draw", "Loss"],
                                   zero_division=0)
    print(f"\n  📊 Rapport de classification :\n{report}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline multimodal de prédiction de matchs de football"
    )
    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Utiliser les vraies données via l'API (requiert FOOTBALL_DATA_API_KEY dans .env)"
    )
    args = parser.parse_args()

    run_pipeline(use_demo_data=not args.real_api)


if __name__ == "__main__":
    main()
