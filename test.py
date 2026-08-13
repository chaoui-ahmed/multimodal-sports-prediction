"""
test.py — Tests interactifs du pipeline multimodal
===================================================
Usage :
    python test.py              → menu interactif
    python test.py --predict    → prédire un match spécifique
    python test.py --all        → lancer tous les tests
"""

import sys, os, argparse

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "stats"))
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "models"))
sys.path.insert(0, os.path.join(ROOT_DIR, "src", "nlp"))
os.chdir(ROOT_DIR)

# ─────────────────────────────────────────────────────
# TESTS UNITAIRES
# ─────────────────────────────────────────────────────

def test_api():
    """Test 1 — Connexion à l'API football-data.org"""
    print("\n🔌 TEST 1 : Connexion API football-data.org")
    print("─" * 45)
    import requests
    API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    if not API_KEY:
        print("  ⏭️  FOOTBALL_DATA_API_KEY not set — skipping (see .env.example)")
        return None
    r = requests.get(
        "https://api.football-data.org/v4/competitions/PL/matches",
        headers={"X-Auth-Token": API_KEY}
    )
    if r.status_code == 200:
        matches = r.json()["matches"]
        played  = [m for m in matches if m["score"]["fullTime"]["home"] is not None]
        print(f"  ✅ API OK — {len(matches)} matchs en tout, {len(played)} joués")
        m = played[-1]
        print(f"  🏟️  Dernier match : {m['homeTeam']['name']} {m['score']['fullTime']['home']}-{m['score']['fullTime']['away']} {m['awayTeam']['name']}")
        return True
    else:
        print(f"  ❌ Erreur {r.status_code} : {r.text[:100]}")
        return False


def test_embeddings():
    """Test 2 — Chargement des embeddings NLP"""
    print("\n🧠 TEST 2 : Embeddings NLP (fichiers .pt)")
    print("─" * 45)
    import torch
    ok = True
    for path, label in [("data/news_embeddings.pt", "récents"), ("data/historical_embeddings.pt", "historiques")]:
        if os.path.exists(path):
            d = torch.load(path, weights_only=False)
            print(f"  ✅ {label:12s} : {d['embeddings'].shape[0]:5d} articles, dim={d['embeddings'].shape[1]}")
        else:
            print(f"  ❌ {path} introuvable")
            ok = False
    return ok


def test_model_architecture():
    """Test 3 — Architecture du modèle (forward pass)"""
    print("\n🏗️  TEST 3 : Architecture MultimodalSportsModel")
    print("─" * 45)
    import torch
    from fusion_model import MultimodalSportsModel
    model = MultimodalSportsModel()
    nb_params = sum(p.numel() for p in model.parameters())

    # Différentes tailles de batch
    for batch in [1, 8, 32]:
        stats = torch.randn(batch, 5, 2)
        nlp   = torch.randn(batch, 768)
        out   = model(stats, nlp)
        assert out.shape == (batch, 3), f"Shape inattendue : {out.shape}"
        print(f"  ✅ Batch={batch:2d} → sortie {tuple(out.shape)} OK")

    print(f"  🧮 Paramètres totaux : {nb_params:,}")
    return True


def test_dataloader():
    """Test 4 — Chargement du dataset multimodal"""
    print("\n📦 TEST 4 : Chargement du dataset multimodal")
    print("─" * 45)

    train_csv = "data/processed/train_stats.csv"
    if not os.path.exists(train_csv):
        print(f"  ⚠️  {train_csv} absent — génération en cours...")
        from generate_demo_stats import generate_processed_stats
        generate_processed_stats()

    from fusion_dataloader import MultimodalSportsDataset
    from torch.utils.data import DataLoader

    ds = MultimodalSportsDataset(train_csv)
    print(f"  ✅ Dataset chargé : {len(ds)} samples")

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    stats_b, nlp_b, labels_b = next(iter(loader))
    print(f"  📐 stats_seq shape : {tuple(stats_b.shape)}   (attendu : [8, 5, 2])")
    print(f"  📐 nlp_vec   shape : {tuple(nlp_b.shape)}  (attendu : [8, 768])")
    print(f"  📐 labels    shape : {tuple(labels_b.shape)}      (attendu : [8])")
    print(f"  🏷️  Labels exemple : {labels_b.tolist()} (0=Win, 1=Draw, 2=Loss)")
    return True


def test_inference(home_team=None, away_team=None):
    """Test 5 — Prédiction d'un match"""
    print("\n🔮 TEST 5 : Inférence (prédiction d'un match)")
    print("─" * 45)

    import torch, numpy as np
    from fusion_model import MultimodalSportsModel, predict_match

    model_path = "src/models/multimodal_v1.pth"
    if not os.path.exists(model_path):
        print(f"  ⚠️  Modèle absent ({model_path}). Lance d'abord : python src/main.py")
        return False

    model = MultimodalSportsModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    print(f"  ✅ Modèle chargé depuis {model_path}")

    # Simulation : forme récente des équipes (valeurs normalisées entre 0 et 1)
    # [home_form, away_form] pour les 5 derniers matchs
    if home_team and away_team:
        print(f"\n  🏟️  Match : {home_team} (domicile) vs {away_team} (extérieur)")
        # Forme arbitraire — en vrai, à calculer depuis les CSV
        home_form = float(input("  → Forme domicile (0.0 à 1.0, ex: 0.7 = bonne forme) : ") or "0.6")
        away_form = float(input("  → Forme extérieur (0.0 à 1.0) : ") or "0.4")
        stats_seq = np.array([[home_form, away_form]] * 5, dtype=np.float32)
    else:
        # Valeurs de démo
        print("  ℹ️  Utilisation de valeurs de forme aléatoires (mode démo)")
        import random
        home_form = round(random.uniform(0.3, 0.9), 2)
        away_form = round(random.uniform(0.2, 0.8), 2)
        stats_seq = np.array([[home_form, away_form]] * 5, dtype=np.float32)
        print(f"  → Forme domicile : {home_form} | Forme extérieur : {away_form}")

    nlp_vec = torch.zeros(768)  # pas de news récentes → vecteur nul
    result  = predict_match(model, stats_seq, nlp_vec)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  🏆 Prédiction : {result['prediction']:<24s}│")
    print(f"  ├─────────────────────────────────────────┤")
    for label, prob in result["probabilités"].items():
        bar = "█" * int(float(prob.strip("%")) / 5)
        print(f"  │  {label:<22s} {prob:>6s}  {bar}")
    print(f"  └─────────────────────────────────────────┘")
    return True


def predict_real_match():
    """Mode interactif : prédire un vrai match en entrant les équipes"""
    print("\n" + "═" * 50)
    print("  ⚽  PRÉDICTION D'UN MATCH")
    print("═" * 50)

    TEAMS = [
        "Arsenal FC", "Aston Villa FC", "Brentford FC", "Brighton & Hove Albion FC",
        "Chelsea FC", "Crystal Palace FC", "Everton FC", "Fulham FC",
        "Ipswich Town FC", "Leicester City FC", "Liverpool FC",
        "Manchester City FC", "Manchester United FC", "Newcastle United FC",
        "Nottingham Forest FC", "Southampton FC", "Tottenham Hotspur FC",
        "West Ham United FC", "Wolverhampton Wanderers FC", "AFC Bournemouth"
    ]
    print("\n  Équipes disponibles (Premier League 2024/25) :")
    for i, t in enumerate(TEAMS, 1):
        print(f"    {i:2d}. {t}")

    try:
        hi = int(input("\n  → Numéro équipe DOMICILE : ")) - 1
        ai = int(input("  → Numéro équipe EXTÉRIEUR : ")) - 1
        home = TEAMS[hi]
        away = TEAMS[ai]
    except (ValueError, IndexError):
        print("  Entrée invalide, utilisation de valeurs par défaut.")
        home, away = "Liverpool FC", "Arsenal FC"

    test_inference(home, away)


# ─────────────────────────────────────────────────────
# MENU PRINCIPAL
# ─────────────────────────────────────────────────────

def run_all_tests():
    results = {}
    results["API"]          = test_api()
    results["Embeddings"]   = test_embeddings()
    results["Architecture"] = test_model_architecture()
    results["Dataloader"]   = test_dataloader()
    results["Inference"]    = test_inference()

    print("\n" + "═" * 45)
    print("  RÉSUMÉ DES TESTS")
    print("═" * 45)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name}")
    total = sum(results.values())
    print(f"\n  {total}/{len(results)} tests passés")


def interactive_menu():
    print("\n╔══════════════════════════════════════════╗")
    print("║   ⚽  Multimodal Sports — Tests          ║")
    print("╚══════════════════════════════════════════╝")
    print("""
  1. 🔌 Tester la connexion API
  2. 🧠 Vérifier les embeddings NLP
  3. 🏗️  Tester l'architecture du modèle
  4. 📦 Tester le chargement des données
  5. 🔮 Lancer une inférence de démo
  6. ⚽ Prédire un vrai match (interactif)
  7. 🚀 Lancer TOUS les tests
  0. Quitter
""")
    choice = input("  Choix → ").strip()

    actions = {
        "1": test_api,
        "2": test_embeddings,
        "3": test_model_architecture,
        "4": test_dataloader,
        "5": test_inference,
        "6": predict_real_match,
        "7": run_all_tests,
    }

    if choice in actions:
        try:
            actions[choice]()
        except Exception as e:
            print(f"\n  ❌ Erreur : {e}")
            import traceback; traceback.print_exc()
    elif choice == "0":
        print("  Bye 👋")
    else:
        print("  Choix invalide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true", help="Prédire un match (interactif)")
    parser.add_argument("--all",     action="store_true", help="Lancer tous les tests")
    args = parser.parse_args()

    if args.predict:
        predict_real_match()
    elif args.all:
        run_all_tests()
    else:
        interactive_menu()
