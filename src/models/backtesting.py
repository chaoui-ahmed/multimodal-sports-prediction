"""
╔══════════════════════════════════════════════════════════════════╗
║           SIMULATEUR DE BACKTESTING - PARIS SPORTIFS             ║
║      Value Bets + Fractional Kelly Criterion + Métriques         ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python backtesting.py --csv predictions.csv --bankroll 1000 --kelly-fraction 0.5

Format du CSV attendu (colonnes obligatoires):
    date, match_id, home_team, away_team,
    prob_home, prob_draw, prob_away,         <- Probabilités prédites par l'IA (entre 0 et 1, somme = 1)
    odd_home, odd_draw, odd_away,            <- Cotes bookmaker (ex: 2.10)
    result                                   <- Résultat réel : 'H' (home), 'D' (draw), 'A' (away)
"""

import pandas as pd
import numpy as np
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ─────────────────────────────────────────────
#  STRUCTURES DE DONNÉES
# ─────────────────────────────────────────────

@dataclass
class Bet:
    """Représente un pari individuel."""
    date: str
    match_id: str
    home_team: str
    away_team: str
    outcome: str          # 'H', 'D', 'A'
    ai_prob: float        # Probabilité estimée par l'IA
    bookmaker_prob: float # Probabilité implicite du bookmaker (1 / cote)
    odd: float            # Cote du bookmaker
    edge: float           # Avantage calculé
    stake: float          # Mise en euros
    result: str           # Résultat réel
    won: bool             # Le pari est-il gagnant ?
    pnl: float            # Profit ou perte en euros


@dataclass
class BankrollHistory:
    """Historique de la bankroll au fil du temps."""
    dates: list = field(default_factory=list)
    values: list = field(default_factory=list)
    bets_placed: list = field(default_factory=list)


# ─────────────────────────────────────────────
#  CLASSE PRINCIPALE : BETTOR
# ─────────────────────────────────────────────

class Bettor:
    """
    Le Parieur virtuel.

    Paramètres
    ----------
    initial_bankroll : float
        Budget de départ en euros (ex: 1000).
    kelly_fraction : float
        Fraction du critère de Kelly à appliquer (0.0 à 1.0).
        0.5 = "Half Kelly" — recommandé pour réduire le risque de ruine.
    min_edge : float
        Edge minimum requis pour déclencher un pari (ex: 0.05 = 5%).
    max_bet_pct : float
        Limite maximale de mise en % de la bankroll (garde-fou).
    min_stake : float
        Mise minimale en euros (évite les mises dérisoires).
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.5,
        min_edge: float = 0.05,
        max_bet_pct: float = 0.10,
        min_stake: float = 1.0,
    ):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_pct = max_bet_pct
        self.min_stake = min_stake

        self.bets: list[Bet] = []
        self.history = BankrollHistory()
        self.skipped_matches = 0

    # ── 1. DÉTECTION DES VALUE BETS ──────────────────────────────────

    def is_value_bet(self, ai_prob: float, bookmaker_odd: float) -> tuple[bool, float]:
        """
        Détecte si une opportunité est un Value Bet.

        Un Value Bet existe quand :
            ai_prob > probabilité implicite du bookmaker
            et l'edge dépasse le seuil minimum.

        Formule edge : (ai_prob * odd) - 1
            > 0  → valeur positive (Value Bet)
            = 0  → pari équitable
            < 0  → valeur négative (à éviter)

        Retourne : (est_un_value_bet, edge_calculé)
        """
        bookmaker_prob = 1.0 / bookmaker_odd
        edge = (ai_prob * bookmaker_odd) - 1.0

        is_value = (ai_prob > bookmaker_prob) and (edge >= self.min_edge)
        return is_value, edge

    # ── 2. CRITÈRE DE KELLY FRACTIONNEL ──────────────────────────────

    def kelly_stake(self, ai_prob: float, bookmaker_odd: float) -> float:
        """
        Calcule la mise optimale via le Critère de Kelly Fractionnel.

        Formule de Kelly :
            f* = (b * p - q) / b
            où :
              b = odd - 1  (gain net pour 1€ misé)
              p = probabilité de gagner (estimée par l'IA)
              q = 1 - p    (probabilité de perdre)

        Fraction Kelly :
            mise = f* * kelly_fraction * bankroll

        Le kelly_fraction (ex: 0.5) réduit le risque de ruine tout en
        préservant une croissance à long terme.
        """
        b = bookmaker_odd - 1.0
        p = ai_prob
        q = 1.0 - p

        if b <= 0:
            return 0.0

        kelly_pct = (b * p - q) / b

        # Kelly négatif = pas de valeur → on ne mise pas
        if kelly_pct <= 0:
            return 0.0

        # Application de la fraction de sécurité
        fractional_kelly = kelly_pct * self.kelly_fraction

        # Garde-fous
        fractional_kelly = min(fractional_kelly, self.max_bet_pct)
        stake = fractional_kelly * self.bankroll

        return max(stake, self.min_stake) if stake >= self.min_stake else 0.0

    # ── 3. TRAITEMENT D'UN MATCH ──────────────────────────────────────

    def process_match(self, row: pd.Series) -> Optional[Bet]:
        """
        Analyse un match et décide de parier ou non.

        Pour chaque issue possible (H, D, A), on vérifie s'il y a une
        Value Bet. On ne joue que l'issue avec le meilleur edge.
        """
        candidates = []

        for outcome, prob_col, odd_col in [
            ("H", "prob_home", "odd_home"),
            ("D", "prob_draw", "odd_draw"),
            ("A", "prob_away", "odd_away"),
        ]:
            ai_prob = row[prob_col]
            odd = row[odd_col]

            is_value, edge = self.is_value_bet(ai_prob, odd)

            if is_value:
                candidates.append((outcome, ai_prob, odd, edge))

        if not candidates:
            self.skipped_matches += 1
            return None

        # On choisit l'issue avec le meilleur edge
        best = max(candidates, key=lambda x: x[3])
        outcome, ai_prob, odd, edge = best

        stake = self.kelly_stake(ai_prob, odd)
        if stake <= 0:
            self.skipped_matches += 1
            return None

        # Sécurité : on ne mise pas plus que la bankroll disponible
        stake = min(stake, self.bankroll)

        # Résolution du pari
        won = (row["result"] == outcome)
        pnl = stake * (odd - 1) if won else -stake

        # Mise à jour de la bankroll
        self.bankroll += pnl

        bet = Bet(
            date=str(row.get("date", "unknown")),
            match_id=str(row.get("match_id", "unknown")),
            home_team=str(row.get("home_team", "?")),
            away_team=str(row.get("away_team", "?")),
            outcome=outcome,
            ai_prob=ai_prob,
            bookmaker_prob=1.0 / odd,
            odd=odd,
            edge=edge,
            stake=stake,
            result=row["result"],
            won=won,
            pnl=pnl,
        )
        self.bets.append(bet)
        return bet

    # ── 4. BOUCLE TEMPORELLE PRINCIPALE ──────────────────────────────

    def run(self, df: pd.DataFrame, verbose: bool = True) -> None:
        """
        Lance la simulation sur tout le dataset (saison complète).
        Les matchs sont triés par date pour respecter l'ordre chronologique.
        """
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        if verbose:
            print(f"\n{'═'*60}")
            print(f"  🏦 Bankroll initiale : {self.initial_bankroll:.2f}€")
            print(f"  📊 Kelly fraction    : {self.kelly_fraction} ({self.kelly_fraction*100:.0f}%)")
            print(f"  🎯 Edge minimum      : {self.min_edge*100:.1f}%")
            print(f"  📅 Matchs à analyser : {len(df)}")
            print(f"{'═'*60}\n")

        # Enregistrement de la bankroll initiale
        self.history.dates.append("Start")
        self.history.values.append(self.bankroll)
        self.history.bets_placed.append(0)

        for _, row in df.iterrows():
            bet = self.process_match(row)

            # Snapshot de la bankroll après chaque pari joué
            if bet is not None:
                self.history.dates.append(bet.date)
                self.history.values.append(self.bankroll)
                self.history.bets_placed.append(len(self.bets))

                if verbose:
                    status = "✅ GAGNÉ" if bet.won else "❌ PERDU"
                    print(
                        f"  {status} | {bet.home_team} vs {bet.away_team} "
                        f"| Pari: {bet.outcome} @ {bet.odd:.2f} "
                        f"| Mise: {bet.stake:.2f}€ "
                        f"| P&L: {'+' if bet.pnl >= 0 else ''}{bet.pnl:.2f}€ "
                        f"| Bankroll: {self.bankroll:.2f}€"
                    )

    # ── 5. MÉTRIQUES DE PERFORMANCE ───────────────────────────────────

    def compute_metrics(self) -> dict:
        """
        Calcule les métriques de performance de la stratégie.

        ROI (Return on Investment) :
            ROI = (bankroll_finale - bankroll_initiale) / bankroll_initiale * 100
            Mesure la croissance globale du capital.

        Yield :
            Yield = profit_net / volume_total_misé * 100
            Mesure l'efficacité par euro misé (standard des tipsters).

        Win Rate :
            Pourcentage de paris gagnants.

        Sharpe-like :
            Ratio profit / volatilité des P&L (indicateur de régularité).
        """
        if not self.bets:
            return {"error": "Aucun pari effectué."}

        total_staked = sum(b.stake for b in self.bets)
        total_pnl = sum(b.pnl for b in self.bets)
        wins = sum(1 for b in self.bets if b.won)
        n_bets = len(self.bets)

        roi = (self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100
        yield_pct = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        win_rate = wins / n_bets * 100 if n_bets > 0 else 0

        pnls = np.array([b.pnl for b in self.bets])
        sharpe = (np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0

        avg_odd = np.mean([b.odd for b in self.bets])
        avg_edge = np.mean([b.edge for b in self.bets]) * 100
        avg_stake = np.mean([b.stake for b in self.bets])

        max_bankroll = max(self.history.values)
        min_bankroll = min(self.history.values)
        max_drawdown = (max_bankroll - min_bankroll) / max_bankroll * 100

        return {
            "bankroll_initiale": self.initial_bankroll,
            "bankroll_finale": self.bankroll,
            "profit_net": total_pnl,
            "roi_pct": roi,
            "yield_pct": yield_pct,
            "n_paris": n_bets,
            "n_gagnes": wins,
            "n_perdus": n_bets - wins,
            "win_rate_pct": win_rate,
            "volume_total_mise": total_staked,
            "mise_moyenne": avg_stake,
            "cote_moyenne": avg_odd,
            "edge_moyen_pct": avg_edge,
            "matchs_ignores": self.skipped_matches,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
        }

    def print_report(self) -> None:
        """Affiche un rapport complet et lisible en console."""
        m = self.compute_metrics()
        if "error" in m:
            print(f"\n⚠️  {m['error']}")
            return

        profit_emoji = "📈" if m["profit_net"] >= 0 else "📉"
        roi_sign = "+" if m["roi_pct"] >= 0 else ""
        yield_sign = "+" if m["yield_pct"] >= 0 else ""

        print(f"\n{'═'*60}")
        print(f"  📋 RAPPORT DE BACKTESTING - SAISON 2025")
        print(f"{'═'*60}")
        print(f"\n  {profit_emoji} PERFORMANCE FINANCIÈRE")
        print(f"  {'─'*40}")
        print(f"  Bankroll initiale     : {m['bankroll_initiale']:>10.2f}€")
        print(f"  Bankroll finale       : {m['bankroll_finale']:>10.2f}€")
        print(f"  Profit net            : {m['profit_net']:>+10.2f}€")
        print(f"  ROI                   : {roi_sign}{m['roi_pct']:>9.2f}%")
        print(f"  Yield                 : {yield_sign}{m['yield_pct']:>9.2f}%")
        print(f"  Max Drawdown          : {m['max_drawdown_pct']:>9.2f}%")
        print(f"  Sharpe Ratio          : {m['sharpe_ratio']:>10.3f}")

        print(f"\n  🎲 STATISTIQUES DES PARIS")
        print(f"  {'─'*40}")
        print(f"  Paris joués           : {m['n_paris']:>10}")
        print(f"  Paris gagnés          : {m['n_gagnes']:>10}")
        print(f"  Paris perdus          : {m['n_perdus']:>10}")
        print(f"  Win Rate              : {m['win_rate_pct']:>9.2f}%")
        print(f"  Matchs ignorés        : {m['matchs_ignores']:>10}")

        print(f"\n  💰 PARAMÈTRES DE MISE")
        print(f"  {'─'*40}")
        print(f"  Volume total misé     : {m['volume_total_mise']:>10.2f}€")
        print(f"  Mise moyenne          : {m['mise_moyenne']:>10.2f}€")
        print(f"  Cote moyenne          : {m['cote_moyenne']:>10.2f}")
        print(f"  Edge moyen            : {m['edge_moyen_pct']:>9.2f}%")
        print(f"{'═'*60}\n")

    def export_results(self, output_path: str = "backtesting_results.csv") -> None:
        """Exporte les résultats détaillés dans un CSV."""
        if not self.bets:
            print("⚠️  Aucun pari à exporter.")
            return

        rows = []
        for b in self.bets:
            rows.append({
                "date": b.date,
                "match": f"{b.home_team} vs {b.away_team}",
                "pari": b.outcome,
                "prob_ai": round(b.ai_prob, 4),
                "prob_bookmaker": round(b.bookmaker_prob, 4),
                "cote": b.odd,
                "edge_pct": round(b.edge * 100, 2),
                "mise": round(b.stake, 2),
                "resultat": b.result,
                "gagne": b.won,
                "pnl": round(b.pnl, 2),
            })

        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"  💾 Résultats exportés : {output_path}")

    def export_bankroll_history(self, output_path: str = "bankroll_history.csv") -> None:
        """Exporte l'historique de la bankroll pour visualisation."""
        df = pd.DataFrame({
            "date": self.history.dates,
            "bankroll": self.history.values,
            "n_bets": self.history.bets_placed,
        })
        df.to_csv(output_path, index=False)
        print(f"  💾 Historique bankroll : {output_path}")


# ─────────────────────────────────────────────
#  GÉNÉRATION DE DONNÉES DE DÉMONSTRATION
# ─────────────────────────────────────────────

def generate_demo_data(n_matches: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Génère un dataset fictif pour tester le simulateur.
    Remplace par ton CSV Kaggle réel en production.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2025-01-10", periods=n_matches, freq="3D")
    teams = [
        "PSG", "Lyon", "Marseille", "Monaco", "Lille",
        "Rennes", "Nice", "Lens", "Strasbourg", "Nantes",
    ]

    rows = []
    for i, date in enumerate(dates):
        home, away = rng.choice(teams, size=2, replace=False)

        # Probabilités réelles simulées
        true_probs = rng.dirichlet([2, 1, 1.5])  # [home, draw, away]

        # L'IA est légèrement meilleure que les bookmakers (~60% du temps)
        ai_noise = rng.normal(0, 0.04, size=3)
        ai_probs = np.clip(true_probs + ai_noise, 0.01, 0.99)
        ai_probs /= ai_probs.sum()

        # Cotes bookmaker avec marge (~5%)
        bk_probs = true_probs + rng.normal(0, 0.02, size=3)
        bk_probs = np.clip(bk_probs, 0.05, 0.95)
        bk_probs /= bk_probs.sum()
        margin = 1.05
        odds = (1.0 / (bk_probs * margin)).round(2)

        # Résultat réel tiré selon les vraies probs
        result = rng.choice(["H", "D", "A"], p=true_probs)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "match_id": f"M{i+1:04d}",
            "home_team": home,
            "away_team": away,
            "prob_home": round(float(ai_probs[0]), 4),
            "prob_draw": round(float(ai_probs[1]), 4),
            "prob_away": round(float(ai_probs[2]), 4),
            "odd_home": float(odds[0]),
            "odd_draw": float(odds[1]),
            "odd_away": float(odds[2]),
            "result": result,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulateur de Backtesting - Paris Sportifs (Value Bets + Kelly)"
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Chemin vers ton CSV de prédictions Kaggle")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Budget de départ en euros (défaut: 1000)")
    parser.add_argument("--kelly-fraction", type=float, default=0.5,
                        help="Fraction de Kelly, entre 0 et 1 (défaut: 0.5)")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Edge minimum pour parier, entre 0 et 1 (défaut: 0.05)")
    parser.add_argument("--max-bet-pct", type=float, default=0.10,
                        help="Mise max en %% de la bankroll (défaut: 0.10)")
    parser.add_argument("--export", action="store_true",
                        help="Exporter les résultats en CSV")
    parser.add_argument("--quiet", action="store_true",
                        help="Mode silencieux (pas de détail pari par pari)")
    args = parser.parse_args()

    # ── Chargement des données ──
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"❌ Fichier introuvable : {args.csv}")
            sys.exit(1)
        print(f"📂 Chargement : {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        print("⚠️  Aucun CSV fourni → utilisation de données de démonstration (100 matchs)")
        df = generate_demo_data(n_matches=100)

    # ── Vérification des colonnes ──
    required_cols = [
        "prob_home", "prob_draw", "prob_away",
        "odd_home", "odd_draw", "odd_away",
        "result",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Colonnes manquantes dans le CSV : {missing}")
        sys.exit(1)

    # ── Simulation ──
    bettor = Bettor(
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        min_edge=args.min_edge,
        max_bet_pct=args.max_bet_pct,
    )

    bettor.run(df, verbose=not args.quiet)
    bettor.print_report()

    # ── Export optionnel ──
    if args.export:
        bettor.export_results("backtesting_results.csv")
        bettor.export_bankroll_history("bankroll_history.csv")


if __name__ == "__main__":
    main()
