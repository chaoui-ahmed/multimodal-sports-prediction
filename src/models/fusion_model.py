"""
fusion_model.py
---------------
Modèle multimodal qui fusionne :
  - Branche Stats (LSTM d'Estelle) : séquence de matchs → embedding [64]
  - Branche NLP  (RoBERTa d'Archi) : vecteur d'articles → réduit à [64]
  → Fusion par concaténation → [128] → Dense → 3 classes (Win/Draw/Loss)

Usage :
    python fusion_model.py          # lance un test + entraînement demo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fusion_dataloader import MultimodalSportsDataset, STAT_FEATURES

# Taille de l'entrée stats (synchronisé automatiquement avec generate_demo_stats)
STATS_INPUT_SIZE = len(STAT_FEATURES)   # 8 (form, goals_scored, goals_conceded, elo) x 2

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Calculer la loss standard SANS les poids pour extraire la vraie probabilité (pt)
        ce_loss_unweighted = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss_unweighted)
        
        # 2. Appliquer le mécanisme focal
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_unweighted
        
        # 3. Appliquer les poids de classes a posteriori
        if self.weight is not None:
            w = self.weight[targets]
            focal_loss = focal_loss * w
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ══════════════════════════════════════════════════════════════════
# 1. ARCHITECTURE DU MODELE
# ══════════════════════════════════════════════════════════════════

class MultimodalSportsModel(nn.Module):
    """
    Architecture en deux branches + couche de fusion.

    Branche Stats (LSTM)
    ────────────────────
    Input : [batch, seq=5, features=2]  (home_form, away_form)
    → LSTM 2 couches, hidden=64
    → Sortie : embedding_stats [batch, 64]

    Branche NLP (projection linéaire)
    ───────────────────────────────────
    Input : [batch, 768]   (vecteur CLS de RoBERTa)
    → Linear(768 → 256) + ReLU + Dropout
    → Linear(256 → 64)  + ReLU
    → Sortie : embedding_nlp [batch, 64]

    Fusion
    ──────
    Concat([embedding_stats, embedding_nlp])  → [batch, 128]
    → Linear(128 → 64) + ReLU + Dropout(0.3)
    → Linear(64  → 3)                         → logits Win/Draw/Loss
    """

    def __init__(self,
                 stats_input_size=STATS_INPUT_SIZE,
                 lstm_hidden=128,
                 lstm_layers=2,
                 nlp_input_size=768,
                 nlp_hidden=256,
                 fusion_hidden=128,
                 num_classes=3,
                 dropout=0.3):
        super().__init__()

        # ── Branche 1 : LSTM (stats)
        self.lstm = nn.LSTM(
            input_size=stats_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)  # stabilise le LSTM

        # ── Branche 2 : Projection NLP (embeddings d'Archi)
        self.nlp_projector = nn.Sequential(
            nn.Linear(nlp_input_size, nlp_hidden),
            nn.LayerNorm(nlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nlp_hidden, lstm_hidden),
            nn.ReLU(),
        )

        # ── Couche de fusion
        fusion_input = lstm_hidden + lstm_hidden   # 128 + 128 = 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, stats_seq, nlp_vec):
        # -- Branche stats : on prend uniquement la sortie du DERNIER pas de temps
        lstm_out, _ = self.lstm(stats_seq)          # [batch, seq, 128]
        embedding_stats = self.lstm_norm(lstm_out[:, -1, :])  # [batch, 128]

        # -- Branche NLP : projection 768 → 128
        embedding_nlp = self.nlp_projector(nlp_vec) # [batch, 128]

        # -- Fusion : concaténation puis classification
        fused  = torch.cat([embedding_stats, embedding_nlp], dim=1)  # [batch, 256]
        logits = self.fusion(fused)                 # [batch, 3]
        return logits


# ══════════════════════════════════════════════════════════════════
# 2. BOUCLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════

def train(csv_train="data/processed/train_stats.csv",
          csv_test="data/processed/test_stats.csv",
          recent_emb="data/news_embeddings.pt",
          historical_emb="data/historical_embeddings.pt",
          epochs=20,
          batch_size=32,
          lr=5e-4,
          save_path="src/models/multimodal_v1.pth"):

    import numpy as np
    from fusion_dataloader import compute_class_weights

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n\u26f3️  Entraînement sur : {device}")

    print("\n📂 Chargement du dataset d'entraînement...")
    train_dataset = MultimodalSportsDataset(csv_train, recent_emb, historical_emb)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("\n📂 Chargement du dataset de test...")
    test_dataset = MultimodalSportsDataset(csv_test, recent_emb, historical_emb)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Diagnostic NLP coverage
    nlp_nonzero = sum(1 for v in train_dataset.nlp_vectors if v.abs().sum() > 0)
    print(f"\n📡 Couverture NLP : {nlp_nonzero}/{len(train_dataset.nlp_vectors)} "
          f"vecteurs non-nuls ({100*nlp_nonzero/max(1,len(train_dataset.nlp_vectors)):.1f}%)")

    # Distribution des labels
    labels_all = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    counts = np.bincount(labels_all, minlength=3)
    total  = sum(counts)
    print(f"📊 Distribution train → Win: {counts[0]} ({100*counts[0]//total}%) | "
          f"Draw: {counts[1]} ({100*counts[1]//total}%) | "
          f"Loss: {counts[2]} ({100*counts[2]//total}%)")

    model = MultimodalSportsModel().to(device)
    print(f"\n🧠 Paramètres du modèle : {sum(p.numel() for p in model.parameters()):,}")

    # Poids de classes (corrige le biais vers 'Win')
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer     = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    LABELS = ["Win", "Draw", "Loss"]
    best_val_acc = 0.0

    print(f"\n{'─'*70}")
    print(f"  {'Ep':>3} | {'Loss':>7} | {'Train':>6} | {'Val':>6} "
          f"| {'->Win':>5} {'->Draw':>6} {'->Loss':>6} | {'LR':>8}")
    print(f"{'─'*70}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for stats_seq, nlp_vec, labels in train_loader:
            stats_seq = stats_seq.to(device)
            nlp_vec   = nlp_vec.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            logits = model(stats_seq, nlp_vec)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)
            train_loss    += loss.item()

        model.eval()
        val_preds_all, val_labels_all = [], []
        with torch.no_grad():
            for stats_seq, nlp_vec, labels in test_loader:
                logits = model(stats_seq.to(device), nlp_vec.to(device))
                val_preds_all.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels_all.extend(labels.numpy())

        val_preds  = np.array(val_preds_all)
        val_labels = np.array(val_labels_all)
        val_acc    = 100 * (val_preds == val_labels).mean()
        train_acc  = 100 * train_correct / train_total
        avg_loss   = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        pred_pcts  = 100 * np.bincount(val_preds, minlength=3) / len(val_preds)

        marker = " ★" if val_acc > best_val_acc else ""
        print(f"  {epoch:>3} | {avg_loss:>7.4f} | {train_acc:>5.1f}% | {val_acc:>5.1f}%"
              f" | {pred_pcts[0]:>4.0f}%  {pred_pcts[1]:>4.0f}%    {pred_pcts[2]:>4.0f}%"
              f" | {current_lr:.1e}{marker}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"{'─'*70}")
    print(f"\n✅ Entraînement terminé. Meilleure val acc : {best_val_acc:.1f}%")
    print(f"   Modèle sauvegardé → {save_path}")

    # Rapport final
    from sklearn.metrics import classification_report, confusion_matrix
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    fp, fl = [], []
    with torch.no_grad():
        for stats_seq, nlp_vec, labels in test_loader:
            logits = model(stats_seq.to(device), nlp_vec.to(device))
            fp.extend(logits.argmax(dim=1).cpu().numpy())
            fl.extend(labels.numpy())

    print("\n📋 Rapport de classification (meilleur modèle) :")
    print(classification_report(fl, fp, target_names=LABELS, zero_division=0))
    cm = confusion_matrix(fl, fp)
    print("  Matrice de confusion (lignes=réel, cols=prédit) :")
    print(f"         Win  Draw  Loss")
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:<5} {row[0]:>4}  {row[1]:>4}  {row[2]:>4}")

    return model


# ══════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION TEMPORELLE (TimeSeriesSplit)
# ══════════════════════════════════════════════════════════════════

def cross_validate(df,
                   all_dates,
                   embedding_matrix,
                   n_splits=5,
                   epochs=15,
                   batch_size=32,
                   lr=5e-4,
                   save_path="src/models/multimodal_v1.pth"):
    """
    Cross-validation temporelle sur K folds glissants.

    Chaque fold :
      - Train = les matchs des folds précédents (passé)
      - Val   = le fold suivant (futur immédiat)

    Cela simule la vraie situation : prédire sur du futur non vu.

    Args:
        df              : DataFrame complet avec home_form, away_form, scores, dates
        all_dates       : liste de dates des articles NLP
        embedding_matrix: tenseur [N, 768] des embeddings NLP
        n_splits        : nombre de folds temporels
        epochs          : epochs par fold
        batch_size      : taille de batch
        lr              : learning rate
        save_path       : où sauvegarder le meilleur modèle global

    Returns:
        model entraîné sur toutes les données (retrain final)
    """
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import f1_score, accuracy_score
    from fusion_dataloader import MultimodalSportsDatasetFromDF, compute_class_weights

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Cross-validation sur : {device}")
    print(f"   {n_splits} folds temporels | {epochs} epochs/fold | LR={lr}")

    df = df.sort_values('date').reset_index(drop=True)
    indices = np.arange(len(df))

    tscv = TimeSeriesSplit(n_splits=n_splits)
    LABELS = ["Win", "Draw", "Loss"]

    fold_results = []

    print(f"\n{'═'*70}")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(indices), 1):
        df_train = df.iloc[train_idx]
        df_val   = df.iloc[val_idx]

        print(f"\n  📅 Fold {fold}/{n_splits}")
        print(f"     Train : {len(df_train)} matchs  "
              f"({df_train['date'].min().strftime('%d/%m/%Y')} → "
              f"{df_train['date'].max().strftime('%d/%m/%Y')})")
        print(f"     Val   : {len(df_val)} matchs  "
              f"({df_val['date'].min().strftime('%d/%m/%Y')} → "
              f"{df_val['date'].max().strftime('%d/%m/%Y')})")

        # Datasets
        train_ds = MultimodalSportsDatasetFromDF(df_train, all_dates, embedding_matrix)
        val_ds   = MultimodalSportsDatasetFromDF(df_val,   all_dates, embedding_matrix)

        if len(train_ds) < batch_size or len(val_ds) < 1:
            print(f"     ⚠️  Fold trop petit, ignoré.")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # Modèle frais pour chaque fold
        model     = MultimodalSportsModel().to(device)
        cw        = compute_class_weights(train_ds).to(device)
        criterion = FocalLoss(weight=cw, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        print(f"\n     {'Ep':>3} | {'Loss':>7} | {'Train':>6} | {'Val':>6} "
              f"| {'->W':>4} {'->D':>4} {'->L':>4}")
        print(f"     {'─'*50}")

        best_fold_acc = 0.0
        best_fold_preds = None

        for epoch in range(1, epochs + 1):
            model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0

            for stats_seq, nlp_vec, labels in train_loader:
                stats_seq, nlp_vec, labels = (
                    stats_seq.to(device), nlp_vec.to(device), labels.to(device)
                )
                optimizer.zero_grad()
                logits = model(stats_seq, nlp_vec)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                preds       = logits.argmax(dim=1)
                tr_correct += (preds == labels).sum().item()
                tr_total   += labels.size(0)
                tr_loss    += loss.item()

            model.eval()
            vp, vl = [], []
            with torch.no_grad():
                for stats_seq, nlp_vec, labels in val_loader:
                    logits = model(stats_seq.to(device), nlp_vec.to(device))
                    vp.extend(logits.argmax(dim=1).cpu().numpy())
                    vl.extend(labels.numpy())

            val_acc   = 100 * accuracy_score(vl, vp)
            tr_acc    = 100 * tr_correct / tr_total
            avg_loss  = tr_loss / len(train_loader)
            pp        = 100 * np.bincount(np.array(vp), minlength=3) / len(vp)
            current_lr = optimizer.param_groups[0]['lr']

            marker = " ★" if val_acc > best_fold_acc else ""
            print(f"     {epoch:>3} | {avg_loss:>7.4f} | {tr_acc:>5.1f}% | {val_acc:>5.1f}%"
                  f" | {pp[0]:>3.0f}% {pp[1]:>3.0f}% {pp[2]:>3.0f}%{marker}")

            scheduler.step()

            if val_acc > best_fold_acc:
                best_fold_acc   = val_acc
                best_fold_preds = (np.array(vl), np.array(vp))

        if best_fold_preds is not None:
            yl, yp = best_fold_preds
            f1_macro = f1_score(yl, yp, average='macro', zero_division=0)
            f1_per   = f1_score(yl, yp, average=None,    zero_division=0, labels=[0,1,2])
            fold_results.append({
                'fold': fold,
                'acc':  best_fold_acc,
                'f1':   f1_macro,
                'f1_win': f1_per[0], 'f1_draw': f1_per[1], 'f1_loss': f1_per[2],
            })
            print(f"\n     Fold {fold} meilleur → Acc: {best_fold_acc:.1f}% | "
                  f"F1-macro: {f1_macro:.3f} "
                  f"(Win:{f1_per[0]:.2f} Draw:{f1_per[1]:.2f} Loss:{f1_per[2]:.2f})")

    # ── Résumé des folds
    print(f"\n{'═'*70}")
    print("  📊 RÉSUMÉ CROSS-VALIDATION")
    print(f"{'─'*70}")
    print(f"  {'Fold':>5} | {'Acc':>6} | {'F1':>6} | {'F1-Win':>7} | {'F1-Draw':>8} | {'F1-Loss':>8}")
    print(f"{'─'*70}")
    for r in fold_results:
        print(f"  {r['fold']:>5} | {r['acc']:>5.1f}% | {r['f1']:>6.3f} "
              f"| {r['f1_win']:>7.3f} | {r['f1_draw']:>8.3f} | {r['f1_loss']:>8.3f}")

    if fold_results:
        avg_acc = np.mean([r['acc']  for r in fold_results])
        avg_f1  = np.mean([r['f1']   for r in fold_results])
        std_acc = np.std( [r['acc']  for r in fold_results])
        print(f"{'─'*70}")
        print(f"  Moyenne : Acc = {avg_acc:.1f}% ± {std_acc:.1f}% | F1-macro = {avg_f1:.3f}")

    # ── Retrain final sur toutes les données
    print(f"\n{'═'*70}")
    print("  🔁 RETRAIN FINAL sur toutes les données...")
    full_ds     = MultimodalSportsDatasetFromDF(df, all_dates, embedding_matrix)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model     = MultimodalSportsModel().to(device)
    cw        = compute_class_weights(train_ds).to(device)
    criterion = FocalLoss(weight=cw, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"  {'Ep':>3} | {'Loss':>7} | {'Acc':>6}")
    print(f"  {'─'*25}")

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for stats_seq, nlp_vec, labels in full_loader:
            stats_seq, nlp_vec, labels = (
                stats_seq.to(device), nlp_vec.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            logits = model(stats_seq, nlp_vec)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            preds   = logits.argmax(dim=1)
            t_corr += (preds == labels).sum().item()
            t_tot  += labels.size(0)
            t_loss += loss.item()
        print(f"  {epoch:>3} | {t_loss/len(full_loader):>7.4f} | {100*t_corr/t_tot:>5.1f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\n  ✅ Modèle final sauvegardé → {save_path}")
    return model


# ══════════════════════════════════════════════════════════════════
# 3. INFERENCE (prédire un seul match)
# ══════════════════════════════════════════════════════════════════

def predict_match(model, stats_seq_np, nlp_vec_tensor, device='cpu'):
    model.eval()
    with torch.no_grad():
        stats = torch.tensor(stats_seq_np, dtype=torch.float32).unsqueeze(0).to(device)
        nlp   = nlp_vec_tensor.unsqueeze(0).to(device)
        logits = model(stats, nlp)
        probs  = torch.softmax(logits, dim=1)[0]

    labels = ["Win (domicile)", "Draw", "Loss (domicile)"]
    
    # Probabilités brutes
    p_win, p_draw, p_loss = probs[0].item(), probs[1].item(), probs[2].item()
    
    # Threshold Moving : Seuils asymétriques basés sur la fréquence naturelle
    # On abaisse l'exigence pour prédire un Nul ou une Défaite
    pred_idx = 0 # Défaut: Win
    
    if p_draw >= 0.28:  # Si le Nul dépasse 28% (au lieu de 33.3%), on le tente
        pred_idx = 1
    elif p_loss >= 0.35: # Si la Défaite dépasse 35%, on la tente
        pred_idx = 2
    else:
        pred_idx = 0 # Sinon, Victoire à domicile

    # Optionnel : si le modèle est VRAIMENT confiant sur Win, on override
    if p_win > 0.55:
        pred_idx = 0

    return {
        "prediction": labels[pred_idx],
        "probabilités": {labels[i]: f"{probs[i].item()*100:.1f}%" for i in range(3)}
    }


# ══════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test de l'architecture avant d'entraîner
    print("── Test de l'architecture ──")
    model_test = MultimodalSportsModel()
    dummy_stats = torch.randn(4, 5, 2)   # batch=4, seq=5, features=2
    dummy_nlp   = torch.randn(4, 768)    # batch=4, nlp_dim=768
    out = model_test(dummy_stats, dummy_nlp)
    print(f"✅ Forward pass OK → shape sortie : {out.shape}")   # [4, 3]
    print(f"   Paramètres totaux : {sum(p.numel() for p in model_test.parameters()):,}")

    # Lancement de l'entraînement complet
    trained_model = train()
