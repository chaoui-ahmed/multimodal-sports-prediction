import torch
import torch.nn as nn
import torch.optim as optim
# On importe le Dataset multimodal et le modèle de fusion
from data_loader import MultimodalDataset, DataLoader
from lstm_model import MultimodalFootballModel

def train_model():
    # 1. Configuration - On affine les paramètres
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 20           # On double pour laisser le temps au LR plus faible d'apprendre
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001 # On divise par 10 pour plus de précision (évite le collapse)
    
    # Chemins des fichiers
    STATS_PATH = "data/processed/train_stats.csv"
    EMBEDDINGS_PATH = "data/historical_embeddings.pt"

    # 2. Chargement des données multimodales
    print("Chargement du dataset multimodal...")
    # On s'assure que shuffle=True est bien là pour mélanger les classes à chaque epoch
    train_dataset = MultimodalDataset(STATS_PATH, EMBEDDINGS_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialisation du modèle multimodal
    model = MultimodalFootballModel().to(device)
    
    # --- STRATÉGIE ANTI-BIAIS ---
    # On ajuste les poids pour être moins agressif que 2.5, 
    # mais assez pour sortir du "tout Win"
    weights = torch.tensor([1.0, 1.8, 2.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Utilisation d'Adam avec un LR plus petit
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Début de l'entraînement multimodal sur {device}...")
    print(f"Paramètres : LR={LEARNING_RATE}, Epochs={EPOCHS}, Weights={weights.tolist()}")

    # 4. Boucle d'entraînement
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (stats, nlp, target) in enumerate(train_loader):
            stats, nlp, target = stats.to(device), nlp.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (Fusion Stats + NLP)
            output = model(stats, nlp)
            
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optionnel mais recommandé pour stabiliser le LSTM)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # 5. Sauvegarde du modèle final
    torch.save(model.state_dict(), "src/models/multimodal_v1.pth")
    print("\n✅ Entraînement terminé.")
    print("✅ Modèle multimodal sauvegardé sous src/models/multimodal_v1.pth")

if __name__ == "__main__":
    train_model()