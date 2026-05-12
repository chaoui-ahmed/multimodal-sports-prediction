import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from data_loader import MultimodalDataset, DataLoader
from lstm_model import MultimodalFootballModel
import os

# Créer le dossier pour les figures si il n'existe pas
os.makedirs('reports/figures', exist_ok=True)

def generate_graphs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Charger les données (On prend une partie pour l'évaluation)
    dataset = MultimodalDataset("data/processed/train_stats.csv", "data/historical_embeddings.pt")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # 2. Charger le modèle entraîné
    model = MultimodalFootballModel().to(device)
    model.load_state_dict(torch.load("src/models/multimodal_v1.pth"))
    model.eval()
    
    # 3. Récupérer les prédictions
    stats, nlp, targets = next(iter(loader))
    stats, nlp = stats.to(device), nlp.to(device)
    
    with torch.no_grad():
        outputs = model(stats, nlp)
        _, preds = torch.max(outputs, 1)
    
    y_true = targets.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # --- GRAPH 1 : Matrice de Confusion ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Win', 'Draw', 'Loss'], 
                yticklabels=['Win', 'Draw', 'Loss'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Multimodal Model Confusion Matrix')
    plt.savefig('reports/figures/confusion_matrix.png')
    print("✅ Matrice de confusion sauvegardée.")

    # --- GRAPH 2 : Courbe de Loss (Basée sur tes logs de tout à l'heure) ---
    plt.figure(figsize=(10, 5))
    epochs = range(1, 11)
    # On utilise les chiffres réels que tu as obtenus
    baseline_loss = [1.0658, 1.0645, 1.0638, 1.0632, 1.0628, 1.0625, 1.0622, 1.0621, 1.0620, 1.0620]
    multimodal_loss = [1.0666, 1.0621, 1.0632, 1.0630, 1.0623, 1.0632, 1.0598, 1.0611, 1.0616, 1.0609]
    
    plt.plot(epochs, baseline_loss, label='Baseline (Stats Only)', marker='o', linestyle='--')
    plt.plot(epochs, multimodal_loss, label='Multimodal (Stats + NLP)', marker='s', color='green')
    plt.axhline(y=1.060, color='r', linestyle=':', label='Target Threshold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Convergence: Baseline vs Multimodal')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/loss_comparison.png')
    print("✅ Courbe de loss sauvegardée.")

if __name__ == "__main__":
    generate_graphs()