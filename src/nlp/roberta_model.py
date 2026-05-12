import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings('ignore')

def extract_and_save_embeddings(csv_path="../../data/latest_news.csv", save_path="../../data/news_embeddings.pt"):
    print(f"Chargement des articles depuis {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Fichier introuvable. Vérifie ton chemin ../../ !")
        return

    model_name = "xlm-roberta-base"
    print(f"Chargement de l'IA ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print(f"Extraction des vecteurs pour {len(df)} articles (ça peut prendre quelques secondes)...")
    
    liste_vecteurs = []
    
    # On boucle sur TOUS les articles
    for index, row in df.iterrows():
        texte = row['title']
        
        inputs = tokenizer(texte, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # On extrait le token [CLS]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # On ajoute ce vecteur à notre liste
        liste_vecteurs.append(cls_embedding)

    # On colle tous les petits vecteurs (1, 768) en une seule grosse matrice
    matrice_finale = torch.cat(liste_vecteurs, dim=0)
    
    print(f"Extraction finie")
    
    # --- SAUVEGARDE ---
    # On sauvegarde un dictionnaire PyTorch qui contient les titres ET les matrices
    donnees_a_sauvegarder = {
        "titres": df['title'].tolist(),
        "sources": df['source'].tolist(),
        "dates": df['date'].tolist(),
        "embeddings": matrice_finale
    }
    
    torch.save(donnees_a_sauvegarder, save_path)
    print(f"Fini ! Sauvegardé à '{save_path}'")

if __name__ == "__main__":
    extract_and_save_embeddings()