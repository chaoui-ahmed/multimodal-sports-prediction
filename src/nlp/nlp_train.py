import os
import torch
import warnings
import pandas as pd
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')

def extract_historical_embeddings():
    print("Téléchargement direct avec Pandas depuis Hugging Face...")
    
    url_dataset = "hf://datasets/ZhangYi0820/Transfermarkt_News_Archive/news.csv"
    df = pd.read_csv(url_dataset)
    
    model_name = "xlm-roberta-base"
    print(f"Chargement de l'IA ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print(f"Extraction des vecteurs pour {len(df)} articles...")
    
    liste_vecteurs = []
    dates_list = []
    
    for index, row in df.iterrows():
        texte = str(row['Title']) + " - " + str(row['Content'])
        date_article = str(row['Time'])
        
        inputs = tokenizer(texte, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        liste_vecteurs.append(cls_embedding)
        dates_list.append(date_article)

        if index % 1000 == 0 and index > 0:
            print(f"Progression : {index} / {len(df)} articles traités...")

    matrice_finale = torch.cat(liste_vecteurs, dim=0)
    print(f"Extraction terminée. Taille : {matrice_finale.shape}")
    
    # Sauvegarde
    dossier_script = os.path.dirname(os.path.abspath(__file__))
    racine_projet = os.path.dirname(os.path.dirname(dossier_script))
    
    os.makedirs(os.path.join(racine_projet, "data"), exist_ok=True)
    chemin_sauvegarde = os.path.join(racine_projet, "data", "historical_embeddings.pt")
    
    donnees_a_sauvegarder = {
        "dates": dates_list,
        "embeddings": matrice_finale
    }
    
    torch.save(donnees_a_sauvegarder, chemin_sauvegarde)
    print(f"Succès ! Fichier final sauvegardé : '{chemin_sauvegarde}'")

if __name__ == "__main__":
    extract_historical_embeddings()