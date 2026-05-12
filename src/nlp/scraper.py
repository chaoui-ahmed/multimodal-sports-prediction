import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# All sources
RSS_FEEDS = {
    "ESPN (International)": "https://www.espn.com/espn/rss/soccer/news",
    "Foot Mercato (France)": "https://www.footmercato.net/flux-rss", 
    "BBC Sport (UK)": "http://feeds.bbci.co.uk/sport/football/rss.xml"
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
}

def fetch_rss_headlines(source_name, url):
    """Récupère les articles d'un flux RSS spécifique."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Erreur {response.status_code} avec {source_name}")
            return []

        soup = BeautifulSoup(response.content, features="xml")
        articles = []
        
        for item in soup.findAll('item'):
            # Certains flux utilisent 'description', d'autres non. On sécurise le code.
            desc = item.description.text if item.description else ""
            
            articles.append({
                'source': source_name,
                'date': item.pubDate.text if item.pubDate else "Date inconnue",
                'title': item.title.text,
                'description': desc
            })
            
        return articles
        
    except Exception as e:
        print(f"Impossible de scraper {source_name} : {e}")
        return []

if __name__ == "__main__":
    print("Démarrage du scraping multi-sources...\n")
    all_news = []
    
    # On boucle sur toutes nos sources
    for name, url in RSS_FEEDS.items():
        print(f"Scraping en cours : {name}...")
        news_from_source = fetch_rss_headlines(name, url)
        all_news.extend(news_from_source)
    
    print(f"\nTerminé ! Total d'articles récupérés : {len(all_news)}")
    
    # Petit aperçu rapide des données mixées
    print("\n--- Aperçu des 3 premiers articles ---")
    for article in all_news[:3]:
        print(f"[{article['source']}] {article['title']}")

    df = pd.DataFrame(all_news)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/latest_news.csv", index=False, encoding='utf-8')
    print("Données sauvegardées dans 'data/latest_news.csv'")