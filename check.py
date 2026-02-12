import requests
import os
from dotenv import load_dotenv

# 1. Încarcă cheia API
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

if not api_key:
    print("❌ EROARE: Nu am găsit TOGETHER_API_KEY în .env")
    exit()

url = "https://api.together.xyz/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print("⏳ Descarc lista BRUTĂ de la Together AI (fără filtre)...\n")

try:
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Eroare API: {response.status_code}")
        print(response.text)
        exit()
        
    models = response.json()
    
    # Colectăm doar numele și tipul pentru afișare
    model_list = []
    for m in models:
        mid = m.get('id', 'N/A')
        mtype = m.get('type', 'unknown')
        # Formatăm ca "[TIP] Nume Model"
        model_list.append(f"[{mtype.upper()}] {mid}")

    # Sortăm alfabetic ca să fie ușor de citit
    model_list.sort()

    # --- AFIȘARE ---
    print(f"✅ Total modele accesibile: {len(models)}")
    print("="*80)
    
    for m_name in model_list:
        print(m_name)
        
    print("="*80)
    print("🔍 SUGERARE: Dă scroll și caută modele care conțin 'Vision', 'VL' sau 'Qwen'.")

except Exception as e:
    print(f"❌ Eroare critică: {e}")