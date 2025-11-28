import google.generativeai as genai
import os
from dotenv import load_dotenv

# Încarcă cheia
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("------------------------------------------------")
print("LISTA MODELELOR DISPONIBILE PENTRU TINE:")
print("------------------------------------------------")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Eroare: {e}")

print("------------------------------------------------")