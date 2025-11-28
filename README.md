# AI Textbook Validator

Acest proiect este un tool educațional bazat pe AI (**Google Gemini 2.0 Flash**) care asistă profesorii în verificarea manualelor școlare. Aplicația detectează automat erori de conținut, anacronisme și probleme logice, ignorând greșelile gramaticale minore.

##  Funcționalități
- **Human-in-the-loop:** Profesorul validează sugestiile AI-ului.
- **Batch Processing:** Poate analiza manuale PDF întregi, pagină cu pagină.
- **Mod Hibrid:** Acceptă upload de PDF sau text introdus manual.
- **Filtrare Inteligentă:** Ignoră problemele de encoding (diacritice) din PDF-uri.

##  Tehnologii Folosite
- **Python 2**
- **Streamlit** (Interfață Grafică)
- **Google Gemini API** (LLM Engine)
- **PyPDF2** (Procesare documente)

##  Cum se rulează local

1. Clonează acest repository.
2. Instalează dependențele:
   ```bash
   pip install -r requirements.txt
