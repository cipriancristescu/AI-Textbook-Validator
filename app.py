import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader
import json
import pandas as pd
import time

# 1. Configurare
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Lipseste cheia API! Verifica fisierul .env")
else:
    genai.configure(api_key=api_key)

st.set_page_config(page_title="AI Manual Validator", layout="wide")

if "data_analiza" not in st.session_state:
    st.session_state.data_analiza = None

# --- PROMPT ACTUALIZAT (FĂRĂ DIACRITICE) ---
SYSTEM_PROMPT = """
Ești un AUDITOR ACADEMIC. Analizează textul educațional.
Sarcina ta este să identifici ERORI DE CONȚINUT.
IGNORĂ complet erorile de scriere, lipsa diacriticelor sau formatarea ciudată.

Concentrează-te DOAR pe:
1. Erori Factuale (Ani, Nume, Locații, Date științifice greșite).
2. Erori Logice & Matematice (Calcule greșite, Contradicții).
3. Anacronisme (Elemente din timpuri greșite).

Răspunde DOAR cu JSON valid:
[
  {
    "pagina": "nr paginii (daca e cazul)", 
    "text_original": "citatul scurt cu eroarea",
    "tip_eroare": "Factuală / Matematică / Logică",
    "explicatie": "motivul pe scurt",
    "sugestie_corectare": "varianta corecta"
  }
]
Dacă nu sunt erori de conținut, returnează [].
"""

# Funcție sigură de parsare JSON
def safe_json_parse(json_string):
    try:
        clean = json_string.replace("```json", "").replace("```", "").strip()
        # Încercăm să reparăm JSON-ul dacă e tăiat
        if not clean.endswith("]"):
             last_brace = clean.rfind("}")
             if last_brace != -1:
                 clean = clean[:last_brace+1] + "]"
        return json.loads(clean)
    except:
        return []

# 2. Interfața
st.title("📚 AI Textbook Validator")
st.markdown("Verifică erori de **conținut** (Istorie, Mate, Științe). Ignoră greșelile gramaticale.")

# Sidebar - Doar butoane de control
with st.sidebar:
    st.header("Control")
    # Buton Reset
    if st.button("🗑️ Șterge Tot / Reset"):
        st.session_state.data_analiza = None
        st.rerun()

# 3. ZONA DE INPUT (HIBRIDĂ)
col_input, col_rezultat = st.columns([1, 1])

with col_input:
    st.subheader("1. Introducere Date")
    
    # TABURI: Alegem între PDF și Text
    tab1, tab2 = st.tabs(["📂 Încărcare PDF", "✍️ Text Manual"])
    
    source_type = None
    uploaded_file = None
    manual_text = ""
    
    with tab1:
        uploaded_file = st.file_uploader("Alege manualul (PDF)", type="pdf")
        if uploaded_file:
            source_type = "pdf"
            st.info("Mod: Procesare Pagină-cu-Pagină (Batch)")
            
    with tab2:
        manual_text = st.text_area("Lipeste textul aici:", height=300, placeholder="Ex: 2 + 2 = 5 sau Ștefan cel Mare a trăit în 2020.")
        if manual_text:
            source_type = "text"
            st.info("Mod: Analiză Rapidă")

    # Butonul unic de start
    start_btn = st.button("🚀 Începe Analiza", type="primary", use_container_width=True)


# 4. LOGICA DE PROCESARE
if start_btn:
    # Curățăm rezultatele vechi
    st.session_state.data_analiza = None
    
    # CAZUL 1: Niciun input
    if not source_type:
        st.warning("Te rog încarcă un PDF sau scrie un text!")
    
    # CAZUL 2: Text Manual (Simplu)
    elif source_type == "text":
        with col_rezultat:
            st.subheader("Rezultate")
            with st.spinner("Analizează textul..."):
                try:
                    model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.0, "response_mime_type": "application/json"})
                    response = model.generate_content(SYSTEM_PROMPT + "\n\nTEXT:\n" + manual_text)
                    errors = safe_json_parse(response.text)
                    
                    if errors:
                        df = pd.DataFrame(errors)
                        df.insert(0, "Validat", False)
                        st.session_state.data_analiza = df
                    else:
                        st.success("Nu s-au găsit erori de conținut! ✅")
                except Exception as e:
                    st.error(f"Eroare: {e}")

    # CAZUL 3: PDF (Batch Processing - Complex)
    elif source_type == "pdf":
        with col_rezultat:
            st.subheader("Progres Analiză")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            reader = PdfReader(uploaded_file)
            total_pages = len(reader.pages)
            all_errors = []
            
            BATCH_SIZE = 5 # Câte pagini analizează o dată
            
            try:
                model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.0, "response_mime_type": "application/json"})
                
                for i in range(0, total_pages, BATCH_SIZE):
                    # Pregătim lotul de text
                    batch_text = ""
                    end_page = min(i + BATCH_SIZE, total_pages)
                    for p_index in range(i, end_page):
                        batch_text += f"[Pagina {p_index+1}]\n" + reader.pages[p_index].extract_text() + "\n"
                    
                    # Update UI
                    status_text.text(f"Scanez paginile {i+1} - {end_page}...")
                    progress_bar.progress(end_page / total_pages)
                    
                    # Call AI
                    try:
                        response = model.generate_content(SYSTEM_PROMPT + "\n\nTEXT:\n" + batch_text)
                        batch_errors = safe_json_parse(response.text)
                        if batch_errors:
                            all_errors.extend(batch_errors)
                    except:
                        continue # Dacă o pagină dă eroare, trecem mai departe
                    
                    time.sleep(0.5) # Pauză mică

                status_text.text("Gata!")
                
                if all_errors:
                    df = pd.DataFrame(all_errors)
                    df.insert(0, "Validat", False)
                    st.session_state.data_analiza = df
                else:
                    st.success("Manualul pare corect din punct de vedere al conținutului! ✅")

            except Exception as e:
                st.error(f"Eroare critică: {e}")


# 5. AFIȘAREA TABELULUI FINAL (Comun pentru ambele cazuri)
if st.session_state.data_analiza is not None:
    st.divider()
    st.subheader(f"📋 Raport Final ({len(st.session_state.data_analiza)} erori)")
    
    edited_df = st.data_editor(
        st.session_state.data_analiza,
        column_config={
            "Validat": st.column_config.CheckboxColumn("Confirm", default=True),
            "pagina": "Pag.",
            "text_original": "Text Original",
            "tip_eroare": "Tip",
            "explicatie": "Explicație",
            "sugestie_corectare": "Corectură",
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Export
    if st.button("Descarcă Lista Aprobată (CSV)"):
        raport = edited_df[edited_df["Validat"] == True]
        csv = raport.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "erata.csv", "text/csv")