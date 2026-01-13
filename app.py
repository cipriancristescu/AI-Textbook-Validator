import streamlit as st
import fitz  # PyMuPDF
import os
import json
import requests
import pandas as pd
import base64
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURARE ---
load_dotenv()
st.set_page_config(page_title="AI Manual Auditor", layout="wide", initial_sidebar_state="expanded")

# Verificare chei API
if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.error("Lipsesc cheile API din fișierul .env")
    st.stop()

# Inițializare State
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "stage": "upload",
        "pdf_bytes": None,
        "chapters": [],
        "structure_data": [],
        "final_report": [],
        "debug_img": None
    }

# --- CLASA CORE: AGENT COUNCIL ---

class AgentCouncil:
    def __init__(self):
        self.or_key = os.getenv("OPENROUTER_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.or_key}",
            "HTTP-Referer": "http://localhost:8501",
            "Content-Type": "application/json"
        }

    def _call_llm(self, model, system_prompt, user_text):
        """Wrapper generic pentru OpenRouter."""
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                "response_format": {"type": "json_object"}
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            return json.loads(resp.json()['choices'][0]['message']['content'])
        except Exception as e:
            return {}

    def _tavily_search(self, query):
        """Caută pe web folosind Tavily."""
        try:
            resp = requests.post("https://api.tavily.com/search", json={
                "api_key": self.tavily_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": 3
            })
            results = resp.json().get('results', [])
            return "\n".join([f"- {r['content']} (Sursă: {r['url']})" for r in results])
        except:
            return None

    # --- AGENT 1: NUANȚE & BIAS (Claude 3.5 Sonnet) ---
    def agent_bias(self, text):
        prompt = """
        Ești un critic istoric și literar. Analizează textul pentru Propagandă, Bias și Limbaj Absolutist.
        Identifică omisiuni majore de context sau interpretări naționaliste excesive.
        Format JSON: {"erori": [{"tip": "Bias/Nuanță", "text": "...", "sugestie": "...", "explicatie": "..."}]}
        """
        res = self._call_llm("anthropic/claude-3.5-sonnet", prompt, text[:15000])
        return res.get("erori", [])

    # --- AGENT 2: FACTUAL LOCAL & GRAMATICĂ (GPT-4o) ---
    def agent_local_facts(self, text):
        extract_prompt = "Extrage 3-5 afirmații factuale verificabile (ani, nume, tratate) din text. JSON: {'claims': []}"
        claims_res = self._call_llm("openai/gpt-4o", extract_prompt, text[:10000])
        claims = claims_res.get("claims", [])
        
        evidence = ""
        if claims:
            for c in claims:
                res = self._tavily_search(f"Verifică istoric România: {c}")
                if res: evidence += f"\nCLAIM: {c}\nEVIDENCE: {res}\n"

        verify_prompt = f"""
        Ești editor și fact-checker.
        1. Gramatică: Identifică dezacorduri grave. IGNORĂ erorile de encoding (fonturi stricate).
        2. Fapte: Compară textul cu dovezile. Raportează DOAR contradicții clare. Dacă nu ai dovezi, presupune că textul e corect.
        
        DOVEZI WEB:
        {evidence}
        
        Format JSON: {{"erori": [{{"tip": "Factual/Gramatică", "text": "...", "sugestie": "...", "explicatie": "..."}}]}}
        """
        res = self._call_llm("openai/gpt-4o", verify_prompt, text[:15000])
        return res.get("erori", [])

    # --- AGENT 3: AUDITOR INTERNAȚIONAL (Claude 3.5 + Search EN) ---
    def agent_international(self, text):
        trans_prompt = "Extrage 2-3 teme majore istorice din text și tradu-le în interogări de căutare în ENGLEZĂ. JSON: {'queries': []}"
        q_res = self._call_llm("anthropic/claude-3.5-sonnet", trans_prompt, text[:10000])
        queries = q_res.get("queries", [])
        
        evidence = ""
        if queries:
            for q in queries:
                res = self._tavily_search(q)
                if res: evidence += f"\nQUERY: {q}\nINTL EVIDENCE: {res}\n"
        
        compare_prompt = f"""
        Ești un auditor internațional. Compară narativul din textul românesc cu consensul istoric internațional (dovezile de mai jos).
        Semnalează dacă manualul prezintă o versiune distorsionată sau izolată a istoriei.
        
        DOVEZI INTERNAȚIONALE:
        {evidence}
        
        Format JSON: {{"erori": [{{"tip": "Perspectivă Int.", "text": "...", "sugestie": "...", "explicatie": "..."}}]}}
        """
        res = self._call_llm("anthropic/claude-3.5-sonnet", compare_prompt, text[:15000])
        return res.get("erori", [])

    # --- THE JUDGE: SINTEZĂ (FĂRĂ FILTRU PYTHON EXTERN) ---
    def the_judge(self, all_raw_errors):
        prompt = """
        Ești Editorul Șef Suprem. Ai primit rapoarte de la 3 agenți.
        Sarcina ta: Curăță lista, elimină dublurile și șterge zgomotul.
        
        INSTRUCȚIUNI STRICTE:
        1. ȘTERGE ORICE eroare de encoding/font (ex: 's,coal,', 't]ara', 'm\re', 'þ' în loc de 'ț', litere lipsă). NU le include în raport.
        2. ȘTERGE erorile factuale unde explicația este "nu am găsit sursa".
        3. Păstrează doar problemele reale de conținut (Bias, Fapte greșite, Dezacorduri majore).
        
        Returnează JSON STRICT: 
        {"final_errors": [{"tip": "...", "text": "...", "sugestie": "...", "explicatie": "..."}]}
        """
        input_data = json.dumps(all_raw_errors)
        res = self._call_llm("openai/gpt-4o", prompt, input_data)
        
        # Safety check pentru return
        if isinstance(res, dict) and "final_errors" in res:
            return res["final_errors"]
        elif isinstance(res, list):
            return res
        return []

    def run_ensemble_analysis(self, text, chapter_title):
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_bias = executor.submit(self.agent_bias, text)
            future_local = executor.submit(self.agent_local_facts, text)
            future_intl = executor.submit(self.agent_international, text)
            
            raw_errors = []
            raw_errors.extend(future_bias.result())
            raw_errors.extend(future_local.result())
            raw_errors.extend(future_intl.result())
            
        final_errors = self.the_judge(raw_errors)
        
        for e in final_errors:
            e['capitol'] = chapter_title
            
        return final_errors

# --- MODUL VISION (GPT4o) ---

def extract_structure_vision(pdf_bytes, page_num):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_idx = int(page_num) - 1
        
        if page_idx < 0 or page_idx >= len(doc):
            st.error(f"Pagina {page_num} nu există. Documentul are {len(doc)} pagini.")
            return [], None
        
        pix = doc[page_idx].get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        b64_img = base64.b64encode(img_data).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        prompt_vision = """
        Privește această imagine a unui CUPRINS. Extrage Titlurile și Pagina de Start.
        Ignoră titlul paginii. Caută linii punctate și numere.
        
        Returnează JSON STRICT: 
        { "items": [ {"titlu": "Titlu Capitol", "start": 5} ] }
        """
        
        payload = {
            "model": "openai/gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_vision},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }
        
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        
        if resp.status_code != 200: return [], img_data

        content = json.loads(resp.json()['choices'][0]['message']['content'])
        
        if isinstance(content, dict):
            if 'items' in content: return content['items'], img_data
            for v in content.values():
                if isinstance(v, list): return v, img_data
        
        if isinstance(content, list): return content, img_data
        return [], img_data
        
    except Exception as e:
        return [], None

# --- UTILITARE ---

def extract_chapter_text(pdf_bytes, structure):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    final_chapters = []
    
    structure = sorted(structure, key=lambda x: int(x.get('start', 0)))
    for i in range(len(structure)):
        if 'end' not in structure[i] or structure[i]['end'] == 0:
            if i < len(structure) - 1:
                structure[i]['end'] = int(structure[i+1]['start']) - 1
            else:
                structure[i]['end'] = total
    
    for item in structure:
        start = int(item['start'])
        end = int(item['end'])
        text_chunk = ""
        for p in range(max(0, start-1), min(total, end)):
            text_chunk += f"\n\n[[PAGINA {p+1}]]\n{doc[p].get_text()}"
        
        final_chapters.append({
            "titlu": item.get('titlu', 'Capitol'),
            "interval": f"{start}-{end}",
            "text": text_chunk
        })
    return final_chapters

# --- INTERFAȚA UTILIZATOR (FRONTEND) ---

st.title("AI Manual Auditor")
st.markdown("Sistem Multi-Agent: **GPT-4o Vision** (Structură) | **Claude 3.5** (Nuanțe) | **GPT-4o** (Fapte & Sinteză)")

# 1. UPLOAD
if st.session_state.app_state["stage"] == "upload":
    with st.container(border=True):
        st.subheader("1. Încărcare Document & Structură")
        uploaded = st.file_uploader("Selectează fișier PDF", type="pdf")
        
        if uploaded:
            st.session_state.app_state["pdf_bytes"] = uploaded.getvalue()
            doc_len = len(fitz.open(stream=uploaded.getvalue(), filetype="pdf"))
            
            st.info(f"Document încărcat. Total pagini: {doc_len}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                toc_page = st.number_input(
                    "La ce pagină este Cuprinsul (Index PDF)?", 
                    min_value=1, max_value=doc_len, value=3
                )
                
            with col2:
                st.write("") 
                st.write("") 
                if st.button("📸 Scanează Vizual Cuprinsul", type="primary"):
                    with st.spinner("GPT-4o privește pagina..."):
                        struct, debug_img = extract_structure_vision(uploaded.getvalue(), toc_page)
                        st.session_state.app_state["structure_data"] = struct
                        st.session_state.app_state["debug_img"] = debug_img
            
            # Bloc Validare (Acum este persistent în afara if-ului de buton)
            if st.session_state.app_state["structure_data"]:
                st.success(f"Am detectat {len(st.session_state.app_state['structure_data'])} capitole!")
                if st.session_state.app_state["debug_img"]:
                    with st.expander("Vezi imaginea scanată"):
                        st.image(st.session_state.app_state["debug_img"], use_container_width=True)
                
                if st.button("➡️ Mergi la Validare"):
                    st.session_state.app_state["stage"] = "approve"
                    st.rerun()
            elif st.session_state.app_state["debug_img"]:
                 st.error("Nu am putut detecta structura.")

            st.divider()
            if st.button("Omite Cuprins (Segmentare Automată 20 pag)"):
                 doc = fitz.open(stream=uploaded.getvalue(), filetype="pdf")
                 st.session_state.app_state["structure_data"] = [{"titlu": f"Segment {i+1}", "start": i+1} for i in range(0, len(doc), 20)]
                 st.session_state.app_state["stage"] = "approve"
                 st.rerun()

# 2. APPROVE STRUCTURE
elif st.session_state.app_state["stage"] == "approve":
    with st.container(border=True):
        st.subheader("2. Validare Structură")
        
        df = pd.DataFrame(st.session_state.app_state["structure_data"])
        if 'start' not in df.columns: df['start'] = 1
        if 'end' not in df.columns: df['end'] = 0
        
        edited_df = st.data_editor(
            df,
            column_config={
                "titlu": "Titlu Capitol",
                "start": st.column_config.NumberColumn("Start", format="%d"),
                "end": st.column_config.NumberColumn("Final", format="%d")
            },
            use_container_width=True, num_rows="dynamic"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirmă și Analizează", type="primary"):
                structure = edited_df.to_dict("records")
                st.session_state.app_state["chapters"] = extract_chapter_text(st.session_state.app_state["pdf_bytes"], structure)
                st.session_state.app_state["stage"] = "analyze"
                st.rerun()
        with col2:
            if st.button("Resetează"):
                st.session_state.app_state["stage"] = "upload"
                st.session_state.app_state["structure_data"] = []
                st.session_state.app_state["debug_img"] = None
                st.rerun()

# 3. ANALYZE
elif st.session_state.app_state["stage"] == "analyze":
    chapters = st.session_state.app_state["chapters"]
    council = AgentCouncil()
    
    st.subheader(f"3. Analiză ({len(chapters)} Capitole)")
    
    if not st.session_state.app_state["final_report"]:
        if st.button("Start Audit Multi-Agent", type="primary"):
            all_errors = []
            
            progress_bar = st.progress(0)
            status_container = st.status("Consiliul analizează...", expanded=True)
            
            for i, cap in enumerate(chapters):
                status_container.write(f"Analizez: **{cap['titlu']}**")
                errs = council.run_ensemble_analysis(cap['text'], cap['titlu'])
                all_errors.extend(errs)
                progress_bar.progress((i + 1) / len(chapters))
            
            # Nu mai folosim filtrul Python, folosim outputul direct din Judge
            st.session_state.app_state["final_report"] = all_errors
            status_container.update(label="Analiză completă!", state="complete", expanded=False)
            st.rerun()

    # 4. REPORT
    if st.session_state.app_state["final_report"]:
        report = st.session_state.app_state["final_report"]
        
        st.success(f"Am identificat {len(report)} probleme relevante.")
        
        df_rep = pd.DataFrame(report)
        if not df_rep.empty:
            if "validat" not in df_rep.columns: df_rep["validat"] = False
            
            val_df = st.data_editor(
                df_rep,
                column_config={
                    "validat": st.column_config.CheckboxColumn("Valid?", default=False),
                    "tip": "Tip Eroare",
                    "text": "Text Original",
                    "explicatie": "Analiză AI",
                    "capitol": "Capitol"
                },
                use_container_width=True,
                height=500
            )
            
            valid_rows = val_df[val_df["validat"] == True]
            st.divider()
            
            c1, c2 = st.columns([1, 4])
            with c1:
                if not valid_rows.empty:
                    csv = valid_rows.to_csv(index=False).encode('utf-8')
                    st.download_button("Descarcă CSV", csv, "raport_audit.csv", "text/csv")
            with c2:
                pass
        else:
            st.info("Documentul pare curat.")
            
        if st.button("Analizează alt document"):
             st.session_state.app_state = {"stage": "upload", "pdf_bytes": None, "chapters": [], "structure_data": [], "final_report": [], "debug_img": None}
             st.rerun()