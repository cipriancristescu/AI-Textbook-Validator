import streamlit as st
import fitz  # PyMuPDF
import os
import json
import time
import base64
import requests
import pandas as pd
from dotenv import load_dotenv
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Gemini SDK (as in your working version)
from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================
load_dotenv()
st.set_page_config(page_title="AI Manual Auditor", layout="wide", initial_sidebar_state="expanded")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TOGETHER_API_KEY or not TAVILY_API_KEY or not GEMINI_API_KEY:
    st.error("Lipsesc chei în .env: TOGETHER_API_KEY, TAVILY_API_KEY și/sau GEMINI_API_KEY.")
    st.stop()

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# --- Models (Together) ---
# claims: Qwen2.5 72B (bun la extracție structurată)
# bias: Llama 3.3 70B (bun la instrucțiuni / critică)
# factcheck: DeepSeek-R1 (bun la verificare / raționament)
# international: DeepSeek-R1 (bun la comparație / query building)
# judge: Llama 3.3 70B (bun la dedup/curățare)
TEXT_MODELS = {
    "claims": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "bias": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "factcheck": "deepseek-ai/DeepSeek-R1",
    "international": "deepseek-ai/DeepSeek-R1",
    "judge": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

# TOC Vision candidates (try in order)
VISION_MODEL_CANDIDATES = [
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# ============================================================
# SESSION STATE
# ============================================================
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "stage": "upload",
        "pdf_bytes": None,
        "doc_len": 0,
        "structure_data": [],
        "chapters": [],
        "final_report": [],
        "debug_imgs": [],      # list of (page, img_bytes)
        "gemini_cache": {},    # cache for grammar ranges
    }

# ============================================================
# HELPERS: text + safe JSON extraction
# ============================================================
def normalize_text_minimal(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFC", s)

def _extract_balanced_block(s: str, open_ch: str, close_ch: str) -> Optional[str]:
    start = s.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def safe_extract_json(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    obj = _extract_balanced_block(t, "{", "}")
    if obj:
        try:
            return json.loads(obj)
        except Exception:
            pass

    arr = _extract_balanced_block(t, "[", "]")
    if arr:
        try:
            return json.loads(arr)
        except Exception:
            pass

    return None

def looks_like_encoding_garbage(s: str) -> bool:
    """
    Heuristic: dacă textul are multe caractere ciudate (þ, �, etc) sau secvențe tipice de encoding stricat.
    """
    if not s:
        return True
    bad = ["þ", "�", "\x00", "t]ara", "s,coal,", "m\\re"]
    lower = s.lower()
    if any(b in lower for b in bad):
        return True
    # prea multe caractere non-alfanumerice
    weird = sum(1 for ch in s if ord(ch) < 9 or (0x7f <= ord(ch) <= 0x9f))
    return weird > 0

# ============================================================
# Together API wrappers
# ============================================================
def together_chat_json(
    model: str,
    system_prompt: str,
    user_text: str,
    timeout: int = 90,
    max_retries: int = 3
) -> Any:
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
        "top_p": 1,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{TOGETHER_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=timeout)
            data = r.json()
            if r.status_code != 200:
                last_err = data.get("error") or data
                time.sleep(0.8 * (attempt + 1))
                continue

            content = data["choices"][0]["message"]["content"]
            parsed = safe_extract_json(content)
            if parsed is not None:
                return parsed

            # fallback: retry without response_format
            payload2 = dict(payload)
            payload2.pop("response_format", None)
            r2 = requests.post(f"{TOGETHER_BASE_URL}/chat/completions", headers=headers, json=payload2, timeout=timeout)
            data2 = r2.json()
            if r2.status_code != 200:
                last_err = data2.get("error") or data2
                time.sleep(0.8 * (attempt + 1))
                continue

            content2 = data2["choices"][0]["message"]["content"]
            parsed2 = safe_extract_json(content2)
            if parsed2 is not None:
                return parsed2

            last_err = {"message": "Could not parse JSON", "raw": (content2 or "")[:900]}
            time.sleep(0.8 * (attempt + 1))

        except Exception as e:
            last_err = str(e)
            time.sleep(0.8 * (attempt + 1))

    return {"_error": last_err}

def together_vision_json(model: str, prompt_text: str, b64_png: str, timeout: int = 90) -> Any:
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}}],
        }],
        "temperature": 0,
        "top_p": 1,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(f"{TOGETHER_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        return {"_error": {"message": "Non-JSON response from Together (vision)"}}
    if r.status_code != 200:
        return {"_error": data.get("error") or data}

    content = data["choices"][0]["message"]["content"]
    parsed = safe_extract_json(content)
    if parsed is not None:
        return parsed
    return {"_error": {"message": "Vision returned non-JSON", "raw": content[:900]}}

# ============================================================
# Tavily
# ============================================================
def tavily_search(query: str, max_results: int = 3, search_depth: str = "advanced") -> Optional[str]:
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
            },
            timeout=45,
        )
        data = r.json()
        results = data.get("results", [])
        return "\n".join([f"- {x.get('content','')} (Sursă: {x.get('url','')})" for x in results])
    except Exception:
        return None

# ============================================================
# PDF tools
# ============================================================
def render_page_png(pdf_bytes: bytes, page_1based: int, dpi: int = 280) -> Optional[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    idx = page_1based - 1
    if idx < 0 or idx >= len(doc):
        return None
    pix = doc[idx].get_pixmap(dpi=dpi)
    return pix.tobytes("png")

def extract_chapter_text(pdf_bytes: bytes, structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)

    structure = sorted(structure, key=lambda x: int(x.get("start", 0) or 0))

    for i in range(len(structure)):
        if int(structure[i].get("end", 0) or 0) == 0:
            if i < len(structure) - 1:
                structure[i]["end"] = int(structure[i + 1]["start"]) - 1
            else:
                structure[i]["end"] = total

    chapters = []
    for it in structure:
        start = max(1, int(it.get("start", 1)))
        end = min(total, int(it.get("end", total)))
        text = ""
        for p in range(start - 1, end):
            text += f"\n\n[[PAGINA {p+1}]]\n{normalize_text_minimal(doc[p].get_text())}"

        chapters.append({
            "titlu": normalize_text_minimal(it.get("titlu", "Capitol")),
            "interval": f"{start}-{end}",
            "start": start,
            "end": end,
            "text": text
        })
    return chapters

# ============================================================
# TOC vision: multi-page + merge/dedup/sort
# ============================================================
def _norm_title_key(t: str) -> str:
    t = normalize_text_minimal(str(t or "")).strip()
    t = " ".join(t.split())
    t = t.rstrip(".")
    return t.lower()

def merge_dedup_sort_toc_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        title = str(it.get("titlu", "")).strip()
        try:
            start = int(it.get("start", 0))
        except Exception:
            continue
        if not title or start <= 0:
            continue
        key = _norm_title_key(title)
        if key not in best:
            best[key] = {"titlu": normalize_text_minimal(title), "start": start}
        else:
            if start < best[key]["start"]:
                best[key]["start"] = start
            if len(title) > len(best[key]["titlu"]):
                best[key]["titlu"] = normalize_text_minimal(title)
    merged = list(best.values())
    merged.sort(key=lambda x: int(x.get("start", 10**9)))
    return merged

def extract_toc_multi(pdf_bytes: bytes, toc_start: int, toc_end: int) -> Tuple[List[Dict[str, Any]], List[Tuple[int, bytes]], List[str]]:
    all_items = []
    debug_imgs = []
    errors = []

    prompt = """
Privește această pagină de CUPRINS (Table of Contents).
Extrage capitolele + NUMĂR DE PAGINĂ (pagina de START).
Nu inventa. Ignoră titlul paginii.

Returnează JSON STRICT:
{"items":[{"titlu":"Titlu Capitol","start":5}]}
"""

    for p in range(toc_start, toc_end + 1):
        img = render_page_png(pdf_bytes, p, dpi=280)
        if not img:
            errors.append(f"Pag {p}: nu pot randa pagina")
            continue

        debug_imgs.append((p, img))
        b64 = base64.b64encode(img).decode("utf-8")

        out = None
        last_err = None
        for vm in VISION_MODEL_CANDIDATES:
            res = together_vision_json(vm, prompt, b64, timeout=90)
            if isinstance(res, dict) and res.get("_error"):
                last_err = res["_error"]
                continue
            out = res
            break

        if out is None:
            errors.append(f"Pag {p}: {last_err}")
            continue

        if isinstance(out, dict) and isinstance(out.get("items"), list):
            all_items.extend(out["items"])
        elif isinstance(out, dict):
            for v in out.values():
                if isinstance(v, list):
                    all_items.extend(v)
                    break

    merged = merge_dedup_sort_toc_items(all_items)
    return merged, debug_imgs, errors

# ============================================================
# FACTUAL COUNCIL (Bias + Local facts + International) + Judge
# ============================================================
class FactualCouncil:
    def __init__(
        self,
        max_claims_per_chapter: int,
        tavily_max_results: int,
        tavily_depth: str
    ):
        self.max_claims = max_claims_per_chapter
        self.max_results = tavily_max_results
        self.depth = tavily_depth

    # --- Agent 1: Bias/Semantic ---
    def agent_bias(self, text: str) -> List[Dict[str, Any]]:
        system = """
Ești un critic istoric și literar.
Analizează textul pentru Propagandă, Bias și Limbaj Absolutist.
Identifică omisiuni majore de context sau interpretări naționaliste excesive.

NU da lecții și NU face recomandări de stil generale.
Raportează doar pasaje problematice (citat scurt) + motiv.

Returnează JSON STRICT:
{"erori":[{"tip":"Bias/Nuanta","text":"...","sugestie":"...","explicatie":"..."}]}
"""
        res = together_chat_json(TEXT_MODELS["bias"], system, text[:15000], timeout=90)
        if isinstance(res, dict) and res.get("_error"):
            return []
        erori = res.get("erori", []) if isinstance(res, dict) else []
        out = []
        for e in erori:
            if not isinstance(e, dict):
                continue
            t = str(e.get("text", "")).strip()
            if not t or looks_like_encoding_garbage(t):
                continue
            out.append({
                "tip": "Bias/Nuanta",
                "text": t,
                "sugestie": str(e.get("sugestie", "")).strip(),
                "explicatie": str(e.get("explicatie", "")).strip()
            })
        return out

    # --- Agent 2: Factual Local (claims -> Tavily -> verify) ---
    def extract_claims(self, text: str) -> List[str]:
        system = """
Ești un auditor factual STRICT.
Extrage 20-30 afirmații factuale verificabile din text (ani, date, nume, tratate, cifre, evenimente).
NU oferi sugestii didactice.
Fiecare afirmație trebuie să fie scurtă și concretă.

Returnează JSON STRICT:
{"claims":["...","..."]}
"""
        res = together_chat_json(TEXT_MODELS["claims"], system, text[:14000], timeout=90)
        if isinstance(res, dict) and res.get("_error"):
            return []
        claims = res.get("claims", []) if isinstance(res, dict) else []
        out = []
        for c in claims:
            c = str(c).strip()
            if c and len(c) >= 5:
                out.append(c)
        # flux simplu: luăm primele N (nu scoring)
        return out[: self.max_claims]

    def build_evidence_for_claims(self, claims: List[str]) -> str:
        evidence = ""
        for c in claims:
            q = f"Verifică afirmația (istorie România / internațional): {c}"
            ev = tavily_search(q, max_results=self.max_results, search_depth=self.depth)
            if ev:
                evidence += f"\nCLAIM: {c}\nEVIDENCE:\n{ev}\n"
        return evidence

    def agent_factual_local(self, text: str) -> List[Dict[str, Any]]:
        claims = self.extract_claims(text)
        evidence = self.build_evidence_for_claims(claims)

        system = """
Ești editor și fact-checker STRICT.
NU oferi sugestii didactice.

Pentru fiecare CLAIM din DOVEZI:
- Verdict = GRESIT dacă dovezile contrazic clar afirmația.
- Verdict = NECONFIRMAT dacă dovezile nu confirmă clar sau sunt insuficiente.
Dacă nu ai dovezi relevante, pune NECONFIRMAT (nu inventa).

Returnează JSON STRICT:
{"erori":[
  {"tip":"Factual","verdict":"GRESIT|NECONFIRMAT","text":"<claim>","sugestie":"","explicatie":"..."}
]}
"""
        user = f"""
TEXT (context):
{text[:15000]}

DOVEZI WEB:
{evidence}

Atenție: Doar factual, fără "ar fi mai bine...".
"""
        res = together_chat_json(TEXT_MODELS["factcheck"], system, user, timeout=140)
        if isinstance(res, dict) and res.get("_error"):
            return []

        erori = res.get("erori", []) if isinstance(res, dict) else []
        out = []
        for e in erori:
            if not isinstance(e, dict):
                continue
            verdict = str(e.get("verdict", "")).upper().strip()
            if verdict not in ("GRESIT", "NECONFIRMAT"):
                continue
            txt = str(e.get("text", "")).strip()
            if not txt or looks_like_encoding_garbage(txt):
                continue
            out.append({
                "tip": f"Factual ({verdict})",
                "text": txt,
                "sugestie": str(e.get("sugestie", "")).strip(),
                "explicatie": str(e.get("explicatie", "")).strip()
            })
        return out

    # --- Agent 3: International auditor (themes -> EN queries -> Tavily -> compare) ---
    def agent_international(self, text: str) -> List[Dict[str, Any]]:
        system_q = """
Extrage 2-4 teme / afirmații majore din text și transformă-le în interogări de căutare în ENGLEZĂ.
Interogările trebuie să fie scurte și precise (nume+eveniment+an dacă există).
NU inventa teme care nu apar în text.

Returnează JSON STRICT:
{"queries":["...","..."]}
"""
        q_res = together_chat_json(TEXT_MODELS["international"], system_q, text[:12000], timeout=90)
        if isinstance(q_res, dict) and q_res.get("_error"):
            return []
        queries = q_res.get("queries", []) if isinstance(q_res, dict) else []
        queries = [str(q).strip() for q in queries if str(q).strip()][:4]

        evidence = ""
        for q in queries:
            ev = tavily_search(q, max_results=self.max_results, search_depth=self.depth)
            if ev:
                evidence += f"\nQUERY: {q}\nINTL_EVIDENCE:\n{ev}\n"

        system_cmp = """
Ești un auditor internațional.
Compară narativul textului (română) cu dovezile internaționale (ENG).
Semnalează DOAR dacă există:
- contradicții clare,
- omisiuni majore care schimbă sensul,
- formulări tendențioase față de consensul general.

NU da lecții, NU generaliza.

Returnează JSON STRICT:
{"erori":[{"tip":"Perspectiva Int.","text":"...","sugestie":"...","explicatie":"..."}]}
"""
        user_cmp = f"""
TEXT (context):
{text[:15000]}

DOVEZI INTERNAȚIONALE:
{evidence}
"""
        res = together_chat_json(TEXT_MODELS["international"], system_cmp, user_cmp, timeout=140)
        if isinstance(res, dict) and res.get("_error"):
            return []
        erori = res.get("erori", []) if isinstance(res, dict) else []
        out = []
        for e in erori:
            if not isinstance(e, dict):
                continue
            t = str(e.get("text", "")).strip()
            if not t or looks_like_encoding_garbage(t):
                continue
            out.append({
                "tip": "Perspectiva Int.",
                "text": t,
                "sugestie": str(e.get("sugestie", "")).strip(),
                "explicatie": str(e.get("explicatie", "")).strip()
            })
        return out

    # --- Judge: dedup + remove noise ---
    def judge(self, all_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        system = """
Ești Editorul Șef (Judge).
Curăță lista: elimină dublurile și șterge zgomotul.

INSTRUCȚIUNI:
1) ȘTERGE intrarile care par eroari de encoding/font.
2) ȘTERGE intrarile care sunt sugestie didactică, dar fără problemă reală.
3) Păstrează doar problemele reale: Bias/Nuanță, Perspectivă Internationala, Factual (GRESIT/NECONFIRMAT).
4) Unește dublurile (texte similare).

Returnează JSON STRICT:
{"final":[{"tip":"...","text":"...","sugestie":"...","explicatie":"..."}]}
"""
        res = together_chat_json(TEXT_MODELS["judge"], system, json.dumps(all_raw, ensure_ascii=False), timeout=120)
        if isinstance(res, dict) and res.get("_error"):
            return all_raw

        final = res.get("final", []) if isinstance(res, dict) else []
        out = []
        seen = set()
        for e in final:
            if not isinstance(e, dict):
                continue
            tip = str(e.get("tip", "")).strip()
            txt = str(e.get("text", "")).strip()
            if not tip or not txt:
                continue
            if looks_like_encoding_garbage(txt):
                continue
            key = (tip.lower(), txt.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "tip": tip,
                "text": txt,
                "sugestie": str(e.get("sugestie","")).strip(),
                "explicatie": str(e.get("explicatie","")).strip(),
            })
        return out

    def run_on_chapter(self, chapter_text: str, chapter_title: str) -> List[Dict[str, Any]]:
        raw = []
        # nu folosim ThreadPoolExecutor ca să evităm warnings/instabilitate în Streamlit
        raw.extend(self.agent_bias(chapter_text))
        raw.extend(self.agent_factual_local(chapter_text))
        raw.extend(self.agent_international(chapter_text))

        cleaned = self.judge(raw)
        for e in cleaned:
            e["capitol"] = chapter_title
        return cleaned

# ============================================================
# GRAMMAR via Gemini on PDF (KEEP LOGIC; REMOVE PAGE FIELD IN FINAL REPORT)
# ============================================================
def gemini_grammar_range(pdf_bytes: bytes, start_page: int, end_page: int, model: str, max_errors: int) -> Dict[str, Any]:
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
Ești corector de LIMBA ROMÂNĂ pentru manuale școlare.
Analizează DOAR paginile {start_page}–{end_page} din document.

Raportează DOAR greșeli gramaticale CERTE (dezacorduri evidente, forme greșite, acorduri, regimuri, timpuri).
NU raporta:
- sugestii didactice (“ar fi mai bine…”),
- stil / reformulări alternative acceptabile,
- punctuație fină.

Pentru fiecare greșeală:
- citează EXACT propoziția (1–2 propoziții),
- dă corectarea,
- explică scurt regula,
- pagina: dacă e sigură, folosește numărul, altfel null.
Maxim {max_errors} erori.

Returnează STRICT JSON:
{{
  "erori":[
    {{"pagina": null, "text":"...", "corect":"...", "explicatie":"..."}}
  ]
}}
"""

    last_err = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    prompt,
                ],
            )
            txt = resp.text or ""
            parsed = safe_extract_json(txt)
            if isinstance(parsed, dict) and isinstance(parsed.get("erori"), list):
                return parsed

            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed2 = safe_extract_json(txt[start:end+1])
                if isinstance(parsed2, dict) and isinstance(parsed2.get("erori"), list):
                    return parsed2

            return {"erori": [], "_raw": txt[:1200]}
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))

    return {"erori": [], "_error": last_err}

def grammar_run_all(pdf_bytes: bytes, doc_len: int, model: str, pages_per_chunk: int, max_errors_per_chunk: int) -> List[Dict[str, Any]]:
    out = []
    cache = st.session_state.app_state["gemini_cache"]

    for start in range(1, doc_len + 1, pages_per_chunk):
        end = min(doc_len, start + pages_per_chunk - 1)
        key = (start, end, model, max_errors_per_chunk)

        if key in cache:
            parsed = cache[key]
        else:
            parsed = gemini_grammar_range(pdf_bytes, start, end, model, max_errors_per_chunk)
            cache[key] = parsed

        for e in parsed.get("erori", []):
            if not isinstance(e, dict):
                continue
            out.append({
                "tip": "Gramatica",
                "verdict": "GRESIT",
                "capitol": f"Pagini {start}-{end}",
                # IMPORTANT: nu mai păstrăm pagina în raportul final
                "text": str(e.get("text","")).strip(),
                "sugestie": str(e.get("corect","")).strip(),
                "explicatie": str(e.get("explicatie","")).strip(),
            })
    return out

# ============================================================
# UI
# ============================================================
st.title("AI Manual Auditor (Together factual + Gemini gramatică)")

with st.sidebar:
    st.header("Ce rulez?")
    analysis_mode = st.radio(
        "Selectează analiza",
        ["Ambele", "Factual", "Gramatica"],
        index=0
    )

    st.divider()
    st.header("Factual (Tavily)")
    tavily_depth = st.selectbox("search_depth", ["advanced", "basic"], index=0)
    tavily_max_results = st.slider("max_results / query", 1, 6, 3)
    max_claims_per_chapter = st.slider("max claims / capitol (factual)", 3, 15, 8)

    st.divider()
    st.header("Gemini (Gramatică)")
    gem_model = st.text_input("Model Gemini", value=DEFAULT_GEMINI_MODEL)
    pages_per_chunk = st.slider("Pagini per chunk", 5, 40, 20)
    max_err_chunk = st.slider("Max erori / chunk", 5, 60, 30)

# 1) UPLOAD + TOC multi-page
if st.session_state.app_state["stage"] == "upload":
    with st.container(border=True):
        st.subheader("1) Încărcare PDF + Cuprins (interval pagini)")
        uploaded = st.file_uploader("Selectează fișier PDF", type="pdf")

        if uploaded:
            pdf_bytes = uploaded.getvalue()
            st.session_state.app_state["pdf_bytes"] = pdf_bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            doc_len = len(doc)
            st.session_state.app_state["doc_len"] = doc_len
            st.info(f"Document încărcat. Total pagini: {doc_len}")

            c1, c2 = st.columns(2)
            with c1:
                toc_start = st.number_input("Cuprins: pagina START", min_value=1, max_value=doc_len, value=min(3, doc_len))
            with c2:
                toc_end = st.number_input("Cuprins: pagina END", min_value=1, max_value=doc_len, value=min(3, doc_len))

            if st.button("📸 Scanează cuprinsul (Vision)", type="primary"):
                a, b = min(int(toc_start), int(toc_end)), max(int(toc_start), int(toc_end))
                with st.spinner(f"Scanez cuprinsul pe paginile {a}-{b}..."):
                    struct, dbg, errs = extract_toc_multi(pdf_bytes, a, b)
                    st.session_state.app_state["structure_data"] = struct
                    st.session_state.app_state["debug_imgs"] = dbg
                if errs:
                    st.warning("Unele pagini au dat erori la scanarea cuprinsului.")
                    with st.expander("Detalii erori"):
                        for e in errs:
                            st.write(e)

            if st.session_state.app_state["structure_data"]:
                st.success(f"Capitole detectate: {len(st.session_state.app_state['structure_data'])}")
                with st.expander("Vezi paginile de cuprins scanate"):
                    for p, img in st.session_state.app_state["debug_imgs"]:
                        st.markdown(f"**Pagina {p}**")
                        st.image(img, width="stretch")

                if st.button("➡️ Mergi la Validare Structură"):
                    st.session_state.app_state["stage"] = "approve"
                    st.rerun()

            st.divider()
            if st.button("Omite cuprins (segmentare automată 20 pag)"):
                st.session_state.app_state["structure_data"] = [{"titlu": f"Segment {i//20 + 1}", "start": i + 1} for i in range(0, doc_len, 20)]
                st.session_state.app_state["stage"] = "approve"
                st.rerun()

# 2) APPROVE STRUCTURE
elif st.session_state.app_state["stage"] == "approve":
    with st.container(border=True):
        st.subheader("2) Validare Structură")
        df = pd.DataFrame(st.session_state.app_state["structure_data"])
        if "titlu" not in df.columns: df["titlu"] = "Capitol"
        if "start" not in df.columns: df["start"] = 1
        if "end" not in df.columns: df["end"] = 0

        edited_df = st.data_editor(
            df,
            column_config={
                "titlu": "Titlu Capitol",
                "start": st.column_config.NumberColumn("Start", format="%d"),
                "end": st.column_config.NumberColumn("Final", format="%d"),
            },
            width="stretch",
            num_rows="dynamic"
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirmă și Analizează", type="primary"):
                structure = edited_df.to_dict("records")
                st.session_state.app_state["chapters"] = extract_chapter_text(st.session_state.app_state["pdf_bytes"], structure)
                st.session_state.app_state["final_report"] = []
                st.session_state.app_state["gemini_cache"] = {}
                st.session_state.app_state["stage"] = "analyze"
                st.rerun()
        with c2:
            if st.button("Resetează"):
                st.session_state.app_state = {
                    "stage": "upload",
                    "pdf_bytes": None,
                    "doc_len": 0,
                    "structure_data": [],
                    "chapters": [],
                    "final_report": [],
                    "debug_imgs": [],
                    "gemini_cache": {},
                }
                st.rerun()

# 3) ANALYZE
elif st.session_state.app_state["stage"] == "analyze":
    chapters = st.session_state.app_state["chapters"]
    pdf_bytes = st.session_state.app_state["pdf_bytes"]
    doc_len = st.session_state.app_state["doc_len"]

    st.subheader(f"3) Analiză ({len(chapters)} capitole)")

    if not st.session_state.app_state["final_report"]:
        if st.button("Start Audit", type="primary"):
            rows: List[Dict[str, Any]] = []
            status = st.status("Rulez auditul...", expanded=True)

            # FACTUAL (Bias + Local + Intl + Judge) per chapter
            if analysis_mode in ("Ambele", "Factual"):
                council = FactualCouncil(
                    max_claims_per_chapter=max_claims_per_chapter,
                    tavily_max_results=tavily_max_results,
                    tavily_depth=tavily_depth,
                )
                progress = st.progress(0.0)
                for i, cap in enumerate(chapters, start=1):
                    status.write(f"Factual: **{cap['titlu']}** ({cap['interval']})")
                    errs = council.run_on_chapter(cap["text"], cap["titlu"])
                    for e in errs:
                        rows.append({
                            "validat": False,
                            "tip": e.get("tip",""),
                            "capitol": e.get("capitol",""),
                            "text": e.get("text",""),
                            "sugestie": e.get("sugestie",""),
                            "explicatie": e.get("explicatie",""),
                        })
                    progress.progress(i / max(1, len(chapters)))

            # GRAMMAR (Gemini), chunked across entire doc
            if analysis_mode in ("Ambele", "Gramatica"):
                status.write("Gramatică (Gemini pe PDF, chunked)...")
                g_rows = grammar_run_all(pdf_bytes, doc_len, gem_model, pages_per_chunk, max_err_chunk)
                for r in g_rows:
                    rows.append({
                        "validat": False,
                        "tip": "Gramatica",
                        "capitol": r.get("capitol",""),
                        "text": r.get("text",""),
                        "sugestie": r.get("sugestie",""),
                        "explicatie": r.get("explicatie",""),
                    })

            st.session_state.app_state["final_report"] = rows
            status.update(label="Analiză completă!", state="complete", expanded=False)
            st.rerun()

    # 4) REPORT
    report = st.session_state.app_state["final_report"]
    if report:
        st.success(f"Total intrări: {len(report)}")

        df_rep = pd.DataFrame(report)
        for col in ["validat", "tip", "capitol", "text", "sugestie", "explicatie"]:
            if col not in df_rep.columns:
                df_rep[col] = ""

        df_rep = df_rep[["validat", "tip", "capitol", "text", "sugestie", "explicatie"]]

        val_df = st.data_editor(
            df_rep,
            column_config={
                "validat": st.column_config.CheckboxColumn("Valid?", default=False),
                "tip": "Tip",
                "capitol": "Capitol/Chunk",
                "text": "Text / Claim",
                "sugestie": "Sugestie/Corectare",
                "explicatie": "Explicație",
            },
            width="stretch",
            height=520
        )

        valid_rows = val_df[val_df["validat"] == True]
        st.divider()

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if not valid_rows.empty:
                csv = valid_rows.to_csv(index=False).encode("utf-8")
                st.download_button("Descarcă CSV (validate)", csv, "raport_validat.csv", "text/csv")
        with c2:
            if st.button("Repornește analiza"):
                st.session_state.app_state["final_report"] = []
                st.session_state.app_state["gemini_cache"] = {}
                st.rerun()
        with c3:
            if st.button("Analizează alt document"):
                st.session_state.app_state = {
                    "stage": "upload",
                    "pdf_bytes": None,
                    "doc_len": 0,
                    "structure_data": [],
                    "chapters": [],
                    "final_report": [],
                    "debug_imgs": [],
                    "gemini_cache": {},
                }
                st.rerun()
