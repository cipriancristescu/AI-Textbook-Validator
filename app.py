#py -3.14 -m streamlit run app.py

import re
import streamlit as st
import fitz  # PyMuPDF
import os
import json
import time
import requests
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# Gemini SDK
from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================
load_dotenv()
st.set_page_config(page_title="AI Manual Auditor", layout="wide", initial_sidebar_state="expanded")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TOGETHER_API_KEY or not GEMINI_API_KEY:
    st.error("Lipsesc chei în .env: TOGETHER_API_KEY și/sau GEMINI_API_KEY.")
    st.stop()

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Together AI — modele text serverless accesibile cu cheia curentă
TOGETHER_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Gemini — folosit pentru TOC, fact-checking și gramatică
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_SEGMENT_SIZE = 10

# Delay minim între apeluri Gemini (free tier: 15 req/min)
GEMINI_MIN_DELAY_S = 4.5

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
        "fact_report": [],
        "grammar_report": [],
        "gemini_cache": {},
        "debug_log": [],
        "fact_debug_log": [],
        "grammar_debug_log": [],
        "audit_ran": False,
        "_last_gemini_call": 0.0,
    }

for _k, _v in [
    ("fact_report", []),
    ("grammar_report", []),
    ("fact_debug_log", []),
    ("grammar_debug_log", []),
    ("debug_log", []),
    ("audit_ran", False),
    ("_last_gemini_call", 0.0),
]:
    if _k not in st.session_state.app_state:
        st.session_state.app_state[_k] = _v


# ============================================================
# HELPERS
# ============================================================
def normalize_text_minimal(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFC", s)


def strip_thinking_blocks(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def safe_extract_json(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    text = strip_thinking_blocks(text)
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        block = _extract_balanced_block(t, open_ch, close_ch)
        if block:
            try:
                return json.loads(block)
            except Exception:
                pass
    return None


def _coerce_items_from_json(data: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get(key), list):
        return [x for x in data[key] if isinstance(x, dict)]
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _dedup_fact_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        page = str(item.get("pagina", ""))
        text = " ".join(str(item.get("text", "")).lower().split())[:120]
        sugestie = " ".join(str(item.get("sugestie", "")).lower().split())[:120]
        key = (page, text, sugestie)
        if text and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ============================================================
# GEMINI RATE LIMITER
# ============================================================
def _gemini_throttle() -> None:
    """Ensures at least GEMINI_MIN_DELAY_S between consecutive Gemini calls."""
    last = st.session_state.app_state.get("_last_gemini_call", 0.0)
    elapsed = time.time() - last
    if elapsed < GEMINI_MIN_DELAY_S:
        time.sleep(GEMINI_MIN_DELAY_S - elapsed)
    st.session_state.app_state["_last_gemini_call"] = time.time()


def _gemini_generate(model: str, contents: list, max_retries: int = 4) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls Gemini with throttling and retry on 429.
    Returns (text, error_message).
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        _gemini_throttle()
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            return resp.text or "", None
        except Exception as e:
            last_err = str(e)
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                m = re.search(r"retryDelay.*?'(\d+)s'", err_str)
                wait = int(m.group(1)) + 3 if m else 65
                time.sleep(wait)
            else:
                time.sleep(2.0 * (attempt + 1))
    return None, last_err


# ============================================================
# TOGETHER AI — TEXT ONLY (JUDGE)
# ============================================================
def together_chat_json(
    model: str,
    system_prompt: str,
    user_text: str,
    timeout: int = 90,
    max_retries: int = 3,
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
    last_raw = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{TOGETHER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            data = r.json()
            if r.status_code != 200:
                last_err = data.get("error") or data
                time.sleep(0.8 * (attempt + 1))
                continue
            content = data["choices"][0]["message"]["content"]
            last_raw = content
            parsed = safe_extract_json(content)
            if parsed is not None:
                return parsed
            # Retry without response_format
            payload2 = {k: v for k, v in payload.items() if k != "response_format"}
            r2 = requests.post(
                f"{TOGETHER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload2,
                timeout=timeout,
            )
            data2 = r2.json()
            if r2.status_code != 200:
                last_err = data2.get("error") or data2
                time.sleep(0.8 * (attempt + 1))
                continue
            content2 = data2["choices"][0]["message"]["content"]
            last_raw = content2
            parsed2 = safe_extract_json(content2)
            if parsed2 is not None:
                return parsed2
            last_err = {"message": "Nu s-a putut parsa JSON", "raw_preview": (content2 or "")[:800]}
            time.sleep(0.8 * (attempt + 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(0.8 * (attempt + 1))
    return {"_error": last_err, "_raw": (last_raw or "")[:1200]}


# ============================================================
# PDF UTILS
# ============================================================
def chapter_title_for_page(chapters: List[Dict[str, Any]], page_1based: int) -> str:
    for cap in chapters:
        try:
            start = int(cap.get("start", 0))
            end = int(cap.get("end", 0))
        except Exception:
            continue
        if start <= page_1based <= end:
            return str(cap.get("titlu", "")).strip()
    return ""


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
            text += f"\n\n[[PAGINA {p + 1}]]\n{normalize_text_minimal(doc[p].get_text())}"
        chapters.append({
            "titlu": normalize_text_minimal(it.get("titlu", "Capitol")),
            "interval": f"{start}-{end}",
            "start": start,
            "end": end,
            "text": text,
        })
    return chapters


# ============================================================
# TOC EXTRACTION VIA GEMINI
# ============================================================
def _norm_title_key(t: str) -> str:
    t = normalize_text_minimal(str(t or "")).strip()
    t = " ".join(t.split()).rstrip(".")
    return t.lower()


def merge_dedup_sort_toc_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Any] = {}
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


def gemini_extract_toc(
    pdf_bytes: bytes,
    toc_start: int,
    toc_end: int,
    model: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Sends PDF to Gemini and extracts the table of contents from pages toc_start–toc_end.
    Returns (items, error_message).
    """
    prompt = f"""Look at pages {toc_start} to {toc_end} of this PDF.
These pages contain the Table of Contents (Cuprins) of a Romanian textbook.

Extract every chapter/section title together with its starting page number.
Do NOT invent entries. Only extract what is explicitly listed in the table of contents.
Ignore page headers, footers, and decorative elements.

Return STRICT JSON:
{{"items": [{{"titlu": "Chapter Title", "start": 5}}]}}"""

    txt, err = _gemini_generate(
        model,
        [types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
    )
    if err:
        return [], err
    parsed = safe_extract_json(txt or "")
    if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        items = [x for x in parsed["items"] if isinstance(x, dict)]
        return merge_dedup_sort_toc_items(items), None
    return [], f"Could not parse TOC JSON. Raw: {(txt or '')[:400]}"


# ============================================================
# EXCEL EXPORT
# ============================================================
def _safe_cell_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return str(v)


def dataframe_to_pretty_excel_bytes(df: pd.DataFrame, sheet_name: str = "Raport") -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name[:31]
    ws.freeze_panes = "A2"

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    columns = list(df.columns)
    ws.append(columns)
    for c_idx, col_name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=c_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border

    for _, row in df.iterrows():
        ws.append([row.get(col, "") for col in columns])

    preferred_widths = {"validat": 10, "tip": 24, "capitol": 28, "text": 70, "sugestie": 70, "explicatie": 70}
    for c_idx, col_name in enumerate(columns, start=1):
        letter = get_column_letter(c_idx)
        max_line_len = len(str(col_name))
        for r_idx in range(2, ws.max_row + 1):
            cell = ws.cell(row=r_idx, column=c_idx)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = border
            txt = _safe_cell_text(cell.value)
            lines = txt.splitlines() or [txt]
            longest = max((len(line) for line in lines), default=0)
            max_line_len = max(max_line_len, min(longest, 120))
        width = preferred_widths.get(col_name.lower()) or min(max(max_line_len + 2, 12), 80)
        ws.column_dimensions[letter].width = width

    for r_idx in range(2, ws.max_row + 1):
        max_lines = 1
        for c_idx in range(1, ws.max_column + 1):
            col_letter = get_column_letter(c_idx)
            width = ws.column_dimensions[col_letter].width or 15
            txt = _safe_cell_text(ws.cell(row=r_idx, column=c_idx).value)
            logical_lines = txt.splitlines() or [txt]
            usable = max(int(width) - 2, 8)
            estimated = sum(max(1, (len(ln) // usable) + (1 if len(ln) % usable else 0)) for ln in logical_lines)
            max_lines = max(max_lines, estimated)
        ws.row_dimensions[r_idx].height = min(max(20 * max_lines, 20), 180)
    ws.row_dimensions[1].height = 28

    out = BytesIO()
    wb.save(out)
    out.seek(0)
    return out.getvalue()


# ============================================================
# FACT-CHECKING: GEMINI DETECTORS + TOGETHER JUDGE
# ============================================================

_DETECTOR_A_SYSTEM = """Ești Detector A — auditor tehnic specializat pe cod și sintaxă.

Analizează paginile indicate din manualul PDF românesc de informatică.

Caută EXCLUSIV erori de cod și sintaxă:
- operatori C/C++ greșiți: cout >> în loc de cout <<, cin << în loc de cin >>;
- sintaxă C/C++ imposibilă (Typedef, int:var, lipsă punct și virgulă obligatoriu);
- bucle for/while cu variabila de inițializare diferită de cea din condiție sau increment;
- literal în condiție în loc de variabilă: for(i=1; 1<n; i++);
- pseudocod cu cifra 0 înlocuită cu litera o în expresii numerice;
- cod care produce alt rezultat decât cel afirmat explicit în text;
- void main() sau #include <iostream.h> prezentate ca standard modern fără nicio mențiune că sunt vechi.

Nu raporta: gramatică, diacritice, layout, indentare, stil, lipsă comentarii,
using namespace std, bits/stdc++.h, nume scurte de variabile, variabile globale,
indexare 0-based/1-based dacă nu contrazice explicit textul, recomandări de modernizare.

Dacă nu ești sigur, nu raporta. Fii conservator."""

_DETECTOR_B_SYSTEM = """Ești Detector B — auditor conceptual independent.

Analizează paginile indicate din manualul PDF românesc de informatică.

Caută EXCLUSIV erori conceptuale și de definiții:
- stivă descrisă ca FIFO (corect: LIFO);
- coadă descrisă ca LIFO (corect: FIFO);
- complexitate algoritmică evident greșită (ex: O(n) pentru un algoritm O(n²));
- definiție fundamental greșită a unei structuri de date sau algoritm;
- contradicție clară între explicația din text și pseudocodul/codul prezentat;
- afirmație falsă verificabilă despre comportamentul C/C++ (indexare, pointeri, conversii).

Nu raporta: gramatică, stil de cod, recomandări, modernizări opționale,
lucruri neclare sau care depind de context lipsă, probleme acceptabile la nivel de liceu.

Dacă nu ești sigur, nu raporta. Fii conservator."""

_DETECTOR_RESPONSE_FORMAT = """
Returnează STRICT JSON (fără text în afara JSON-ului):
{
  "erori": [
    {
      "pagina": 12,
      "categorie": "COD|PSEUDOCOD|ALGORITM|CONCEPT|DEFINITIE|STANDARD|COMPLEXITATE",
      "fragment": "fragment exact din manual",
      "corect": "varianta corectă sau regula corectă",
      "explicatie": "de ce este greșit",
      "incredere": 0.85
    }
  ]
}
Dacă nu găsești erori clare, returnează {"erori": []}.
"""

_JUDGE_SYSTEM = """Ești validator expert pentru un manual școlar de informatică.

Primești o listă de candidați de erori detectați de doi agenți independenți.
Sarcina ta:
1. Elimină duplicatele — dacă același fragment e raportat de ambii, păstrează o singură intrare.
2. Elimină fals pozitivele evidente.
3. Păstrează doar erorile tehnice reale, concrete și localizabile.

RESPINGE întotdeauna:
- gramatică, diacritice, fonturi, layout, indentare, stil de cod;
- recomandări și modernizări opționale;
- fragmente vagi sau prea generale (sub 10 caractere semnificative);
- candidați cu câmpul "fragment" gol sau identic cu "corect";
- cod de competiție acceptabil: bits/stdc++.h, using namespace std, variabile scurte, globale, scanf/printf;
- int main() fără return 0 (valid în C++11+);
- probleme acceptabile pedagogic la nivel de liceu.

CONFIRMĂ întotdeauna:
- cout >> sau cin << (operatori inversați în C++);
- Typedef cu majusculă (C++ este case-sensitive);
- int:var sau unsigned:var (sintaxă imposibilă);
- for(l=0; i<n; i++) — variabilă diferită între inițializare și condiție;
- for(i=1; 1<n; i++) — literal în loc de variabilă în condiție;
- stivă descrisă ca FIFO sau coadă ca LIFO;
- complexitate evidentă greșită (O(n) pentru bubble sort etc.);
- void main() / #include <iostream.h> fără nicio contextualizare.

Returnează STRICT JSON:
{
  "erori_validate": [
    {
      "pagina": 12,
      "categorie": "COD|PSEUDOCOD|ALGORITM|CONCEPT|DEFINITIE|STANDARD|COMPLEXITATE",
      "fragment": "fragment exact",
      "corect": "varianta corectă",
      "explicatie": "explicație scurtă",
      "incredere": 0.9
    }
  ]
}"""


def gemini_detector_range(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
    model: str,
    max_errors: int,
    detector_label: str,
    system_prompt: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "agent": detector_label,
        "pages": f"{start_page}-{end_page}",
        "model": model,
        "error": None,
        "items_found": 0,
    }
    prompt = (
        f"{system_prompt}\n\n"
        f"Analizează DOAR paginile {start_page}–{end_page} din PDF-ul atașat.\n"
        f"Raportează cel mult {max_errors} erori.\n"
        f"{_DETECTOR_RESPONSE_FORMAT}"
    )
    txt, err = _gemini_generate(
        model,
        [types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
    )
    if err:
        debug["error"] = err
        return [], debug

    parsed = safe_extract_json(txt or "")
    items = _coerce_items_from_json(parsed, "erori")[:max_errors]
    rows: List[Dict[str, Any]] = []
    for item in items:
        fragment = str(item.get("fragment", "")).strip()
        if not fragment:
            continue
        try:
            confidence = float(item.get("incredere", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        rows.append({
            "pagina": item.get("pagina", start_page),
            "categorie": str(item.get("categorie", "")).strip(),
            "fragment": fragment,
            "corect": str(item.get("corect", "")).strip(),
            "explicatie": str(item.get("explicatie", "")).strip(),
            "incredere": confidence,
            "_source": detector_label,
        })
    debug["items_found"] = len(rows)
    return rows, debug


def judge_text_batch(
    candidates: List[Dict[str, Any]],
    start_page: int,
    end_page: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "agent": "judge",
        "pages": f"{start_page}-{end_page}",
        "model": TOGETHER_JUDGE_MODEL,
        "error": None,
        "candidates_in": len(candidates),
        "confirmed": 0,
        "skipped": False,
    }
    if not candidates:
        debug["skipped"] = True
        return [], debug

    candidates_text = json.dumps(candidates, ensure_ascii=False, indent=2)
    user_text = (
        f"Evaluează și validează următorii {len(candidates)} candidați de erori "
        f"(pagini {start_page}–{end_page}):\n\n{candidates_text}"
    )
    res = together_chat_json(TOGETHER_JUDGE_MODEL, _JUDGE_SYSTEM, user_text, timeout=120)

    if isinstance(res, dict) and res.get("_error"):
        debug["error"] = res.get("_error")
        return [], debug

    validated = _coerce_items_from_json(res, "erori_validate")
    rows: List[Dict[str, Any]] = []
    for item in validated:
        fragment = str(item.get("fragment", "")).strip()
        if not fragment:
            continue
        try:
            confidence = float(item.get("incredere", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        rows.append({
            "validat": True,
            "pagina": item.get("pagina", start_page),
            "categorie": str(item.get("categorie", "")).strip(),
            "text": fragment,
            "sugestie": str(item.get("corect", "")).strip(),
            "explicatie": str(item.get("explicatie", "")).strip(),
            "incredere": confidence,
        })
    debug["confirmed"] = len(rows)
    return rows, debug


def fact_check_run_all(
    pdf_bytes: bytes,
    doc_len: int,
    chapters: List[Dict[str, Any]],
    page_start: int,
    page_end: int,
    pages_per_batch: int,
    max_errors_per_detector: int,
    gem_model: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fact_rows: List[Dict[str, Any]] = []
    fact_debug: List[Dict[str, Any]] = []

    for bs in range(page_start, page_end + 1, pages_per_batch):
        be = min(page_end, bs + pages_per_batch - 1)

        a_rows, a_debug = gemini_detector_range(
            pdf_bytes, bs, be, gem_model, max_errors_per_detector,
            "detector_a", _DETECTOR_A_SYSTEM,
        )
        fact_debug.append(a_debug)

        b_rows, b_debug = gemini_detector_range(
            pdf_bytes, bs, be, gem_model, max_errors_per_detector,
            "detector_b", _DETECTOR_B_SYSTEM,
        )
        fact_debug.append(b_debug)

        candidates = a_rows + b_rows
        validated, j_debug = judge_text_batch(candidates, bs, be)
        fact_debug.append(j_debug)

        for row in validated:
            page = int(row.get("pagina", bs))
            row["capitol"] = chapter_title_for_page(chapters, page) or f"Pagina {page}"
        fact_rows.extend(validated)

    return _dedup_fact_rows(fact_rows), fact_debug


# ============================================================
# GRAMMAR VIA GEMINI
# ============================================================
def gemini_grammar_range(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
    model: str,
    max_errors: int,
) -> Dict[str, Any]:
    prompt = f"""Role: You are a Senior Romanian Philologist specializing in DOOM3 (2021) norms.
Objective: Analyze ONLY pages {start_page}–{end_page} of the PDF and identify linguistic errors.
Categories: Orthography (O), Morphology (M), Syntax (S), Punctuation (P), DOOM3 Specific (D3).
Report only concrete, localizable errors. Do NOT give generic rewriting advice.
Return STRICT JSON with at most {max_errors} items:
{{
  "erori": [
    {{"fragment": "<exact text>", "tip": "O|M|S|P|D3", "corect": "<corrected>", "explicatie": "<brief justification>"}}
  ]
}}"""

    txt, err = _gemini_generate(
        model,
        [types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
    )
    if err:
        return {"erori": [], "_error": err}
    parsed = safe_extract_json(txt or "")
    if isinstance(parsed, dict) and isinstance(parsed.get("erori"), list):
        return parsed
    return {"erori": [], "_raw": (txt or "")[:1200]}


def grammar_run_all(
    pdf_bytes: bytes,
    doc_len: int,
    model: str,
    pages_per_chunk: int,
    max_errors_per_chunk: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    grammar_debug: List[Dict[str, Any]] = []
    cache = st.session_state.app_state["gemini_cache"]

    tip_mapping = {
        "o": "O", "orthography": "O", "ortografie": "O",
        "m": "M", "morphology": "M", "morfologie": "M",
        "s": "S", "syntax": "S", "sintaxa": "S", "sintaxă": "S",
        "p": "P", "punctuation": "P", "punctuatie": "P", "punctuație": "P",
        "d3": "D3", "doom3": "D3", "doom3 specific": "D3",
    }

    for start in range(1, doc_len + 1, pages_per_chunk):
        end = min(doc_len, start + pages_per_chunk - 1)
        key = (start, end, model, max_errors_per_chunk)

        if key in cache:
            parsed = cache[key]
            from_cache = True
        else:
            parsed = gemini_grammar_range(pdf_bytes, start, end, model, max_errors_per_chunk)
            cache[key] = parsed
            from_cache = False

        erori_count = len(parsed.get("erori", [])) if isinstance(parsed.get("erori"), list) else 0
        grammar_debug.append({
            "pages": f"{start}-{end}",
            "from_cache": from_cache,
            "error": parsed.get("_error"),
            "raw_preview": parsed.get("_raw", "")[:400] if "_raw" in parsed else "",
            "erori_gasite": erori_count,
        })

        for e in (parsed.get("erori") or []):
            if not isinstance(e, dict):
                continue
            frag = str(e.get("fragment", "")).strip() or str(e.get("text", "")).strip()
            raw_tip = (
                str(e.get("tip", "")).strip()
                or str(e.get("type", "")).strip()
                or str(e.get("categorie", "")).strip()
            )
            code = tip_mapping.get(raw_tip.lower(), "")
            if not code:
                rl = raw_tip.lower()
                if "ortograf" in rl: code = "O"
                elif "morfolog" in rl: code = "M"
                elif "sintax" in rl or "syntax" in rl: code = "S"
                elif "punct" in rl: code = "P"
                elif "doom" in rl: code = "D3"

            out.append({
                "tip": f"Gramatica ({code})" if code else "Gramatica",
                "capitol": f"Pagini {start}-{end}",
                "text": frag,
                "sugestie": str(e.get("corect", "")).strip(),
                "explicatie": str(e.get("explicatie", "")).strip(),
            })

    st.session_state.app_state["grammar_debug_log"] = grammar_debug
    return out


# ============================================================
# UI
# ============================================================
st.title("AI Manual Auditor (Gemini fact-checking + gramatică · Together judge)")
st.caption(
    "TOC și fact-checking vizual prin Gemini (PDF direct) · "
    "Judge fals-pozitive prin Together AI Llama-3.3-70B · "
    "Gramatică DOOM3 prin Gemini."
)

with st.sidebar:
    st.header("Ce rulez?")
    analysis_mode = st.radio(
        "Selectează analiza",
        ["Ambele", "Fact-checking tehnic", "Gramatica"],
        index=0,
    )

    st.divider()
    st.header("Modele AI")
    gem_model = st.text_input("Model Gemini (TOC + fact-checking + gramatică)", value=DEFAULT_GEMINI_MODEL)
    st.caption(f"Judge fals-pozitive: `{TOGETHER_JUDGE_MODEL}` (Together AI)")

    st.divider()
    st.header("Fact-checking tehnic")
    pages_per_batch = st.slider("Pagini per batch (Detector A + B)", 5, 25, 15)
    max_errors_per_detector = st.slider("Max erori / detector / batch", 2, 10, 4)
    st.caption(
        "**Gemini free:** 15 req/min. "
        f"Fiecare batch = 2 apeluri Gemini. "
        f"Delay automat ≥{GEMINI_MIN_DELAY_S}s între apeluri."
    )

    st.divider()
    st.header("Gramatică (Gemini)")
    pages_per_chunk = st.slider("Pagini per chunk gramatică", 5, 60, 20)
    max_err_chunk = st.slider("Max erori / chunk gramatică", 5, 80, 40)

    st.divider()
    debug_mode = st.checkbox("Mod debug agenți", value=False)


# ─── 1) UPLOAD + TOC ───────────────────────────────────────
if st.session_state.app_state["stage"] == "upload":
    with st.container(border=True):
        st.subheader("1) Încărcare PDF + Cuprins")
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
                toc_end = st.number_input("Cuprins: pagina END", min_value=1, max_value=doc_len, value=min(4, doc_len))

            if st.button("Extrage cuprinsul (Gemini)", type="primary"):
                a, b = min(int(toc_start), int(toc_end)), max(int(toc_start), int(toc_end))
                with st.spinner(f"Extrag cuprinsul din paginile {a}–{b} cu Gemini..."):
                    items, toc_err = gemini_extract_toc(pdf_bytes, a, b, gem_model)
                if toc_err:
                    st.error(f"Eroare extragere cuprins: {toc_err}")
                elif not items:
                    st.warning("Gemini nu a detectat capitole. Verifică intervalul de pagini sau adaugă manual.")
                else:
                    st.session_state.app_state["structure_data"] = items
                    st.success(f"Capitole detectate: {len(items)}")

            if st.session_state.app_state["structure_data"]:
                if st.button("➡️ Mergi la Validare Structură"):
                    st.session_state.app_state["stage"] = "approve"
                    st.rerun()

            st.divider()
            if st.button("Omite cuprins (segmentare automată 10 pag)"):
                st.session_state.app_state["structure_data"] = [
                    {"titlu": f"Segment {i // DEFAULT_SEGMENT_SIZE + 1}", "start": i + 1}
                    for i in range(0, doc_len, DEFAULT_SEGMENT_SIZE)
                ]
                st.session_state.app_state["stage"] = "approve"
                st.rerun()


# ─── 2) APPROVE STRUCTURE ──────────────────────────────────
elif st.session_state.app_state["stage"] == "approve":
    with st.container(border=True):
        st.subheader("2) Validare Structură")
        df = pd.DataFrame(st.session_state.app_state["structure_data"])
        for col, default in [("titlu", "Capitol"), ("start", 1), ("end", 0)]:
            if col not in df.columns:
                df[col] = default

        edited_df = st.data_editor(
            df,
            column_config={
                "titlu": "Titlu Capitol",
                "start": st.column_config.NumberColumn("Start", format="%d"),
                "end": st.column_config.NumberColumn("Final", format="%d"),
            },
            width="stretch",
            num_rows="dynamic",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirmă și Analizează", type="primary"):
                structure = edited_df.to_dict("records")
                st.session_state.app_state["chapters"] = extract_chapter_text(
                    st.session_state.app_state["pdf_bytes"], structure
                )
                for k in ["final_report", "fact_report", "grammar_report", "debug_log",
                          "fact_debug_log", "grammar_debug_log"]:
                    st.session_state.app_state[k] = []
                st.session_state.app_state["gemini_cache"] = {}
                st.session_state.app_state["audit_ran"] = False
                st.session_state.app_state["stage"] = "analyze"
                st.rerun()
        with c2:
            if st.button("Resetează"):
                st.session_state.app_state = {
                    "stage": "upload", "pdf_bytes": None, "doc_len": 0,
                    "structure_data": [], "chapters": [],
                    "final_report": [], "fact_report": [], "grammar_report": [],
                    "gemini_cache": {}, "debug_log": [], "fact_debug_log": [],
                    "grammar_debug_log": [], "audit_ran": False, "_last_gemini_call": 0.0,
                }
                st.rerun()


# ─── 3) ANALYZE + REPORT ───────────────────────────────────
elif st.session_state.app_state["stage"] == "analyze":
    chapters = st.session_state.app_state["chapters"]
    pdf_bytes = st.session_state.app_state["pdf_bytes"]
    doc_len = st.session_state.app_state["doc_len"]
    audit_ran = st.session_state.app_state.get("audit_ran", False)

    st.subheader(f"3) Analiză ({len(chapters)} capitole)")

    if analysis_mode in ("Ambele", "Fact-checking tehnic"):
        c_s, c_e = st.columns(2)
        with c_s:
            fact_page_start = st.number_input("Fact-checking: pagina START", min_value=1, max_value=doc_len, value=1)
        with c_e:
            fact_page_end = st.number_input("Fact-checking: pagina END", min_value=1, max_value=doc_len, value=doc_len)
    else:
        fact_page_start, fact_page_end = 1, doc_len

    if not audit_ran:
        if st.button("Start Audit", type="primary"):
            fact_rows: List[Dict[str, Any]] = []
            grammar_rows: List[Dict[str, Any]] = []
            fact_debug: List[Dict[str, Any]] = []
            status = st.status("Rulez auditul...", expanded=True)

            if analysis_mode in ("Ambele", "Fact-checking tehnic"):
                ps = min(int(fact_page_start), int(fact_page_end))
                pe = max(int(fact_page_start), int(fact_page_end))
                n_batches = max(1, (pe - ps + 1 + pages_per_batch - 1) // pages_per_batch)
                status.write(
                    f"Fact-checking: {pe - ps + 1} pagini · {n_batches} batch-uri · "
                    f"Detector A + B (Gemini) + Judge (Llama-3.3-70B)"
                )
                progress = st.progress(0.0)
                batch_idx = 0
                for bs in range(ps, pe + 1, pages_per_batch):
                    be = min(pe, bs + pages_per_batch - 1)
                    batch_idx += 1
                    status.write(f"Batch {batch_idx}/{n_batches}: pagini **{bs}–{be}**")

                    a_rows, a_debug = gemini_detector_range(
                        pdf_bytes, bs, be, gem_model, max_errors_per_detector,
                        "detector_a", _DETECTOR_A_SYSTEM,
                    )
                    fact_debug.append(a_debug)

                    b_rows, b_debug = gemini_detector_range(
                        pdf_bytes, bs, be, gem_model, max_errors_per_detector,
                        "detector_b", _DETECTOR_B_SYSTEM,
                    )
                    fact_debug.append(b_debug)

                    validated, j_debug = judge_text_batch(a_rows + b_rows, bs, be)
                    fact_debug.append(j_debug)

                    for row in validated:
                        page = int(row.get("pagina", bs))
                        row["capitol"] = chapter_title_for_page(chapters, page) or f"Pagina {page}"
                    fact_rows.extend(validated)
                    progress.progress(batch_idx / n_batches)

                fact_rows = _dedup_fact_rows(fact_rows)

            if analysis_mode in ("Ambele", "Gramatica"):
                status.write("Gramatică (Gemini pe PDF, chunked)...")
                grammar_rows = grammar_run_all(pdf_bytes, doc_len, gem_model, pages_per_chunk, max_err_chunk)

            st.session_state.app_state["fact_report"] = fact_rows
            st.session_state.app_state["grammar_report"] = grammar_rows
            st.session_state.app_state["final_report"] = fact_rows + grammar_rows
            st.session_state.app_state["fact_debug_log"] = fact_debug
            st.session_state.app_state["debug_log"] = fact_debug
            st.session_state.app_state["audit_ran"] = True
            status.update(label="Analiză completă!", state="complete", expanded=False)
            st.rerun()

    if audit_ran:
        fact_report = st.session_state.app_state.get("fact_report", [])
        grammar_report = st.session_state.app_state.get("grammar_report", [])

        if not fact_report and not grammar_report:
            st.warning(
                "Auditul a rulat dar nu s-au găsit erori. "
                "Activează 'Mod debug agenți' și rulează din nou pentru detalii API."
            )

        # ── Fact-checking table ──────────────────────────────
        if analysis_mode in ("Ambele", "Fact-checking tehnic"):
            st.markdown("### Fact-checking tehnic")
            if not fact_report:
                st.info("Nu au fost validate erori tehnice/factuale.")
            else:
                st.success(f"Erori tehnice/factuale validate: {len(fact_report)}")
                df_fact = pd.DataFrame(fact_report)
                for col in ["validat", "pagina", "categorie", "capitol", "text", "sugestie", "explicatie", "incredere"]:
                    if col not in df_fact.columns:
                        df_fact[col] = "" if col != "validat" else True
                df_fact = df_fact[["validat", "pagina", "categorie", "capitol", "text", "sugestie", "explicatie", "incredere"]]

                counts = df_fact["categorie"].replace("", "Necategorizat").value_counts()
                cols = st.columns(min(len(counts), 5))
                for i, (tip, cnt) in enumerate(counts.items()):
                    with cols[i % len(cols)]:
                        st.metric(str(tip), int(cnt))

                val_fact_df = st.data_editor(
                    df_fact,
                    column_config={
                        "validat": st.column_config.CheckboxColumn("Valid?", default=True),
                        "pagina": st.column_config.NumberColumn("Pagina", format="%d"),
                        "categorie": "Categorie",
                        "capitol": "Capitol",
                        "text": "Fragment problematic",
                        "sugestie": "Corectare / regulă corectă",
                        "explicatie": "Explicație",
                        "incredere": st.column_config.NumberColumn("Încredere", format="%.2f"),
                    },
                    width="stretch",
                    height=420,
                    key="fact_editor",
                )
                valid_fact = val_fact_df[val_fact_df["validat"] == True].copy()
                c1, c2 = st.columns([1.2, 1.6])
                with c1:
                    if not valid_fact.empty:
                        st.download_button(
                            "Descarcă CSV fact-checking",
                            valid_fact.to_csv(index=False).encode("utf-8"),
                            "raport_fact_checking.csv", "text/csv",
                        )
                with c2:
                    if not valid_fact.empty:
                        st.download_button(
                            "Descarcă Excel fact-checking",
                            dataframe_to_pretty_excel_bytes(valid_fact, "Fact checking"),
                            "raport_fact_checking.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

        if analysis_mode == "Ambele":
            st.divider()

        # ── Grammar table ────────────────────────────────────
        if analysis_mode in ("Ambele", "Gramatica"):
            st.markdown("### Gramatică Gemini")
            if not grammar_report:
                st.info("Nu au fost găsite erori gramaticale.")
            else:
                st.success(f"Erori gramaticale găsite: {len(grammar_report)}")
                df_grammar = pd.DataFrame(grammar_report)
                for col in ["validat", "tip", "capitol", "text", "sugestie", "explicatie"]:
                    if col not in df_grammar.columns:
                        df_grammar[col] = False if col == "validat" else ""
                df_grammar = df_grammar[["validat", "tip", "capitol", "text", "sugestie", "explicatie"]]

                counts_g = df_grammar["tip"].value_counts()
                cols_g = st.columns(min(len(counts_g), 5))
                for i, (tip, cnt) in enumerate(counts_g.items()):
                    with cols_g[i % len(cols_g)]:
                        st.metric(str(tip), int(cnt))

                val_grammar_df = st.data_editor(
                    df_grammar,
                    column_config={
                        "validat": st.column_config.CheckboxColumn("Valid?", default=False),
                        "tip": "Tip",
                        "capitol": "Pagini",
                        "text": "Fragment problematic",
                        "sugestie": "Corectare",
                        "explicatie": "Explicație",
                    },
                    width="stretch",
                    height=420,
                    key="grammar_editor",
                )
                valid_grammar = val_grammar_df[val_grammar_df["validat"] == True].copy()
                c1, c2 = st.columns([1.2, 1.6])
                with c1:
                    if not valid_grammar.empty:
                        st.download_button(
                            "Descarcă CSV gramatică",
                            valid_grammar.to_csv(index=False).encode("utf-8"),
                            "raport_gramatica.csv", "text/csv",
                        )
                with c2:
                    if not valid_grammar.empty:
                        st.download_button(
                            "Descarcă Excel gramatică",
                            dataframe_to_pretty_excel_bytes(valid_grammar, "Gramatica"),
                            "raport_gramatica.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

        st.divider()
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Repornește analiza"):
                for k in ["final_report", "fact_report", "grammar_report",
                          "debug_log", "fact_debug_log", "grammar_debug_log"]:
                    st.session_state.app_state[k] = []
                st.session_state.app_state["gemini_cache"] = {}
                st.session_state.app_state["audit_ran"] = False
                st.rerun()
        with c4:
            if st.button("Analizează alt document"):
                st.session_state.app_state = {
                    "stage": "upload", "pdf_bytes": None, "doc_len": 0,
                    "structure_data": [], "chapters": [],
                    "final_report": [], "fact_report": [], "grammar_report": [],
                    "gemini_cache": {}, "debug_log": [], "fact_debug_log": [],
                    "grammar_debug_log": [], "audit_ran": False, "_last_gemini_call": 0.0,
                }
                st.rerun()

        # ── Debug panels ─────────────────────────────────────
        if debug_mode:
            grammar_debug_log = st.session_state.app_state.get("grammar_debug_log", [])
            if grammar_debug_log:
                gem_errors = [e for e in grammar_debug_log if e.get("error")]
                with st.expander(
                    f"Debug Gramatică Gemini ({len(grammar_debug_log)} chunk-uri · {len(gem_errors)} erori API)",
                    expanded=bool(gem_errors),
                ):
                    summary = [
                        {
                            "Pagini": e.get("pages", ""),
                            "Cache": e.get("from_cache", False),
                            "Erori găsite": e.get("erori_gasite", 0),
                            "Eroare API": str(e.get("error") or ""),
                        }
                        for e in grammar_debug_log
                    ]
                    st.dataframe(pd.DataFrame(summary), use_container_width=True)

            fact_debug_log = st.session_state.app_state.get("fact_debug_log", [])
            if fact_debug_log:
                fact_errors = [e for e in fact_debug_log if e.get("error")]
                with st.expander(
                    f"Debug Fact-checking ({len(fact_debug_log)} apeluri · {len(fact_errors)} erori API)",
                    expanded=bool(fact_errors),
                ):
                    batch_rows = []
                    i = 0
                    while i < len(fact_debug_log):
                        entry = fact_debug_log[i]
                        if entry.get("agent") == "detector_a":
                            row: Dict[str, Any] = {
                                "Pagini": entry.get("pages", ""),
                                "Candidați A": entry.get("items_found", 0),
                                "Eroare A": str(entry.get("error") or ""),
                                "Candidați B": "",
                                "Eroare B": "",
                                "Judge": "",
                                "Validate": "",
                                "Eroare Judge": "",
                            }
                            if i + 1 < len(fact_debug_log) and fact_debug_log[i + 1].get("agent") == "detector_b":
                                b = fact_debug_log[i + 1]
                                row["Candidați B"] = b.get("items_found", 0)
                                row["Eroare B"] = str(b.get("error") or "")
                                i += 1
                            if i + 1 < len(fact_debug_log) and fact_debug_log[i + 1].get("agent") == "judge":
                                j = fact_debug_log[i + 1]
                                row["Judge"] = "Sărit (0 candidați)" if j.get("skipped") else "Rulat"
                                row["Validate"] = j.get("confirmed", 0)
                                row["Eroare Judge"] = str(j.get("error") or "")
                                i += 1
                            batch_rows.append(row)
                        i += 1

                    if batch_rows:
                        st.dataframe(pd.DataFrame(batch_rows), use_container_width=True)
                    if fact_errors:
                        st.markdown("#### Erori API detaliate")
                        for e in fact_errors:
                            st.error(
                                f"**{e.get('agent')}** · pagini {e.get('pages', '?')} · "
                                f"model {e.get('model', '?')}\n\n{e.get('error')}"
                            )
