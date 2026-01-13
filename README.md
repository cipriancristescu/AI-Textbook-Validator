#  AI Manual Auditor (Enterprise Edition)

**Un sistem avansat Multi-Agent pentru auditarea manualelor școlare și a documentelor istorice.**

Acest instrument utilizează o arhitectură de tip "Ensemble" (Consiliu de Agenți AI) pentru a detecta greșeli factuale, erori gramaticale, propagandă, bias și omisiuni de context în manualele școlare (PDF). Se diferențiază prin utilizarea **GPT-4o Vision** pentru structură și a unei echipe de agenți (OpenAI + Anthropic) pentru analiză încrucișată.

---

##  Funcționalități Cheie

### 1. Analiza Vizuală a Cuprinsului (Vision-First)
- **Problema:** OCR-ul clasic eșuează la cuprinsurile stilizate sau pe coloane.
- **Soluția:** Folosim **GPT-4o Vision** pentru a "privi" pagina de cuprins ca pe o imagine și a extrage structura capitolelor perfect, indiferent de fonturi sau layout.

### 2. Arhitectură Multi-Agent
Analiza nu este făcută de un singur model, ci de 3 experți specializați care rulează în paralel:
*  **Agent Nuanțe & Bias (Claude 3.5 Sonnet):** Expert în limba română și analiză de text. Detectează propagandă, limbaj de lemn și ton absolutist.
*  **Auditor Internațional (Claude 3.5 + Tavily):** Traduce afirmațiile cheie în engleză și le verifică în surse internaționale (Google Academic, Britannica) pentru a detecta izolaționismul istoric.
* 🇷🇴 **Fact-Checker Local (GPT-4o + Tavily):** Verifică date fixe (ani, nume, tratate) în surse românești și corectează gramatica.

### 3. Agentul expert
Un model final (**GPT-4o**) primește rapoartele celor 3 agenți și:
* Elimină zgomotul (erori de encoding, fonturi stricate).
* Elimină erorile false (unde nu există surse).
* Compilează un raport unic, curat și validat.

---

##  Arhitectura Sistemului

```mermaid
graph TD
    PDF[PDF Manual] -->|Screenshot Pagina Cuprins| Vision[ GPT-4o Vision]
    Vision --> StructuraJSON

    subgraph "Camera de Analiză (Paralel)"
        StructuraJSON -->|Text Capitol| Claude[ CLAUDE 3.5<br/>Bias & Nuanțe]
        StructuraJSON -->|Text Capitol| GPT_RO[🇷🇴 GPT-4o + Tavily<br/>Fapte Locale & Gramatică]
        StructuraJSON -->|Text Capitol| GPT_INT[ Claude 3.5 + Tavily<br/>Perspective Internaționale]
    end

    Claude -->|Raport Bias| Judge
    GPT_RO -->|Raport Fapte RO| Judge
    GPT_INT -->|Raport Diferențe| Judge

    Judge[ THE JUDGE (GPT-4o)<br/>Sinteză & Filtrare Logică]
    Judge --> UI[Raport Final CSV]
