# AI Textbook Validator & Auditor

Un sistem software de tip enterprise bazat pe arhitectură Multi-Agent, proiectat pentru automatizarea validării și auditării manualelor școlare și a documentelor istorice în format PDF. Soluția integrează modele de limbaj de mari dimensiuni (LLM) și viziune computerizată (Computer Vision) pentru a efectua extragerea structurală, verificarea factuală, detecția bias-ului semantic și analiza comparativă interculturală.

## Prezentare Generală

AI Textbook Validator adresează limitările metodelor tradiționale de OCR și corectură manuală printr-o arhitectură stratificată de agenți AI specializați. Sistemul procesează documentele pentru a identifica:

* **Integritatea Structurală:** Extragerea precisă a Cuprinsului utilizând modele Vision, eliminând erorile cauzate de layout-uri complexe sau fonturi stilizate.
* **Acuratețea Factuală:** Verificarea datelor istorice, numelor și tratatelor prin referințe încrucișate cu surse locale și internaționale.
* **Bias Semantic:** Detectarea limbajului propagandistic, a tonului absolutist și a omisiunilor contextuale.
* **Consistență Internațională:** Analiza comparativă a narativelor locale față de consensul istoric internațional.

## Arhitectura Sistemului

Aplicația utilizează o **Arhitectură de tip Ensemble**, unde modele distincte sunt alocate unor sarcini specifice în funcție de capabilitățile lor arhitecturale. Procesul este orchestrat de un controler central (`AgentCouncil`) care gestionează execuția paralelă și sinteza datelor.

### 1. Nivelul de Ingestie (Vision-First)
* **Model:** GPT-4o Vision.
* **Funcție:** Analizează pagina de Cuprins ca imagine, nu ca text brut. Această abordare ocolește erorile de codare (encoding) frecvente în PDF-urile vechi și gestionează eficient structurile pe coloane multiple.

### 2. Nivelul de Analiză (Procesare Paralelă)
După extragerea structurală, conținutul este segmentat și procesat concurent de trei agenți specializați:

* **Agent de Analiză Semantică și Bias (Claude 3.5 Sonnet):** Specializat în înțelegerea nuanțelor lingvistice. Scanează textul pentru interpretări subiective, erori logice și lipsă de neutralitate.
* **Agent de Audit Internațional (Claude 3.5 Sonnet + Web Search):** Realizează un audit transcultural. Traduce afirmațiile cheie în engleză, interoghează baze de date internaționale (via Tavily) și semnalează discrepanțele dintre textul local și istoriografia globală.
* **Agent de Verificare Factuală Locală (GPT-4o + Web Search):** Se concentrează pe verificarea rigidă a datelor. Validează date specifice, referințe legislative și figuri istorice locale folosind surse românești, efectuând simultan verificări gramaticale stricte.

### 3. Nivelul de Sinteză (Agregare și Filtrare)
* **Model:** GPT-4o.
* **Funcție:** Acționează ca motor de consens final. Agregă rezultatele celor trei agenți de analiză, elimină duplicatele și filtrează rezultatele fals-pozitive (precum artefactele de encoding sau afirmațiile neverificabile).

## Diagrama Fluxului de Lucru

```mermaid
graph TD
    A[Document PDF] -->|Extragere Imagine Pagină| B[Motor Vision GPT-4o]
    B --> C[Date Structurate JSON]

    subgraph AnalizaParalela [Nucleu de Analiză Paralelă]
        C -->|Text Capitol| D[Agent Bias & Semantică]
        C -->|Text Capitol| E[Agent Audit Internațional]
        C -->|Text Capitol| F[Agent Factual Local]
    end

    D -->|Raport Analiză| G[Modul de Sinteză GPT-4o]
    E -->|Raport Comparativ| G
    F -->|Raport Validare| G

    G -->|Filtrare Zgomot & Logică| H[Raport Final CSV]
