# Agent-driven Biomedical Knowledge Graph

An end-to-end agentic pipeline that automatically extracts biomedical knowledge from scientific literature and stores it as a queryable knowledge graph. The system processes research papers through a 6-step modular pipeline — from raw PDF cleaning to structured Neo4j graph storage — and exposes an interactive natural language QA interface with full source attribution and NCIt ontology traceability.

---

## Demo

```
🧠 Welcome to the Biomedical Knowledge Graph Pipeline
1. Run full pipeline
2. Go directly to QA
3. Exit
Select an option (1/2/3): 2

🔍 Launching interactive QA system...
Choose QA model backend ('ollama', 'openai', or 'gemini'): gemini
✅ Successfully connected to Neo4j

🔎 Starting QA Session with GEMINI (type 'exit' to end)

Question: what is breast cancer

✅ Gemini responded using model: gemini-2.5-flash

💡 Breast cancer is a manifestation of:
- malignancies [source: New_antibody-drug_conjugates_(ADCs)_in_breast_canc]
- malignancy [source: DNA_damage_targeted_therapy_for_advanced_breast_ca]

It is associated with:
- altered gut microbiota [source: Does_the_gut_microbiome_environment_influence_resp]
- germline mutations in BRCA1/2 gene [source: DNA_damage_targeted_therapy_for_advanced_breast_ca]
- hereditary forms [source: DNA_damage_targeted_therapy_for_advanced_breast_ca]

Breast cancer can be diagnosed by:
- Infrared breast thermography [NCIT IDs: C62663] [source: Development_and_validation_of_an_infrared-artifici]
- infrared-artificial intelligence software [source: Development_and_validation_of_an_infrared-artifici]

Breast cancer predisposes to:
- brain metastasis [source: Endocrine_resistant_breast_cancer:_brain_metastasi]

Question: What drugs target both breast cancer and diabetes?
💡 This information is not in the knowledge graph.
```

Every answer is grounded in the knowledge graph — the system cites the exact source paper and NCIt ontology ID for each relationship. When information is absent, it says so honestly rather than hallucinating.

---

## Pipeline Architecture

```
PDF Papers
    │
    ▼
┌─────────────────────┐
│  1. PDF Cleaning     │  Remove noise, headers, references
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. NER Agent        │  BioBERT — extract genes, diseases,
│                     │  drugs, proteins, mutations
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. Entity Cleaning  │  Normalize and deduplicate entities
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. Relation         │  LLM agents — extract causal
│     Extraction       │  relationships between entities
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  5. Ontology         │  NCIt validation — verify entities
│     Validation       │  against NCI Thesaurus, assign IDs
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  6. Neo4j Storage    │  Persist graph + expose QA interface
│     + QA             │  with citation tracking
└─────────────────────┘
```

---

## Key Features

- **Multi-LLM backend** — switch between Ollama (local), OpenAI GPT-4o, and Google Gemini at runtime with automatic fallback model selection
- **Transformer-based NER** — BioBERT extracts biomedical entities (genes, diseases, drugs, proteins, mutations, pathways) from raw text
- **Causal relationship extraction** — LLM agents identify typed relationships between entity pairs across 9 research papers
- **NCIt ontology validation** — every extracted entity is validated against the NCI Thesaurus before graph storage, reducing LLM hallucinations
- **Source-attributed QA** — each answer surfaces the supporting paper filename and NCIt ontology ID for full reproducibility
- **Honest grounding** — returns "This information is not in the knowledge graph" rather than hallucinating answers
- **Resource-efficient** — supports quantized local models (`llama3.2:3b` via Ollama) for consumer hardware, with cloud fallback
- **Robust Neo4j connection** — automatic reconnection on idle dropout via Neo4j Aura

---

## Tech Stack

| Layer | Technology |
|---|---|
| NER model | BioBERT |
| LLM backends | GPT-4o, Gemini 2.5 Flash, LLaMA (Ollama) |
| Graph database | Neo4j Aura |
| Ontology | NCIt — NCI Thesaurus (OWL) |
| Language | Python 3.10+ |
| Local inference | Ollama |

---

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:Abrham-m/Agent-driven-Biomedical-Knowledge-Graph.git
cd Agent-driven-Biomedical-Knowledge-Graph
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
# OpenAI (for GPT-4o backend)
OPENAI_API_KEY=your_openai_api_key

# Google Gemini (for Gemini backend)
GEMINI_API_KEY=your_gemini_api_key

# Neo4j Aura
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 5. Download NLTK data

```bash
python extra_n/download_nltk.py
```

### 6. Download research papers

```bash
python dataset/download_papers.py
```

This downloads all PDFs into `dataset/research_papers/` based on `dataset/papers.json`.

### 7. Set up NCIt ontology index

The pipeline requires `ncit_indexes.pkl` in the root directory. Three options:

**Option A — Auto-generate (recommended):**
```bash
python ontology_download.py
python ontology_inspector.py
```

**Option B — Manual download:**  
Download the OWL file from https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/, place it in the `ncit/` folder as `Cancer_Thesaurus.owl`, then run `ontology_inspector.py`.

**Option C — Use pre-built index:**  
Extract the provided `ncit_indexes.pkl.zip` from the `lib/` folder into the root directory.

### 8. Start local LLM (Ollama only)

If using the Ollama backend:
```bash
ollama serve
ollama pull llama3.2:3b
```

---

## Usage

### Run the full pipeline

Processes all papers end-to-end and populates the knowledge graph:

```bash
python main_pipeline_single_run.py
```

### Go directly to QA

If the graph is already populated, skip straight to the QA interface:

```bash
python main_pipeline_single_run.py
# Select option 2: Go directly to QA
# Choose backend: ollama / openai / gemini
```

### QA session commands

| Command | Description |
|---|---|
| Any question | Query the knowledge graph |
| `summary` | Show graph statistics |
| `types` | List all relationship types in the graph |
| `exit` | End the session |

---

## Project Structure

```
Agent-driven-Biomedical-Knowledge-Graph/
├── main_pipeline_single_run.py   # Entry point
├── dataset/
│   ├── papers.json               # Paper URLs/metadata
│   ├── download_papers.py        # PDF downloader
│   └── research_papers/          # Downloaded PDFs
├── extra_n/
│   └── download_nltk.py          # NLTK setup
├── ncit/
│   └── Cancer_Thesaurus.owl      # NCIt ontology (downloaded)
├── lib/
│   ├── ncit_indexes.pkl.zip      # Pre-built ontology index
│   └── venv.zip                  # Pre-built virtual environment
├── .env                          # API keys (not committed)
└── requirements.txt
```

---

## Troubleshooting

**Neo4j connection drops during QA**  
The system automatically reconnects. If it persists, check your Neo4j Aura instance is active and credentials in `.env` are correct.

**Missing dependencies**  
```bash
pip install -r requirements.txt
```
Or use the pre-built environment: `unzip lib/venv.zip -d ./ && source venv/bin/activate`

**Ollama model not found**  
```bash
ollama pull llama3.2:3b
```

**NCIt index missing**  
Run `python ontology_download.py` then `python ontology_inspector.py`, or extract from `lib/ncit_indexes.pkl.zip`.

---
