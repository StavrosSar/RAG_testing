# RAG POC for Technical Manuals (BM25 vs TF-IDF)

End-to-end **Retrieval-Augmented Generation (RAG)** proof-of-concept for technical manuals.

This repository focuses on:
- **retrieval quality** (BM25 vs TF-IDF),
- **repeatable evaluation**,
- **SQL-backed chunk storage**, and
- a **working FastAPI backend + Streamlit UI demo**.

Answers are **extractive**, with chunk-level citations.

---

## Repository structure

```
data_raw/           # source PDFs
data_text/          # extracted text (optional)
data_processed/     # local retrieval artifacts (optional)
eval/               # eval_questions.jsonl + reports
scripts/            # helpers
src/                # retrievers, API, SQL ingestion, evaluation
tests/              # small smoke tests
ui/                 # optional static UI assets
```

---

## Quickstart (fresh environment)

### 1) Clone the repository
```bash
git clone <REPO_URL>
cd RAG_testing
```

### 2) Create & activate virtual environment

**Windows (CMD / PowerShell)**
```bat
python -m venv venv
venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## SQL-backed chunk storage (MSSQL)

This project supports **retrieval directly from SQL Server** (no local chunk files required at query time).

### Supported setup
- SQL Server (LocalDB or full SQL Server)
- Tested with **SQL Server Management Studio (SSMS) + LocalDB**
- ODBC Driver **18 for SQL Server**

### Environment variable

The API loads chunks from SQL Server using an environment variable.

**Example (LocalDB, Windows Authentication):**
```bat
set "MSSQL_CONN_STR=Driver={ODBC Driver 18 for SQL Server};Server=(localdb)\MSSQLLocalDB;Database=RAGDB;Trusted_Connection=yes;TrustServerCertificate=yes;"
```

Verify that it is set:
```bat
echo %MSSQL_CONN_STR%
```

Optional sanity check:
```bat
python -c "import os,pyodbc; pyodbc.connect(os.environ['MSSQL_CONN_STR']); print('MSSQL connection OK')"
```

---

## env.example (for submission)

Create an `env.example` file in the repository root:

```env
MSSQL_CONN_STR=Driver={ODBC Driver 18 for SQL Server};Server=YOUR_SERVER;Database=YOUR_DB;Trusted_Connection=yes;TrustServerCertificate=yes;
```

⚠️ Do **not** commit real credentials.

---

## API (FastAPI)

The API handles:
- retrieval (BM25 / TF-IDF),
- SQL-backed chunk loading,
- extractive answering,
- citations.

### Run the API (Terminal A)
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Expected output:
```
Uvicorn running on http://127.0.0.1:8000
```

Swagger documentation:
```
http://127.0.0.1:8000/docs
```

### API endpoint
```
POST /query
```

Example payload:
```json
{
  "question": "AC voltage frequency environmental conditions",
  "retriever": "bm25_sql",
  "top_k": 5,
  "mode": "llm"
}
```

---

## Streamlit UI (Chat Demo)

The Streamlit app provides a chat-style interface on top of the API.

### Run Streamlit (Terminal B)
```bash
streamlit run src/chat_ui.py
```

Open in browser:
```
http://localhost:8501
```

### UI features
- Retriever selection:
  - `bm25_sql`, `tfidf_sql` → **SQL-backed chunks**
  - `bm25`, `tfidf` → local retrieval
- LLM vs extractive mode
- Top-K control
- Optional source display

---

## Evaluation (BM25 vs TF-IDF)

Run evaluation against the same `eval/eval_questions.jsonl`.

```bash
python -m src.eval --retriever bm25
python -m src.eval --retriever tfidf
```

Outputs written to `eval/`:
- `report_bm25.csv`, `failures_bm25.jsonl`
- `report_tfidf.csv`, `failures_tfidf.jsonl`

BM25 is used as the **baseline retriever**.

---

## Query (CLI)

BM25:
```bash
python -m src.query_cli --retriever bm25 --k 5 --query "What is the AC voltage?"
```

TF-IDF:
```bash
python -m src.query_cli --retriever tfidf --k 5 --query "What is the AC voltage?"
```

---

## Mini tests

Lightweight smoke tests for retrieval and query utilities:

```bash
pytest -q
```

---

## Retrieval improvements included

### Query normalization & expansion

`src/query_utils.py` applies conservative normalization and expansion for common
manual/spec terminology, such as:

- **AC voltage** ⇄ *input voltage*
- **frequency** → *Hz*
- **environmental conditions** → *temperature / humidity*

These are applied automatically to **both BM25 and TF-IDF** retrievers.

---

## Reproducibility notes

- All commands run from a **fresh virtual environment**
- No hard-coded local paths
- SQL access is fully controlled via environment variables
- `.env` files are ignored; `env.example` is provided

---

## Summary

This repository demonstrates:
- SQL-backed RAG retrieval
- BM25 vs TF-IDF comparison
- Reproducible evaluation
- FastAPI backend
- Streamlit demo UI