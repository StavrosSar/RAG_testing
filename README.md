## RAG POC for Manuals / Procedures

### Overview
End-to-end Retrieval-Augmented Generation (RAG) proof-of-concept for technical manuals,
providing extractive answers with citations, plus an evaluation harness (precision@k + failure logs).

### Goals
- RAG POC for manuals/procedures with citations (grounded answers)
- Evaluation set + precision@k metrics
- Failure logs + suggested fixes

### Pipeline
1. PDF ingestion → text (data_raw → data_text)
2. Cleaning + chunking (overlap) → JSONL (data_processed/chunks.jsonl)
3. Retriever v0:
   - TF-IDF (baseline) OR
   - BM25 (pure Python)
4. Answer generation:
   - extractive sentence selection + citations
5. FastAPI `/query` endpoint
6. Evaluation:
   - 20 labeled questions
   - precision@3, precision@5, hit@5
   - failures.jsonl with failure taxonomy + suggested fixes

---

## Data
- Manuals stored as PDF (data_raw/)
- Extracted text stored as .txt (data_text/)
- Final retrieval corpus: `data_processed/chunks.jsonl` (one chunk per line)

Each chunk has:
- doc_id, chunk_id, text, source (+ offsets if you kept them)

---


---

## Installing the Requirements.
```powershell
& "C:\Users\ssar\AppData\Local\anaconda3\python.exe" -m pip install -r requirements.txt
```

---


## CLI Usage

### Ask a single question (demo)
TF-IDF:
```powershell
& "C:\Users\ssar\AppData\Local\anaconda3\python.exe" -m src.query_cli --retriever tfidf --k 5 --query "What is the AC voltage?"
```

BM25:
```powershell
& "C:\Users\ssar\AppData\Local\anaconda3\python.exe" -m src.query_cli --retriever bm25 --k 5 --query "What is the AC voltage?"
```

## API 
```powershell
&"C:\Users\ssar\AppData\Local\anaconda3\python.exe" -m uvicorn src.api:app --reload
```
Then open http://127.0.0.1:8000/docs and try out "Guidelines for Returning Condition Values on OpenVMS"

## Evaluation 
TF-IDF:
```powershell
& "C:\Users\ssar\AppData\Local\anaconda3\python.exe" -m src.eval --retriever tfidf
```

BM25:
```powershell
& "C:\Users\ssar\AppData\Local\anaconda3\python.exe" -m src.eval --retriever bm25
```

Outputs written to eval/:
- report_tfidf.csv, failures_tfidf.jsonl
- report_bm25.csv, failures_bm25.jsonl

## Runing the Demo
```powershell
& "C:\Users\ssar\AppData\Local\anaconda3\python.exe" -m scrripts\demo_queries.py
```


## UI of the LLM
```powershell
& "C:\Users\ssar\AppData\Local\anaconda3\python.exe" -m streamlit run src\chat_ui.py 
```
http://localhost:8501/
