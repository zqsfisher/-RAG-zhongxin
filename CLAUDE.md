# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python environment: `conda activate rag`

Install (first time): `pip install -e . -r requirements.txt`

Required env vars in `.env` (see `.gitignore`):
- `DASHSCOPE_API_KEY` — primary provider; used for embeddings (`text-embedding-v1`) and LLM (`qwen-turbo-latest`)
- `OPENAI_API_KEY` — optional, for `gpt-4o` / `gpt-4o-mini`
- `GEMINI_API_KEY` — optional
- `IBM_API_KEY` — optional, competition-specific
- `JINA_API_KEY` — optional, for Jina reranker

## Commands

```bash
# Run the full pipeline (edit the `if __name__ == "__main__"` block to toggle steps)
python src/pipeline.py

# Run the Streamlit UI
streamlit run app_streamlit.py

# Test MinerU local PDF parsing
python test_mineru.py
```

## Architecture

This is a RAG system for answering questions about company annual reports (currently SMIC/中芯国际). It was the winning solution in the RAG Challenge competition.

### Data flow

```
PDF files → [MinerU API or local CLI] → .md files
  → TextSplitter.split_markdown_reports() → chunked .json (one per report)
  → VectorDBIngestor (FAISS IP) / BM25Ingestor
  → QuestionsProcessor: retrieval → (optional LLM rerank) → RAG context → LLM answer → page validation
```

### Core modules

| Module | Role |
|---|---|
| `src/pipeline.py` | Main orchestrator with `PipelineConfig` (paths) and `RunConfig` (behavior flags). Run by editing the `if __name__` block. |
| `src/text_splitter.py` | Splits markdown reports by lines (default 30-line chunks, 5-line overlap). Outputs JSON with `metainfo` (sha1, company_name) + `content.chunks`. |
| `src/ingestion.py` | `VectorDBIngestor` uses DashScope embeddings + FAISS (inner product). `BM25Ingestor` builds BM25 indices. Both key on `sha1` for filenames. |
| `src/retrieval.py` | `VectorRetriever` (semantic), `BM25Retriever`, `HybridRetriever` (vector + LLM rerank). Retrieve by `company_name` matching. |
| `src/reranking.py` | `LLMReranker` scores chunks via LLM (supports dashscope/openai). `JinaReranker` uses Jina API. |
| `src/questions_processing.py` | `QuestionsProcessor` — company extraction from question text, retrieval dispatch, RAG context assembly, LLM answer, page reference validation (2-8 pages). Supports single-company and multi-company comparative questions. Thread-parallel batch processing. |
| `src/api_requests.py` | Unified API layer: `APIProcessor` routes to provider-specific processors (`BaseDashscopeProcessor`, `BaseOpenaiProcessor`, `BaseGeminiProcessor`, `BaseIBMAPIProcessor`). DashScope is the default and primary provider. |
| `src/prompts.py` | All LLM prompts + Pydantic response schemas. Supports question types: `name`, `number`, `boolean`, `names`, `string`, `comparative`. Each has strict field validation rules. Also contains `RerankingPrompt` and `AnswerSchemaFixPrompt` for JSON repair. |
| `src/pdf_mineru.py` | Calls MinerU cloud API (extract/task endpoint) to convert PDF → markdown. **Contains a hardcoded API key** — do not commit this. |
| `test_mineru.py` | Alternative: local MinerU CLI via `mineru.cli.client`. |
| `app_streamlit.py` | Streamlit web UI for single-question Q&A. Calls `Pipeline.answer_single_question()`. |

### Key design decisions

- **FAISS file naming**: Uses SHA1 hashes (from `subset.csv`) as filenames to avoid Windows Chinese-path issues with FAISS. Temporary copy to `_faiss_temp/` dir for reading.
- **Embedding provider**: Default is DashScope (`text-embedding-v1`), max batch size 25. Chunks truncated to 2048 chars before embedding.
- **LLM provider**: Default is DashScope Qwen-Turbo. Rate limit is 500 QPM / 500K TPM.
- **Parent document retrieval**: Optional mode where retrieval returns full pages instead of chunks.
- **LLM reranking**: Vector retrieval gets top-N candidates, LLM scores each for relevance (0-1), final score = weighted average of vector distance and LLM score.
- **Comparative questions**: Multi-company questions are decomposed into per-company sub-questions via LLM, answered in parallel, then aggregated.
- **Page validation**: LLM-claimed pages are cross-checked against actual retrieval results. Hallucinated pages are removed; minimum 2 pages enforced.

### Data structure

- `data/stock_data/` — current working dataset (SMIC reports + questions)
  - `pdf_reports/` — source PDFs
  - `questions.json` — question list with `text` and `kind` fields
  - `subset.csv` — maps `sha1` → `file_name` → `company_name`
  - `debug_data/03_reports_markdown/` — intermediate markdown from PDF parsing
  - `databases/chunked_reports/` — chunked JSON files
  - `databases/vector_dbs/` — FAISS `.faiss` files named by SHA1
- `output/` — MinerU extraction output (auto-generated per PDF)

### RunConfig presets (in `pipeline.py`)

- `max_config`: qwen-turbo, parent document retrieval, LLM reranking, parallel=4
- `pdr_config`: gpt-4o, parent document retrieval, parallel=20
- `base_config`: gpt-4o-mini, vector DB only, parallel=10