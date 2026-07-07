# Clinical Decision Support RAG

An evidence-based **Clinical Decision Support System** built with an advanced Retrieval-Augmented
Generation (RAG) pipeline. It retrieves evidence from biomedical sources (PubMed, WHO, CDC,
clinical guidelines) and generates structured, cited reports for healthcare professionals.

> **This system does not diagnose patients and does not replace professional medical judgment.**
> Every generated report carries this disclaimer and is intended purely as decision-support
> evidence retrieval, never as a standalone clinical decision.

---

## Architecture

```
                          ┌─────────────────────────┐
                          │        FastAPI           │
                          │  /index /analyze /health  │
                          │      /sources /metrics    │
                          └────────────┬─────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
         IndexingService       RetrievalService        ReportService
                 │                     │                     │
     ┌───────────┼──────────┐   RetrieverFactory      GroqService (LLM)
     │           │          │   (similarity, multi-      │
  Loaders   Chunking   VectorStore  query, parent,   ResponseFormatter
(PubMed/WHO/  (Recursive  (FAISS)   compression,          │
 CDC/Guide)  CharSplitter)          metadata, hybrid) CitationService
                                       │
                              EmbeddingService
                            (BAAI/bge-small-en-v1.5)
```

Each layer depends only on interfaces one layer below it (Clean Architecture / SOLID). Business
logic lives in `services/`, never in `api/routes/`.

## Folder Structure

| Folder | Purpose |
|---|---|
| `app/api/` | FastAPI routes, DI wiring, middleware |
| `app/core/` | Config, logging, constants, exceptions |
| `app/loaders/` | Source-specific document loaders (PubMed, WHO, CDC, generic guidelines) |
| `app/embeddings/` | Embedding model service + factory |
| `app/vectorstore/` | FAISS index lifecycle management |
| `app/retrievers/` | Similarity, multi-query, parent-document, compression, metadata, hybrid retrievers |
| `app/prompts/` | Prompt templates + assembly for structured report generation |
| `app/llm/` | Groq client (streaming/retry/timeout) + response parsing |
| `app/services/` | Orchestration: indexing, retrieval, report generation, citations |
| `app/models/` | Pydantic request/response/document schemas |
| `app/evaluation/` | Ragas-based retrieval + generation evaluation, benchmark runner |
| `scripts/` | CLI entrypoints for indexing, evaluation, ingestion |
| `tests/` | Pytest suite (offline, using fake embeddings + mocked services) |

## Installation

```bash
git clone <this-repo>
cd clinical-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY
```

## Running Locally

```bash
# 1. Add some guideline documents (or use the PubMed loader with --query-terms)
mkdir -p data/guidelines/general
cp your_guideline.pdf data/guidelines/general/

# 2. Build the index
python scripts/build_index.py --sources guideline --rebuild

# 3. Start the API
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive OpenAPI documentation.

### Example: analyze a clinical question

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Evidence for anticoagulation in AFib with CHA2DS2-VASc score of 2", "retriever_type": "multi_query", "top_k": 6}'
```

### Example: agentic, memory-aware chat

`/agent/chat` runs each query through a LangGraph workflow instead of a fixed
retrieve-then-generate call. Reuse the same `conversation_id` across turns for memory:

```bash
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "session-1", "query": "What treats chest indrawing pneumonia in children?"}'

curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "session-1", "query": "And what about infants under 2 months?"}'
```

On each call, the agent:
1. **Routes** the question — an LLM call decides the best retriever strategy, whether a live
   PubMed search is warranted, and resolves follow-up phrasing ("what about infants...") into a
   standalone question using the conversation's history.
2. **Optionally searches PubMed live** and folds new results into the FAISS index before retrieving.
3. **Retrieves** using the chosen strategy, then **reranks** the candidates with a cross-encoder
   for higher precision than embedding similarity alone.
4. **Checks sufficiency** — if evidence is too thin, it broadens the query and retries (bounded by
   `AGENT_MAX_ITERATIONS`).
5. **Generates** the structured report and **records the turn** in memory for future follow-ups.

Memory is per-`conversation_id`, backed by LangGraph's checkpointer (in-memory by default —
swapping to a persistent backend for production is a one-line change in `clinical_agent.py`).

## Streamlit UI

A Streamlit front-end (`streamlit_app.py`) runs in the same process as the backend services —
no separate `uvicorn` server needed alongside it:

```bash
streamlit run streamlit_app.py
```

It has four tabs:

1. **Index Documents** — build/rebuild the FAISS index from any combination of sources. A WHO
   guideline PDF (*Revised WHO classification and treatment of childhood pneumonia at health
   facilities*, 2014) is bundled at `data/guidelines/general/who_childhood_pneumonia_2014.pdf` so
   there's something to index immediately.
2. **Analyze** — single-query retrieve-then-generate, same as `POST /analyze`.
3. **Agent Chat** — the agentic, memory-aware LangGraph workflow, same as `POST /agent/chat`, as a
   real chat interface. Click "Start a new conversation" to reset memory.
4. **Evaluation** — runs the Ragas + retrieval-metric benchmark against
   `app/evaluation/datasets/sample_dataset.json` (or any dataset path you provide) and displays
   the scores as tables and a bar chart, with per-question detail.

The first load is slow (downloading/loading the embedding and cross-encoder models); subsequent
interactions are fast since `st.cache_resource` keeps them in memory for the life of the process.

## Docker Deployment

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. Persisted FAISS indices live under
`./data/vector_db` on the host, mounted into the container.

## Evaluation

```bash
python scripts/evaluate.py --dataset app/evaluation/datasets/sample_dataset.json --retriever multi_query
```

Computes Recall@K, Precision@K, MRR, nDCG (retrieval) and Faithfulness, Context Precision/Recall,
Answer Relevancy, and an approximate hallucination rate (generation, via Ragas).

## Testing

```bash
pytest --cov=app tests/
```

Tests use a fake, hash-based embeddings implementation and mocked LLM/report services so the
suite runs fully offline and fast — no GPU, network, or API key required.

## Future Improvements

Implemented in this version:
- ~~Cross-encoder reranking stage after initial retrieval~~ — done, see `retrievers/reranker.py`
- ~~Conversation memory for multi-turn clinical Q&A~~ — done, via LangGraph checkpointer
- ~~LangGraph-based multi-step reasoning workflows~~ — done, see `agents/clinical_agent.py`

Still open:
- Full production-grade hybrid BM25 + dense fusion weight tuning (the mechanism exists in
  `hybrid_retriever.py`; weights are currently static)
- Redis-backed checkpointer/caching for agent memory and rate limiting (currently in-memory,
  swappable in one line in `clinical_agent.py`)
- AWS deployment (ECS/Fargate) with S3-backed FAISS index persistence
- Semantic chunking (interface already isolated in `IndexingService._split_documents`)
- Prometheus metrics exporter in place of the current in-memory `/metrics`

## Screenshots

_Add screenshots of the `/docs` OpenAPI UI and example `/analyze` responses here._

`docs/screenshot-openapi.png`
`docs/screenshot-analyze-response.png`
