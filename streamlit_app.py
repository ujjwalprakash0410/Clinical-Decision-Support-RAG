"""Streamlit front-end for the Clinical Decision Support RAG system.

This app runs in the SAME process as the backend services (it imports
directly from `app.*`, exactly like `scripts/build_index.py` and
`scripts/evaluate.py` do) rather than calling a separately-running
FastAPI server over HTTP. That means:

  streamlit run streamlit_app.py

is enough on its own — no `uvicorn main:app` needed alongside it.

Tabs:
  1. Index Documents — ingest sources and (re)build the FAISS index
  2. Analyze          — single-shot retrieve-then-generate report
  3. Agent Chat        — the agentic, memory-aware LangGraph workflow
  4. Evaluation        — run the Ragas + retrieval-metric benchmark
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.api.dependencies.di import (  # noqa: E402
    get_agent_service,
    get_chat_llm,
    get_indexing_service,
    get_report_service,
    get_retrieval_service,
    get_vector_store,
)
from app.core.constants import RetrieverType  # noqa: E402
from app.core.exceptions import ClinicalRAGError  # noqa: E402
from app.embeddings.embedding_factory import get_embedding_service  # noqa: E402
from app.evaluation.benchmark import run_benchmark  # noqa: E402
from app.evaluation.ragas_runner import load_benchmark_dataset, run_full_evaluation  # noqa: E402

st.set_page_config(page_title="Clinical Decision Support RAG", page_icon="\U0001FA7A", layout="wide")

DEFAULT_DATASET_PATH = "app/evaluation/datasets/sample_dataset.json"

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


@st.cache_resource(show_spinner="Loading models (embeddings + LLM client) — first call only...")
def load_services():
    """Load every heavyweight singleton once per Streamlit process."""
    return {
        "indexing_service": get_indexing_service(),
        "report_service": get_report_service(),
        "retrieval_service": get_retrieval_service(),
        "agent_service": get_agent_service(),
        "vector_store": get_vector_store(),
        "chat_llm": get_chat_llm(),
        "embedding_service": get_embedding_service(),
    }


def render_report(report, meta: dict) -> None:
    """Render a ClinicalReport (and surrounding metadata) as Streamlit UI."""
    cols = st.columns(3)
    cols[0].metric("Retriever used", meta.get("retriever_used", "-"))
    cols[1].metric("Documents retrieved", meta.get("documents_retrieved", 0))
    cols[2].metric("Latency (ms)", f"{meta.get('latency_ms', 0):.0f}")

    st.warning(f"**Disclaimer:** {report.disclaimer}")

    st.subheader("Summary")
    st.write(report.summary)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Possible Conditions (differential, not a diagnosis)")
        st.write("\n".join(f"- {item}" for item in report.possible_conditions) or "None listed.")
        st.subheader("Suggested Diagnostic Tests")
        st.write("\n".join(f"- {item}" for item in report.suggested_diagnostic_tests) or "None listed.")
    with col_b:
        st.subheader("\U0001F6A9 Red Flag Symptoms")
        st.write("\n".join(f"- {item}" for item in report.red_flag_symptoms) or "None listed.")
        st.subheader("Clinical Guidelines Referenced")
        st.write("\n".join(f"- {item}" for item in report.clinical_guidelines) or "None listed.")

    st.subheader("Evidence Summary")
    st.write(report.evidence_summary)

    st.subheader("References")
    if report.references:
        ref_rows = [
            {"Label": r.label, "Title": r.title, "Source": r.source, "Year": r.publication_year, "URL": r.url or ""}
            for r in report.references
        ]
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No references attached.")

    col_c, col_d = st.columns(2)
    col_c.metric("Confidence", report.confidence)
    with col_d:
        st.write("**Limitations**")
        st.write(report.limitations)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("\U0001FA7A Clinical Decision Support RAG")
st.caption(
    "Evidence-based clinical decision support. This system does not diagnose patients and does "
    "not replace professional medical judgment."
)

services = load_services()

tab_index, tab_analyze, tab_agent, tab_eval = st.tabs(
    ["\U0001F4E5 Index Documents", "\U0001F50D Analyze", "\U0001F916 Agent Chat", "\U0001F4CA Evaluation"]
)

# --------------------------------------------------------------------------
# Tab 1: Index Documents
# --------------------------------------------------------------------------

with tab_index:
    st.subheader("Ingest documents into the FAISS index")
    st.write(
        "A WHO guideline — *Revised WHO classification and treatment of childhood pneumonia at "
        "health facilities (2014)* — is bundled at `data/guidelines/general/` and matches the "
        "questions in the Evaluation tab. Drop additional `.txt`/`.pdf` files into "
        "`data/guidelines/{who,cdc,general}/` before indexing, or use the PubMed source for a "
        "live search."
    )

    source_options = ["guideline", "who", "cdc", "pubmed"]
    selected_sources = st.multiselect("Sources to ingest", source_options, default=["guideline"])
    rebuild = st.checkbox("Rebuild index from scratch (drops the existing index)", value=True)
    query_terms_raw = st.text_input(
        "PubMed search terms (comma-separated, only used if 'pubmed' is selected)",
        value="childhood pneumonia treatment",
    )

    if st.button("Build / Update Index", type="primary"):
        query_terms = [t.strip() for t in query_terms_raw.split(",") if t.strip()]
        with st.spinner("Loading, chunking, and indexing documents..."):
            try:
                chunk_count, sources_used = services["indexing_service"].index_sources(
                    source_names=selected_sources, query_terms=query_terms, rebuild=rebuild
                )
                if chunk_count > 0:
                    st.success(f"Indexed {chunk_count} chunks from sources: {sources_used}")
                else:
                    st.warning("No documents were found for the selected sources.")
            except ClinicalRAGError as exc:
                st.error(f"Indexing failed: {exc.message}")

    st.divider()
    st.metric("Documents currently in the index", services["vector_store"].document_count)

# --------------------------------------------------------------------------
# Tab 2: Analyze (single-shot)
# --------------------------------------------------------------------------

with tab_analyze:
    st.subheader("Single-query evidence retrieval + structured report")
    query = st.text_area(
        "Clinical question",
        placeholder="What is the recommended antibiotic treatment for fast breathing pneumonia in children?",
        height=100,
    )
    col1, col2 = st.columns(2)
    with col1:
        retriever_choice = st.selectbox(
            "Retriever strategy",
            [r.value for r in RetrieverType],
            index=[r.value for r in RetrieverType].index(RetrieverType.MULTI_QUERY.value),
        )
    with col2:
        top_k = st.slider("Top-K documents", min_value=1, max_value=15, value=6)

    if st.button("Analyze", type="primary", key="analyze_button"):
        if not query or len(query.strip()) < 3:
            st.error("Please enter a clinical question (at least 3 characters).")
        else:
            with st.spinner("Retrieving evidence and generating report..."):
                try:
                    response = services["report_service"].generate_report(
                        query=query, retriever_type=RetrieverType(retriever_choice), k=top_k
                    )
                    render_report(
                        response.report,
                        {
                            "retriever_used": response.retriever_used,
                            "documents_retrieved": response.documents_retrieved,
                            "latency_ms": response.latency_ms,
                        },
                    )
                except ClinicalRAGError as exc:
                    st.error(f"{type(exc).__name__}: {exc.message}")

# --------------------------------------------------------------------------
# Tab 3: Agent Chat (agentic, memory-aware)
# --------------------------------------------------------------------------

with tab_agent:
    st.subheader("Agentic chat (routes retriever choice, can search PubMed live, remembers context)")
    st.caption(f"Conversation ID: `{st.session_state.conversation_id}`")
    if st.button("Start a new conversation"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.chat_messages = []
        st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "report" in message:
                render_report(message["report"], message["meta"])
            else:
                st.write(message["content"])

    user_input = st.chat_input("Ask a clinical question, or a follow-up...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Routing, retrieving, and generating..."):
                try:
                    response = services["agent_service"].chat(
                        conversation_id=st.session_state.conversation_id, query=user_input
                    )
                    meta = {
                        "retriever_used": response.retriever_used,
                        "documents_retrieved": response.documents_retrieved,
                        "latency_ms": response.latency_ms,
                    }
                    render_report(response.report, meta)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "report": response.report, "meta": meta}
                    )
                except ClinicalRAGError as exc:
                    st.error(f"{type(exc).__name__}: {exc.message}")

# --------------------------------------------------------------------------
# Tab 4: Evaluation
# --------------------------------------------------------------------------

with tab_eval:
    st.subheader("Run the evaluation benchmark")
    st.write(
        "Runs each question in the dataset through the live pipeline, then scores it with "
        "**Ragas** (Faithfulness, Context Precision, Context Recall, Answer Relevancy) and "
        "custom retrieval metrics (Recall@K, Precision@K, MRR, nDCG)."
    )
    st.info(
        "\u26a0\ufe0f Retrieval metrics (Recall@K / Precision@K / MRR / nDCG) will show as 0. "
        "The ingestion pipeline does not currently assign a `chunk_id` to indexed chunks, so "
        "retrieved documents can't be matched against the dataset's `relevant_ids`. The Ragas "
        "generation metrics below are unaffected and are the meaningful scores here."
    )

    dataset_path = st.text_input("Dataset path", value=DEFAULT_DATASET_PATH)
    eval_retriever = st.selectbox(
        "Retriever strategy to benchmark",
        [r.value for r in RetrieverType],
        index=[r.value for r in RetrieverType].index(RetrieverType.MULTI_QUERY.value),
        key="eval_retriever",
    )
    eval_k = st.slider("Top-K for evaluation", min_value=1, max_value=15, value=5, key="eval_k")

    if st.button("Run Evaluation", type="primary"):
        try:
            dataset = load_benchmark_dataset(dataset_path)
        except ClinicalRAGError as exc:
            st.error(f"Could not load dataset: {exc.message}")
            dataset = None

        if dataset:
            progress = st.progress(0.0, text="Running questions through the live pipeline...")
            enriched = []
            for i, item in enumerate(dataset):
                enriched.extend(
                    run_benchmark(
                        [item],
                        retrieval_service=services["retrieval_service"],
                        report_service=services["report_service"],
                        retriever_type=RetrieverType(eval_retriever),
                        k=eval_k,
                    )
                )
                progress.progress((i + 1) / len(dataset), text=f"Processed {i + 1}/{len(dataset)} questions")

            with st.spinner("Scoring with Ragas..."):
                try:
                    results = run_full_evaluation(
                        enriched,
                        llm=services["chat_llm"],
                        embeddings=services["embedding_service"].langchain_embeddings,
                        retrieval_k=eval_k,
                    )
                except ClinicalRAGError as exc:
                    st.error(f"Evaluation failed: {exc.message}")
                    results = None

            if results:
                st.success("Evaluation complete.")

                st.subheader("Retrieval Metrics")
                st.dataframe(pd.DataFrame([results["retrieval"]]), use_container_width=True, hide_index=True)

                st.subheader("Generation Metrics (Ragas)")
                gen_df = pd.DataFrame([results["generation"]])
                st.dataframe(gen_df, use_container_width=True, hide_index=True)
                st.bar_chart(gen_df.T.rename(columns={0: "score"}))

                st.metric("Approximate Hallucination Rate", results["hallucination_rate"])

                st.subheader("Per-Question Results")
                detail_rows = [
                    {
                        "Question": item["question"],
                        "Generated Answer": item["answer"][:200],
                        "Documents Retrieved": len(item["contexts"]),
                        "Retrieval Latency (ms)": item["retrieval_latency_ms"],
                    }
                    for item in enriched
                ]
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

