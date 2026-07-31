"""Streamlit Thin Client for the Hybrid RAG Evaluation Pipeline."""

import os
import json
import streamlit as st
import httpx

# Configure page settings
st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API endpoint URL from environment or default
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom styling for premium look
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: #ffffff;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State values
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 5
if "rerank_top_k" not in st.session_state:
    st.session_state["rerank_top_k"] = 5
if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = "openai"

# --- Sidebar Configuration ---
st.sidebar.image(
    "https://img.icons8.com/nolan/128/artificial-intelligence.png",
    width=80,
)
st.sidebar.title("Configuration")

# Health Status Indicator
try:
    health_response = httpx.get(f"{API_URL}/health", timeout=2.0)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("ok"):
            st.sidebar.success("🟢 API Status: Connected")
        else:
            st.sidebar.warning("🟡 API Status: Degraded")
    else:
        st.sidebar.error("🔴 API Status: Error")
except Exception:
    st.sidebar.error("🔴 API Status: Offline")

st.sidebar.divider()

# Sidebar inputs with session state persistence
st.session_state["top_k"] = st.sidebar.slider(
    "Top K Retrieved Chunks",
    min_value=1,
    max_value=20,
    value=st.session_state["top_k"],
)

st.session_state["rerank_top_k"] = st.sidebar.slider(
    "Rerank Top K (Frontend Limit)",
    min_value=1,
    max_value=10,
    value=st.session_state["rerank_top_k"],
)

st.session_state["llm_provider"] = st.sidebar.selectbox(
    "LLM Provider Backend",
    options=["openai", "ollama", "litellm"],
    index=["openai", "ollama", "litellm"].index(st.session_state["llm_provider"]),
)

st.sidebar.info(
    "Note: LLM Provider change reflects frontend configuration metadata; the actual API active backend is governed by server-side configuration."
)

# --- Main Page Layout ---
st.title("🤖 Hybrid RAG Assistant")
st.caption("Cross-Encoder Reranking & Ragas Evaluation Pipeline")

# Stream / Standard Response Toggle
stream_response = st.toggle("Enable Stream Mode", value=True)

query_input = st.text_area(
    "Ask a question to your documents:",
    placeholder="What is RRF and how does it merge sparse and dense retrieval ranks?",
    height=100,
)

if st.button("Generate Answer"):
    if not query_input.strip():
        st.warning("Please enter a valid query.")
    else:
        payload = {"query": query_input, "top_k": st.session_state["top_k"]}

        if stream_response:
            # Streaming Response Flow
            st.subheader("Answer Stream")
            answer_box = st.empty()
            full_response = ""

            try:
                with st.spinner("Retrieving, Reranking & Generating..."):
                    with httpx.stream(
                        "POST",
                        f"{API_URL}/query/stream",
                        json=payload,
                        timeout=30.0,
                    ) as r:
                        if r.status_code != 200:
                            st.error(f"Failed to initiate stream: HTTP {r.status_code}")
                        else:
                            for chunk in r.iter_raw():
                                chunk_str = chunk.decode("utf-8", errors="replace")
                                if "[ERROR]" in chunk_str:
                                    st.error(chunk_str)
                                    break
                                full_response += chunk_str
                                answer_box.markdown(full_response + "▌")
                answer_box.markdown(full_response)
            except Exception as e:
                st.error(f"Streaming error occurred: {e}")
        else:
            # Standard Non-streaming Response Flow
            with st.spinner("Processing full RAG pipeline..."):
                try:
                    response = httpx.post(
                        f"{API_URL}/query",
                        json=payload,
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.subheader("Answer")
                        st.write(data["answer"])

                        # Display Latency
                        st.caption(f"Pipeline Latency: {data['latency_ms']:.2f} ms")

                        # Display context chunks
                        st.divider()
                        st.subheader("Retrieved & Reranked Context Chunks")

                        chunks = data.get("context_chunks", [])
                        scores = data.get("reranked_scores", [])

                        # Apply client-side rerank limit if needed
                        limit = st.session_state["rerank_top_k"]
                        chunks = chunks[:limit]
                        scores = scores[:limit]

                        for idx, (chunk, score) in enumerate(zip(chunks, scores)):
                            chunk_data = chunk.get("chunk", {})
                            text = chunk_data.get("text", "")
                            doc_id = chunk_data.get("document_id", "Unknown")
                            chunk_idx = chunk_data.get("chunk_index", 0)
                            meta = chunk_data.get("metadata", {})

                            with st.expander(
                                f"Chunk {idx+1}: Doc ID '{doc_id}' (Rerank Score: {score:.4f})"
                            ):
                                st.write(text)
                                st.json(
                                    {
                                        "document_id": doc_id,
                                        "chunk_index": chunk_idx,
                                        "metadata": meta,
                                    }
                                )
                    else:
                        st.error(f"Query API failed: {response.text}")
                except Exception as e:
                    st.error(f"Error calling API: {e}")

st.divider()

# --- Evaluation Panel ---
with st.expander("📊 Evaluation Panel", expanded=False):
    st.subheader("Run Ragas Evaluation")
    st.write(
        "Upload a JSON file containing evaluation QA pairs. Format should be a list of QA pairs."
    )
    st.code(
        """[
  {"question": "What is RAG?", "ground_truth": "Retrieval-Augmented Generation is..."},
  {"question": "How does BM25 work?", "ground_truth": "BM25 is a term weighting method..."}
]""",
        language="json",
    )

    uploaded_file = st.file_uploader(
        "Choose a JSON evaluation dataset file",
        type=["json"],
    )

    if st.button("Start Ragas Evaluation"):
        if uploaded_file is None:
            st.warning("Please upload a JSON dataset first.")
        else:
            try:
                dataset = json.load(uploaded_file)
                if not isinstance(dataset, list):
                    st.error("Evaluation dataset must be a JSON array of QA pairs.")
                else:
                    payload = {"qa_dataset": dataset}
                    with st.spinner("Running Ragas evaluation metrics..."):
                        response = httpx.post(
                            f"{API_URL}/evaluate",
                            json=payload,
                            timeout=180.0,  # Evaluation can take longer
                        )

                        if response.status_code == 200:
                            eval_report = response.json().get("report", {})
                            st.success("Evaluation Completed Successfully!")

                            # Metric Cards in Columns
                            st.markdown("### Aggregate Scores")
                            aggregates = eval_report.get("aggregate", {})
                            cols = st.columns(max(1, len(aggregates)))

                            for col, (metric, val) in zip(cols, aggregates.items()):
                                with col:
                                    metric_display = metric.replace("_", " ").title()
                                    st.markdown(
                                        f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{val:.4f}</div>
                                            <div class="metric-label">{metric_display}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                            # Detailed table / results
                            st.markdown("### Per-Question Metrics")
                            questions = eval_report.get("per_question", [])
                            if questions:
                                rows = []
                                for q in questions:
                                    scores = q.get("scores", {})
                                    row = {
                                        "Question": q.get("question"),
                                        "Answer": q.get("answer"),
                                        **scores,
                                    }
                                    rows.append(row)
                                st.dataframe(rows)

                            # Error reporting
                            errors = eval_report.get("errors", [])
                            if errors:
                                st.markdown("### Failures / Warnings")
                                for err in errors:
                                    st.warning(
                                        f"Question: {err.get('question')} | Error: {err.get('error_message')}"
                                    )

                        else:
                            st.error(f"Evaluation API failed: {response.text}")
            except Exception as e:
                st.error(f"Error running evaluation: {e}")
