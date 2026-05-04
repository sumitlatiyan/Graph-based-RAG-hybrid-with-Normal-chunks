"""
Graph RAG — PDF Knowledge Q&A
Streamlit app with query caching: repeated questions cost 0 tokens.
"""

import os
import re
import json
import pickle
import hashlib
from pathlib import Path

import numpy as np
import networkx as nx
import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from Constants import GROQ_API_KEY

# ── Model ──────────────────────────────────────────────────────────────────────
model = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
)

# ── Config ─────────────────────────────────────────────────────────────────────
STORE_DIR        = Path("graph_store")
STORE_DIR.mkdir(exist_ok=True)
INDEX_FILE       = STORE_DIR / "index.json"
QUERY_CACHE_FILE = STORE_DIR / "query_cache.json"   # ← NEW
CHUNK_SIZE       = 400
OVERLAP          = 50
TOP_K            = 5


# ══════════════════════════════════════════════════════════════════════════════
# Embedder
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ══════════════════════════════════════════════════════════════════════════════
# Token counting  (no tiktoken needed — character-based ~4 chars/token)
# ══════════════════════════════════════════════════════════════════════════════

def count_tokens(text: str) -> int:
    """Approximate token count using the ~4 chars/token heuristic for LLMs."""
    return max(1, len(text) // 4)

def estimate_query_tokens(prompt: str, answer: str) -> dict:
    """Return a breakdown of input/output/total token estimates for one query."""
    input_tok  = count_tokens(prompt)
    output_tok = count_tokens(answer)
    return {
        "input_tokens":  input_tok,
        "output_tokens": output_tok,
        "total_tokens":  input_tok + output_tok,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Storage — document index
# ══════════════════════════════════════════════════════════════════════════════

def load_index() -> dict:
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return {}

def save_index(idx: dict):
    INDEX_FILE.write_text(json.dumps(idx, indent=2))

def pdf_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def save_graph_data(doc_id: str, data: dict):
    with open(STORE_DIR / f"{doc_id}.pkl", "wb") as f:
        pickle.dump(data, f)

def load_graph_data(doc_id: str) -> dict:
    with open(STORE_DIR / f"{doc_id}.pkl", "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Storage — query cache  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def load_query_cache() -> dict:
    """Load the on-disk query → answer cache."""
    if QUERY_CACHE_FILE.exists():
        return json.loads(QUERY_CACHE_FILE.read_text())
    return {}

def save_query_cache(cache: dict):
    """Persist the query cache to disk."""
    QUERY_CACHE_FILE.write_text(json.dumps(cache, indent=2))

def make_cache_key(doc_id: str, question: str) -> str:
    """Stable, lowercase cache key scoped to a specific document."""
    return f"{doc_id}::{question.strip().lower()}"


# ══════════════════════════════════════════════════════════════════════════════
# PDF → Chunks
# ══════════════════════════════════════════════════════════════════════════════

def extract_chunks(uploaded_file) -> list:
    reader     = PdfReader(uploaded_file)
    chunks     = []
    buf        = []
    page_start = 1

    for page_num, page in enumerate(reader.pages, start=1):
        content = page.extract_text()
        if not content:
            continue
        words = content.split()
        for w in words:
            buf.append(w)
            if len(buf) == 1:
                page_start = page_num
            if len(buf) >= CHUNK_SIZE:
                chunks.append({"text": " ".join(buf), "page": page_start})
                buf = buf[-OVERLAP:]
                page_start = page_num

    if buf:
        chunks.append({"text": " ".join(buf), "page": page_start})

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Entity & Relation Extraction  (Groq / Llama)
# ══════════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """Extract entities and relationships from these text chunks.

{text}

Reply ONLY with valid JSON (no markdown, no explanation):
{{
  "entities": [
    {{"name": "...", "type": "Person|Organization|Concept|Place|Event|Technology|Other", "description": "short phrase"}}
  ],
  "relations": [
    {{"source": "Entity A", "target": "Entity B", "relation": "verb phrase", "chunk": <integer from [Chunk N] label>}}
  ]
}}
Rules: 3-8 entities per chunk, meaningful relations only, keep entity names consistent."""


def extract_graph(chunks: list, progress_bar):
    entities   = {}
    relations  = []
    chunk_ents = {}
    counter    = [0]

    def upsert(name, etype="Other", desc=""):
        key = name.lower().strip()
        if key not in entities:
            eid = f"e{counter[0]}"
            counter[0] += 1
            entities[key] = {"id": eid, "name": name, "type": etype, "description": desc}
        return entities[key]["id"]

    batch_size = 4
    total      = min(len(chunks), 60)

    for i in range(0, total, batch_size):
        sub = chunks[i : i + batch_size]
        progress_bar.progress(
            int((i / total) * 80),
            text=f"Extracting entities — chunks {i}–{i+len(sub)-1}…",
        )

        combined = "\n\n---\n\n".join(
            f"[Chunk {i+j}]\n{c['text'][:700]}" for j, c in enumerate(sub)
        )

        try:
            response = model.invoke(EXTRACT_PROMPT.format(text=combined))
            raw      = response.content.strip()
            raw      = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
            data     = json.loads(raw)

            for e in data.get("entities", []):
                upsert(e.get("name", "?"), e.get("type", "Other"), e.get("description", ""))

            for r in data.get("relations", []):
                src     = upsert(r.get("source", "?"))
                tgt     = upsert(r.get("target", "?"))
                abs_idx = i + int(r.get("chunk", 0))
                relations.append({
                    "source":    src,
                    "target":    tgt,
                    "relation":  r.get("relation", "related to"),
                    "chunk_idx": abs_idx,
                })
                for eid in (src, tgt):
                    chunk_ents.setdefault(str(abs_idx), [])
                    if eid not in chunk_ents[str(abs_idx)]:
                        chunk_ents[str(abs_idx)].append(eid)

        except Exception as ex:
            st.warning(f"Batch {i} error: {ex}")

    return list(entities.values()), relations, chunk_ents


# ══════════════════════════════════════════════════════════════════════════════
# Query  (with caching)
# ══════════════════════════════════════════════════════════════════════════════

ANSWER_PROMPT = """You are a helpful assistant. Answer the question using the knowledge graph and document excerpts below. Be concise and accurate.

Question: {question}

=== Knowledge Graph ===
{graph_ctx}

=== Document Excerpts ===
{chunk_ctx}

Answer:"""


def query_graph_rag(question: str, graph_data: dict, doc_id: str) -> dict:
    """
    Answer a question using the knowledge graph.

    Results are cached on disk keyed by (doc_id, question).
    Repeated identical questions return instantly with 0 tokens spent.
    """
    # ── 1. Check cache ────────────────────────────────────────────────────────
    cache     = load_query_cache()
    cache_key = make_cache_key(doc_id, question)

    if cache_key in cache:
        cached = cache[cache_key]
        cached["from_cache"] = True          # flag so UI can show the badge
        return cached

    # ── 2. Build graph in memory ──────────────────────────────────────────────
    embedder   = load_embedder()
    chunks     = graph_data["chunks"]
    embeddings = np.array(graph_data["embeddings"])
    entities   = graph_data["entities"]
    relations  = graph_data["relations"]
    chunk_ents = graph_data["chunk_entities"]

    G = nx.DiGraph()
    for e in entities:
        G.add_node(e["id"], **e)
    for r in relations:
        G.add_edge(r["source"], r["target"], relation=r["relation"])

    # ── 3. Semantic retrieval ─────────────────────────────────────────────────
    q_emb    = embedder.encode([question])
    sims     = cosine_similarity(q_emb, embeddings)[0]
    top_idxs = list(map(int, np.argsort(sims)[::-1][:TOP_K]))

    # ── 4. Seed entities → 1-hop expansion ───────────────────────────────────
    seed_ids = set()
    for idx in top_idxs:
        for eid in chunk_ents.get(str(idx), []):
            seed_ids.add(eid)

    expanded = set(seed_ids)
    for eid in list(seed_ids):
        if eid in G:
            expanded.update(G.predecessors(eid))
            expanded.update(G.successors(eid))

    # ── 5. Build context strings ──────────────────────────────────────────────
    chunk_ctx = "\n\n".join(
        f"[Page {chunks[i]['page']}]\n{chunks[i]['text']}"
        for i in top_idxs if i < len(chunks)
    )

    graph_lines = []
    for eid in list(expanded)[:25]:
        if eid not in G:
            continue
        node  = G.nodes[eid]
        out_r = [
            f"{G.nodes[v].get('name', v)} ({d['relation']})"
            for _, v, d in G.out_edges(eid, data=True)
        ][:4]
        in_r  = [
            f"{G.nodes[u].get('name', u)} ({d['relation']})"
            for u, _, d in G.in_edges(eid, data=True)
        ][:2]
        line = f"• {node.get('name', eid)} [{node.get('type', '')}]: {node.get('description', '')}"
        if out_r: line += f" → {', '.join(out_r)}"
        if in_r:  line += f" ← {', '.join(in_r)}"
        graph_lines.append(line)

    graph_ctx = "\n".join(graph_lines) or "No graph context found."

    # ── 6. Build full prompt & count input tokens ─────────────────────────────
    full_prompt = ANSWER_PROMPT.format(
        question=question,
        graph_ctx=graph_ctx,
        chunk_ctx=chunk_ctx,
    )
    input_tok_est = count_tokens(full_prompt)

    # ── 7. Call LLM ───────────────────────────────────────────────────────────
    response = model.invoke(full_prompt)

    output_tok_est = count_tokens(response.content)

    result = {
        "answer":        response.content,
        "chunks_used":   top_idxs,
        "entities_used": list(expanded)[:25],
        "graph_nodes":   G.number_of_nodes(),
        "graph_edges":   G.number_of_edges(),
        "from_cache":    False,
        "tokens": {
            "input":  input_tok_est,
            "output": output_tok_est,
            "total":  input_tok_est + output_tok_est,
        },
    }

    # ── 8. Save to cache ──────────────────────────────────────────────────────
    cache[cache_key] = result
    save_query_cache(cache)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Graph RAG", page_icon="🕸️")
    st.title("🕸️ Graph RAG — PDF Knowledge Q&A")
    st.caption(
        "Upload a PDF → builds a knowledge graph → ask questions. "
        "Graph and answers are cached — repeated questions cost **0 tokens**."
    )

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Add it to Constants.py or as an env variable.")
        st.stop()

    index = load_index()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📚 Saved Documents")
        if index:
            for doc_id, meta in index.items():
                st.markdown(f"**{meta['filename']}**")
                st.caption(
                    f"{meta['chunks']} chunks · "
                    f"{meta['nodes']} nodes · "
                    f"{meta['edges']} edges"
                )
        else:
            st.info("No documents yet.")

        st.divider()

        # ── Cache stats ───────────────────────────────────────────────────────
        st.header("⚡ Query Cache")
        cache = load_query_cache()
        total_cached = len(cache)
        doc_cached   = 0

        # Count entries that belong to the currently loaded doc (if any)
        # We'll update this after the file uploader section runs
        st.session_state.setdefault("current_doc_id", None)
        if st.session_state.current_doc_id:
            doc_cached = sum(
                1 for k in cache
                if k.startswith(st.session_state.current_doc_id + "::")
            )
            st.metric("Cached answers (this doc)", doc_cached)

        st.metric("Total cached answers", total_cached)

        if total_cached > 0 and st.button("🗑️ Clear all cached answers"):
            save_query_cache({})
            st.success("Cache cleared.")
            st.rerun()

        st.divider()

        # ── Session token usage ───────────────────────────────────────────────
        st.header("🔢 Token Usage (this session)")
        st.session_state.setdefault("session_input_tokens",  0)
        st.session_state.setdefault("session_output_tokens", 0)
        st.session_state.setdefault("session_saved_tokens",  0)

        t_in   = st.session_state.session_input_tokens
        t_out  = st.session_state.session_output_tokens
        t_save = st.session_state.session_saved_tokens

        st.metric("Input tokens used",   f"{t_in:,}")
        st.metric("Output tokens used",  f"{t_out:,}")
        st.metric("Total tokens used",   f"{t_in + t_out:,}")
        st.metric("⚡ Tokens saved (cache)", f"{t_save:,}")

        if (t_in + t_out + t_save) > 0:
            pct = int(t_save / (t_in + t_out + t_save) * 100)
            st.progress(pct, text=f"{pct}% requests served from cache")

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        doc_id    = pdf_hash(pdf_bytes)
        uploaded_file.seek(0)

        # Store doc_id so sidebar can show per-doc cache count
        st.session_state.current_doc_id = doc_id

        if doc_id in index:
            st.success(f"✅ **{uploaded_file.name}** already processed — loaded from cache.")
            graph_data = load_graph_data(doc_id)
        else:
            st.info(f"New document: **{uploaded_file.name}**. Building knowledge graph…")
            progress = st.progress(0, text="Extracting text…")

            chunks = extract_chunks(uploaded_file)
            st.write(f"📄 {len(chunks)} chunks extracted.")
            progress.progress(10, text="Embedding chunks…")

            embedder   = load_embedder()
            embeddings = embedder.encode(
                [c["text"] for c in chunks], show_progress_bar=False
            )
            progress.progress(20, text="Extracting entities & relations…")

            entities, relations, chunk_ents = extract_graph(chunks, progress)
            progress.progress(85, text="Saving…")

            graph_data = {
                "chunks":         chunks,
                "embeddings":     embeddings.tolist(),
                "entities":       entities,
                "relations":      relations,
                "chunk_entities": chunk_ents,
            }
            save_graph_data(doc_id, graph_data)

            G_tmp = nx.DiGraph()
            for e in entities:
                G_tmp.add_node(e["id"])
            for r in relations:
                G_tmp.add_edge(r["source"], r["target"])

            index[doc_id] = {
                "filename": uploaded_file.name,
                "chunks":   len(chunks),
                "nodes":    G_tmp.number_of_nodes(),
                "edges":    G_tmp.number_of_edges(),
            }
            save_index(index)
            progress.progress(100, text="Done!")
            st.success("✅ Knowledge graph built and saved!")

        # ── Metrics ───────────────────────────────────────────────────────────
        meta = index[doc_id]
        col1, col2, col3 = st.columns(3)
        col1.metric("Chunks",      meta["chunks"])
        col2.metric("Graph Nodes", meta["nodes"])
        col3.metric("Graph Edges", meta["edges"])

        with st.expander("🔍 Entity preview (top 20)"):
            for e in graph_data["entities"][:20]:
                st.markdown(f"- **{e['name']}** `{e['type']}` — {e['description']}")

        st.divider()
        st.subheader("💬 Ask a Question")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    tok = msg.get("tokens", {})
                    if msg.get("from_cache"):
                        st.caption("⚡ Served from cache — **0 tokens used**")
                    elif tok:
                        st.caption(
                            f"~{tok.get('input', 0):,} in · "
                            f"~{tok.get('output', 0):,} out · "
                            f"**~{tok.get('total', 0):,} total tokens**"
                        )

        question = st.chat_input("Ask anything about the document…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching graph…"):
                    result = query_graph_rag(question, graph_data, doc_id)

                st.markdown(result["answer"])

                tok = result.get("tokens", {})

                # ── Cache / token badge ───────────────────────────────────────
                if result["from_cache"]:
                    st.caption("⚡ Served from cache — **0 tokens used**")
                    # Accumulate saved tokens using the stored estimate
                    st.session_state.session_saved_tokens += tok.get("total", 0)
                else:
                    st.caption(
                        f"🔄 Fresh answer — "
                        f"~{tok.get('input', 0):,} in · "
                        f"~{tok.get('output', 0):,} out · "
                        f"**~{tok.get('total', 0):,} total tokens**"
                    )
                    st.session_state.session_input_tokens  += tok.get("input",  0)
                    st.session_state.session_output_tokens += tok.get("output", 0)

                with st.expander("🔎 Retrieval details"):
                    st.write(f"Chunks used: {result['chunks_used']}")
                    st.write(f"Entities expanded: {len(result['entities_used'])}")
                    st.write(
                        f"Graph: {result['graph_nodes']} nodes, "
                        f"{result['graph_edges']} edges"
                    )
                    if not result["from_cache"] and tok:
                        st.write("**Token breakdown (approximate):**")
                        st.write(f"  • Input  (prompt): ~{tok['input']:,} tokens")
                        st.write(f"  • Output (answer): ~{tok['output']:,} tokens")
                        st.write(f"  • Total:           ~{tok['total']:,} tokens")

            st.session_state.messages.append({
                "role":       "assistant",
                "content":    result["answer"],
                "from_cache": result["from_cache"],
                "tokens":     result.get("tokens", {}),
            })

    else:
        st.info("👆 Upload a PDF to get started.")


if __name__ == "__main__":
    main()
