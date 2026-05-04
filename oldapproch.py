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

model = ChatGroq(
    api_key=GROQ_API_KEY,        # ← pass directly here
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
)

# ── Config ─────────────────────────────────────────────────────────────────────
STORE_DIR  = Path("graph_store")
STORE_DIR.mkdir(exist_ok=True)
INDEX_FILE = STORE_DIR / "index.json"
CHUNK_SIZE = 400
OVERLAP    = 50
TOP_K      = 5

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ══════════════════════════════════════════════════════════════════════════════
# Storage
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
# PDF → Chunks  (PyPDF2, same as your ATS code)
# ══════════════════════════════════════════════════════════════════════════════

def extract_chunks(uploaded_file) -> list:
    reader = PdfReader(uploaded_file)
    chunks = []
    buf    = []
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
        progress_bar.progress(int((i / total) * 80), text=f"Extracting entities — chunks {i}–{i+len(sub)-1}…")

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
                relations.append({"source": src, "target": tgt,
                                   "relation": r.get("relation", "related to"),
                                   "chunk_idx": abs_idx})
                for eid in (src, tgt):
                    chunk_ents.setdefault(str(abs_idx), [])
                    if eid not in chunk_ents[str(abs_idx)]:
                        chunk_ents[str(abs_idx)].append(eid)

        except Exception as ex:
            st.warning(f"Batch {i} error: {ex}")

    return list(entities.values()), relations, chunk_ents


# ══════════════════════════════════════════════════════════════════════════════
# Query
# ══════════════════════════════════════════════════════════════════════════════

ANSWER_PROMPT = """You are a helpful assistant. Answer the question using the knowledge graph and document excerpts below. Be concise and accurate.

Question: {question}

=== Knowledge Graph ===
{graph_ctx}

=== Document Excerpts ===
{chunk_ctx}

Answer:"""


def query_graph_rag(question: str, graph_data: dict) -> dict:
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

    # Semantic retrieval
    q_emb    = embedder.encode([question])
    sims     = cosine_similarity(q_emb, embeddings)[0]
    top_idxs = list(map(int, np.argsort(sims)[::-1][:TOP_K]))

    # Seed entities → 1-hop expansion
    seed_ids = set()
    for idx in top_idxs:
        for eid in chunk_ents.get(str(idx), []):
            seed_ids.add(eid)

    expanded = set(seed_ids)
    for eid in list(seed_ids):
        if eid in G:
            expanded.update(G.predecessors(eid))
            expanded.update(G.successors(eid))

    # Context strings
    chunk_ctx = "\n\n".join(
        f"[Page {chunks[i]['page']}]\n{chunks[i]['text']}"
        for i in top_idxs if i < len(chunks)
    )

    graph_lines = []
    for eid in list(expanded)[:25]:
        if eid not in G:
            continue
        node  = G.nodes[eid]
        out_r = [f"{G.nodes[v].get('name', v)} ({d['relation']})"
                 for _, v, d in G.out_edges(eid, data=True)][:4]
        in_r  = [f"{G.nodes[u].get('name', u)} ({d['relation']})"
                 for u, _, d in G.in_edges(eid, data=True)][:2]
        line  = f"• {node.get('name', eid)} [{node.get('type', '')}]: {node.get('description', '')}"
        if out_r: line += f" → {', '.join(out_r)}"
        if in_r:  line += f" ← {', '.join(in_r)}"
        graph_lines.append(line)

    graph_ctx = "\n".join(graph_lines) or "No graph context found."

    response = model.invoke(
        ANSWER_PROMPT.format(question=question, graph_ctx=graph_ctx, chunk_ctx=chunk_ctx)
    )

    return {
        "answer":        response.content,
        "chunks_used":   top_idxs,
        "entities_used": list(expanded)[:25],
        "graph_nodes":   G.number_of_nodes(),
        "graph_edges":   G.number_of_edges(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Graph RAG", page_icon="🕸️")
    st.title("🕸️ Graph RAG — PDF Knowledge Q&A")
    st.caption("Upload a PDF → builds a knowledge graph → ask questions. Graph is saved so you won't rebuild it next time.")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Set it as an environment variable or import from Constants.py.")
        st.stop()

    index = load_index()

    with st.sidebar:
        st.header("📚 Saved Documents")
        if index:
            for doc_id, meta in index.items():
                st.markdown(f"**{meta['filename']}**")
                st.caption(f"{meta['chunks']} chunks · {meta['nodes']} nodes · {meta['edges']} edges")
        else:
            st.info("No documents yet.")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        doc_id    = pdf_hash(pdf_bytes)
        uploaded_file.seek(0)

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
            embeddings = embedder.encode([c["text"] for c in chunks], show_progress_bar=False)
            progress.progress(20, text="Extracting entities & relations…")

            entities, relations, chunk_ents = extract_graph(chunks, progress)
            progress.progress(85, text="Saving…")

            graph_data = {
                "chunks":        chunks,
                "embeddings":    embeddings.tolist(),
                "entities":      entities,
                "relations":     relations,
                "chunk_entities": chunk_ents,
            }
            save_graph_data(doc_id, graph_data)

            G_tmp = nx.DiGraph()
            for e in entities: G_tmp.add_node(e["id"])
            for r in relations: G_tmp.add_edge(r["source"], r["target"])
            index[doc_id] = {
                "filename": uploaded_file.name,
                "chunks":   len(chunks),
                "nodes":    G_tmp.number_of_nodes(),
                "edges":    G_tmp.number_of_edges(),
            }
            save_index(index)
            progress.progress(100, text="Done!")
            st.success("✅ Knowledge graph built and saved!")

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

        question = st.chat_input("Ask anything about the document…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching graph…"):
                    result = query_graph_rag(question, graph_data)
                st.markdown(result["answer"])
                with st.expander("🔎 Retrieval details"):
                    st.write(f"Chunks used: {result['chunks_used']}")
                    st.write(f"Entities expanded: {len(result['entities_used'])}")
                    st.write(f"Graph: {result['graph_nodes']} nodes, {result['graph_edges']} edges")

            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    else:
        st.info("👆 Upload a PDF to get started.")


if __name__ == "__main__":
    main()
