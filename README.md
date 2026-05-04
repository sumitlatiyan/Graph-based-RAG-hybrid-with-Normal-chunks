# 🕸️ Graph RAG — PDF Knowledge Q&A
 
A **production‑ready Graph RAG (Retrieval‑Augmented Generation)** application that allows you to upload PDFs, automatically build a **Knowledge Graph**, and ask questions grounded in both **semantic search** and **graph reasoning**.
 
Built using **Streamlit**, **NetworkX**, **Sentence Transformers**, and **Llama‑3 (Groq)** — with **persistent on‑disk caching** so repeated questions cost **0 tokens**.
 
---
 
## ✨ Features
 
- 📄 Upload PDF documents
- ✂️ Automatic text chunking with overlap
- 🔍 Semantic search using embeddings
- 🕸️ Knowledge Graph creation (entities + relations)
- 🔗 1‑hop graph expansion for contextual reasoning
- 💾 Persistent disk storage for graphs & embeddings
- ⚡ Query cache → repeated questions cost **0 tokens**
- 📊 Token usage tracking (input / output / saved)
- 💬 Chat‑based Q&A interface
 
---
 
## 🧠 Architecture Overview

PDF │ ▼ Text Chunks │ ├──► Embeddings (Semantic Search) │ └──► LLM Entity & Relation Extraction │ ▼ Knowledge Graph (NetworkX) │ ▼ Hybrid Retrieval ├── Top‑K semantic chunks └── 1‑hop graph expansion │ ▼ LLM Answer Generation (Llama‑3) │ ▼ Cached Response (0 tokens on repeat)
Plain Text
---
## 🚀 How to Run
### Option 1: Hosted App
If the application is already hosted, simply open the URL and start using it.
### Option 2: Run Locally
```bash
streamlit run app.py
(Replace app.py with your actual filename if different.)
🪜 Steps to Use
Open the application
Upload a PDF document
App performs:
Chunk extraction
Embedding generation
Entity & relation extraction
Knowledge graph creation
Ask questions in the chat
Ask the same question again → ⚡ 0 tokens
🔄 End‑to‑End Flow
Upload PDF
Split text into chunks
Create embeddings
Extract entities & relationships (LLM)
Build Knowledge Graph (NetworkX)
Store data on disk
Answer questions using:
Semantic retrieval
Graph expansion (1‑hop neighbors)
Cache answers for instant reuse
📦 Tech Stack
Component	Technology
UI	Streamlit
LLM	Llama‑3.1‑8B (Groq)
Embeddings	Sentence Transformers (MiniLM)
Graph	NetworkX
PDF Parsing	PyPDF2
Similarity	Scikit‑Learn
Storage	JSON + Pickle
Language	Python
🔑 Prerequisites
Python Version
Plain Text
Python >= 3.9


LLM API Key
Create a file named Constants.py:
 
 
 
 
Python
 
GROQ_API_KEY = "your_groq_api_key_here"

``

Show more lines
 
📁 Project Structure
Plain Text
.
├── app.py
├── Constants.py
├── graph_store/
│   ├── index.json
│   ├── query_cache.json
│   ├── <doc_id>.pkl
├── requirements.txt
└── README.md


⚡ Query Caching
Cache key format:
<document_id>::<question>
Cached answers:
Load instantly
Do not invoke the LLM
Save tokens automatically
Cache can be cleared from the sidebar
🔢 Token Accounting
Approximate estimation (~4 characters per token)
Tracks:
Input tokens
Output tokens
Tokens saved via cache
Session‑level metrics shown in the UI
🔍 Retrieval Strategy
Hybrid Graph RAG approach:
Semantic Retrieval
Top‑K chunks using cosine similarity
Graph Reasoning
Seed entities from retrieved chunks
Expand to 1‑hop neighbors
Answer grounded in:
Document excerpts
Knowledge graph context
✅ Why Graph RAG?
Compared to classic vector RAG:
✅ Stronger multi‑hop reasoning
✅ Explicit entity relationships
✅ Improved factual grounding
✅ Explainable retrieval logic
⚠️ Known Limitations
Entity extraction limited to ~60 chunks (cost control)
Token counts are approximate
Image‑only PDFs are not supported
🏁 Summary
This project showcases a clean, production‑grade Graph RAG pipeline, combining semantic retrieval with knowledge graphs and real‑world optimizations like persistent storage, caching, and token tracking.
Ideal for:
Enterprise document Q&A
Knowledge management systems
Research assistants
Policy and technical document exploration
