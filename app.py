"""
app.py - Streamlit chatbot for the AGORA AI Governance dataset.
Usage: streamlit run app.py
"""

import os
import json
import csv
import zipfile
import requests
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
DB_FILE = "vector_db.json"
AGORA_DIR = "agora"
FULLTEXT_DIR = os.path.join(AGORA_DIR, "fulltext")
DOCUMENTS_CSV = os.path.join(AGORA_DIR, "documents.csv")
GDRIVE_FILE_ID = "13V7LTJsagiUyXd1S_z6vfBQNvps2Zzla"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
MODEL = "gpt-4o-mini"

# ── Page setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Governance Chatbot", page_icon="⚖️", layout="centered")
st.title("⚖️ AI Governance Chatbot")
st.caption("Ask anything about global AI laws, regulations, and governance documents.")

# ── Auto-ingest helpers ─────────────────────────────────────────────────────
def download_from_gdrive(file_id: str, dest: str):
    """Download a file from Google Drive."""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    # Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={"id": file_id, "confirm": value}, stream=True)
            break

    with open(dest, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def load_metadata() -> dict:
    metadata = {}
    try:
        with open(DOCUMENTS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = str(row.get("id", "")).strip()
                if doc_id:
                    metadata[doc_id] = {
                        "title": str(row.get("title", "Unknown") or "Unknown")[:200],
                        "jurisdiction": str(row.get("jurisdiction", "Unknown") or "Unknown")[:100],
                        "year": str(row.get("year", "Unknown") or "Unknown")[:20],
                        "summary": str(row.get("summary", "") or ""),
                    }
    except Exception as e:
        st.warning(f"Could not load metadata: {e}")
    return metadata


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_database(openai_client):
    """Download data and build vector_db.json."""
    # Step 1: Download zip
    if not os.path.exists(AGORA_DIR):
        st.info("📥 Downloading AGORA dataset... (this only happens once, may take a minute)")
        zip_path = "agora.zip"
        download_from_gdrive(GDRIVE_FILE_ID, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(".")
        os.remove(zip_path)
        st.info("✅ Dataset downloaded!")

    # Step 2: Load metadata
    metadata = load_metadata()

    # Step 3: Chunk documents
    txt_files = [f for f in os.listdir(FULLTEXT_DIR) if f.endswith(".txt")]
    all_texts, all_metadatas = [], []

    for filename in txt_files:
        doc_id = filename.replace(".txt", "")
        filepath = os.path.join(FULLTEXT_DIR, filename)
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except:
            continue
        if not text:
            continue

        meta = metadata.get(doc_id, {})
        title = meta.get("title", filename)
        jurisdiction = meta.get("jurisdiction", "Unknown")
        year = meta.get("year", "Unknown")

        summary = meta.get("summary", "")
        if summary:
            all_texts.append(f"[{title}] SUMMARY: {summary}")
            all_metadatas.append({"title": title, "jurisdiction": jurisdiction, "year": year})

        for chunk in chunk_text(text):
            all_texts.append(f"[{title}] {chunk}")
            all_metadatas.append({"title": title, "jurisdiction": jurisdiction, "year": year})

    # Step 4: Embed
    progress = st.progress(0, text="🧠 Building vector database...")
    all_embeddings = []
    BATCH = 100
    for i in range(0, len(all_texts), BATCH):
        batch = all_texts[i:i + BATCH]
        response = openai_client.embeddings.create(input=batch, model="text-embedding-3-small")
        all_embeddings.extend([r.embedding for r in response.data])
        progress.progress(min(i + BATCH, len(all_texts)) / len(all_texts),
                         text=f"🧠 Embedding {min(i+BATCH, len(all_texts))}/{len(all_texts)} chunks...")

    progress.empty()

    # Step 5: Save
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": all_texts, "metadatas": all_metadatas, "embeddings": all_embeddings}, f)

    st.success(f"✅ Database built with {len(all_texts):,} chunks!")


# ── Load or build DB ────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OPENAI_API_KEY not found. Add it to your .env file or Streamlit secrets.")
    st.stop()

openai_client = OpenAI(api_key=api_key)

@st.cache_resource
def load_db():
    if not os.path.exists(DB_FILE):
        build_database(openai_client)
    with open(DB_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    db["embeddings"] = np.array(db["embeddings"], dtype=np.float32)
    return db

db = load_db()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Powered by the **AGORA** dataset — 600+ AI governance documents from around the world.")
    st.divider()
    st.metric("Chunks indexed", f"{len(db['texts']):,}")
    st.divider()
    st.markdown("**Try asking:**")
    st.markdown("- What AI laws exist in the EU?")
    st.markdown("- How does the US regulate facial recognition?")
    st.markdown("- Which countries have national AI strategies?")
    st.markdown("- What are the main AI governance themes?")
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── RAG ───────────────────────────────────────────────────────────────────────
def retrieve_context(query: str):
    response = openai_client.embeddings.create(input=[query], model="text-embedding-3-small")
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)
    scores = np.dot(db["embeddings"], query_vec) / (
        np.linalg.norm(db["embeddings"], axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    chunks = [db["texts"][i] for i in top_indices]
    titles = list({db["metadatas"][i]["title"] for i in top_indices})
    return "\n\n---\n\n".join(chunks), titles


def ask_gpt(user_question: str, chat_history: list) -> str:
    context, titles = retrieve_context(user_question)
    system_prompt = f"""You are an expert research assistant specializing in AI governance, policy, and regulation.

You have access to the AGORA dataset — 600+ AI-relevant laws, regulations, standards, and governance documents from the US and around the world.

Answer based on the document excerpts below. Be specific, cite document titles and jurisdictions when possible. If the answer isn't in the excerpts, say so clearly.

Relevant excerpts:
{context}

Sources: {', '.join(titles)}"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in chat_history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_question})

    response = openai_client.chat.completions.create(model=MODEL, max_tokens=1024, messages=messages)
    return response.choices[0].message.content


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about AI governance, laws, or regulations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            answer = ask_gpt(prompt, st.session_state.messages[:-1])
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
