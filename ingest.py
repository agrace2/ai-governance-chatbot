"""
ingest.py - Run this ONCE to process the AGORA dataset and build the vector database.
Usage: python ingest.py
"""

import os
import csv
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
AGORA_DIR = "agora"
FULLTEXT_DIR = os.path.join(AGORA_DIR, "fulltext")
DOCUMENTS_CSV = os.path.join(AGORA_DIR, "documents.csv")
DB_FILE = "vector_db.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ── Helpers ────────────────────────────────────────────────────────────────

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
        print(f"   ⚠️ Could not load metadata: {e}")
    print(f"   Loaded metadata for {len(metadata)} documents")
    return metadata


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AGORA AI Governance Chatbot — Ingestion Pipeline")
    print("=" * 60)

    if not os.path.exists(AGORA_DIR):
        print("❌ Could not find the 'agora' folder.")
        return

    # 1. Load metadata
    print(f"\n📋 Loading document metadata...")
    metadata = load_metadata()

    # 2. Read fulltext files
    print(f"\n📄 Reading fulltext documents...")
    txt_files = [f for f in os.listdir(FULLTEXT_DIR) if f.endswith(".txt")]
    print(f"   Found {len(txt_files)} documents")

    # 3. Chunk documents
    print(f"\n✂️  Chunking documents...")
    all_texts = []
    all_metadatas = []

    for filename in txt_files:
        doc_id = filename.replace(".txt", "")
        filepath = os.path.join(FULLTEXT_DIR, filename)

        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception as e:
            continue

        if not text:
            continue

        meta = metadata.get(doc_id, {})
        title = meta.get("title", filename)
        jurisdiction = meta.get("jurisdiction", "Unknown")
        year = meta.get("year", "Unknown")

        # Add summary chunk
        summary = meta.get("summary", "")
        if summary:
            all_texts.append(f"[{title}] SUMMARY: {summary}")
            all_metadatas.append({"title": title, "jurisdiction": jurisdiction, "year": year})

        # Add fulltext chunks
        for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            all_texts.append(f"[{title}] {chunk}")
            all_metadatas.append({"title": title, "jurisdiction": jurisdiction, "year": year})

    print(f"   ✅ Total chunks: {len(all_texts)}")

    # 4. Embed with OpenAI
    print(f"\n🧠 Embedding with OpenAI...")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_embeddings = []
    BATCH = 100
    for i in range(0, len(all_texts), BATCH):
        batch = all_texts[i:i + BATCH]
        try:
            response = openai_client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            batch_embeddings = [r.embedding for r in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"   Embedded {min(i + BATCH, len(all_texts))} / {len(all_texts)}", flush=True)
        except Exception as e:
            print(f"   ❌ Error at batch {i}: {e}")
            return

    # 5. Save to JSON file
    print(f"\n💾 Saving to {DB_FILE}...")
    db = {
        "texts": all_texts,
        "metadatas": all_metadatas,
        "embeddings": all_embeddings
    }
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f)

    print(f"✅ Done! Saved {len(all_texts)} chunks to {DB_FILE}")
    print(f"   You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
