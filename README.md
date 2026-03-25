# AI Governance Chatbot

I built this chatbot to make it easier to search through hundreds of AI policy documents without having to read them all manually. You can ask it plain English questions and it'll pull relevant answers straight from the source documents.

🔗 **[Live demo](https://ai-governance-chatbot-m5jzvn6eq4jj6n8wntttvs.streamlit.app/)**

---

## What it does

The chatbot is loaded with the **AGORA dataset**, a collection of 600+ AI governance documents from the US and around the world, including laws, executive orders, regulations, and policy standards.

Ask questions like:
- *"What did the 2023 Executive Order on AI cover?"*
- *"How does the EU regulate facial recognition?"*
- *"Which countries have a national AI strategy?"*
- *"What are the main concerns around AI accountability?"*

---

## How it works

When you ask a question, the app searches through 13,000+ chunks of document text to find the most relevant passages, then feeds those passages to GPT-4o-mini to generate an answer. It only answers based on what's in the documents.

This approach is called RAG (Retrieval-Augmented Generation).

---

## Built with

- Streamlit — for the UI
- OpenAI — for embeddings and GPT-4o-mini
- NumPy — for vector similarity search
- AGORA dataset — from the [Emerging Technology Observatory](https://eto.tech/tool-docs/agora/)

---

## Run it locally

```bash
git clone https://github.com/agrace2/ai-governance-chatbot.git
cd ai-governance-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add your OpenAI key to a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

Download the AGORA dataset from [eto.tech](https://eto.tech/tool-docs/agora/#downloads), extract it into the project folder, then run:
```bash
python ingest.py
streamlit run app.py
```

---

Made by Angelina Grace Harrington
