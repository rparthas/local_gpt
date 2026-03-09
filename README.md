# Local GPT

Chainlit app for chatting with uploaded PDFs using local embeddings, ChromaDB, and an Ollama model.

## What It Does

- Uploads PDF files through the Chainlit UI
- Extracts text with `PyPDF2`
- Chunks and embeds document text with `sentence-transformers`
- Stores embeddings in a Chroma collection
- Retrieves relevant chunks and uses Ollama to answer questions with document context

## Stack

- Python 3.11+
- Chainlit
- ChromaDB
- sentence-transformers
- PyPDF2
- Ollama

## Run

```bash
uv sync
ollama serve
uv run chainlit run main.py -w
```

Open the local Chainlit URL, upload one or more PDFs, then start asking questions.

## Notes

- The embedding model is hard-coded to `sentence-transformers/all-MiniLM-L6-v2`.
- The app uses a Chroma collection named `pdf_documents`.
- Storage is local to the running app unless you add persistent Chroma configuration.
