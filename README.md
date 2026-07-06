# PDF RAG Chatbot

A minimal Retrieval-Augmented Generation (RAG) chatbot built from scratch —
no LangChain, no frameworks. Ask questions about your own PDFs using a small
open-source LLM running fully locally.

## How it works

```
PDFs → extract text → clean → chunk → embed → ChromaDB      (ingest.py, run once)

question → embed → similarity search → top-k chunks
        → prompt → llama3.2:1b → answer + sources           (chat.py, chat loop)
```

## Stack

| Component  | Choice                                  |
|------------|-----------------------------------------|
| LLM        | `llama3.2:1b` via [Ollama](https://ollama.com) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB  | ChromaDB (local, persistent)             |
| PDF parser | PyMuPDF                                  |

Runs on modest hardware — a 4GB GPU (or CPU-only) is enough.

## Setup

```bash
# 1. Install Ollama and pull the model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b

# 2. Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
# 1. Drop your PDFs into this folder, then build the index
python ingest.py

# 2. Chat
python chat.py
```

Answers cite the source PDF and page number. Questions the documents can't
answer are refused instead of hallucinated (temperature 0 + a retrieval
distance gate).

## Tuning knobs

- `chunk_size` / `overlap` in `ingest.py` — chunk granularity (default 800/150 chars)
- `n_results` in `chat.py` — how many chunks feed the prompt (default 4)
- Distance threshold in `chat.py` — how aggressively out-of-scope questions are refused
- Model — swap `llama3.2:1b` for `qwen2.5:1.5b` for better answers, same VRAM class
