import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pymupdf
import re

FOOTER_PATTERNS = [
    re.compile(r"^https?://\S+$"),                # URL lines
    re.compile(r"^\d+/\d+$"),                     # page counter like "21/64"
    # date like "7/4/26, 12:30 AM"
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},.*$"),
    # orphaned AM/PM on its own line
    re.compile(r"^(AM|PM)$"),
]


def clean_page(text: str, title: str) -> str:
    """Remove browser print footers (URL, date, page counter, title) from page text."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == title:                     # the page-title footer line
            continue
        if any(p.match(stripped) for p in FOOTER_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


pages = []  # each item: {"text": ..., "source": ..., "page": ...}

for pdf_path in Path(".").glob("*.pdf"):
    doc = pymupdf.open(pdf_path)
    title = pdf_path.stem  # "Help.pdf" → "Help"
    # start=1 → human page numbers
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        text = clean_page(text, title)  # Remove footers
        if not text.strip():        # skip blank/image-only pages
            continue
        pages.append({
            "text": text,
            "source": pdf_path.name,   # .name → "Help.pdf", not the full path
            "page": page_num,
        })
    doc.close()

print(
    f"Loaded {len(pages)} pages from {len(list(Path('.').glob('*.pdf')))} PDFs")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, cutting at sentence ends when possible."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the very end of the text, try to cut at a
        # sentence boundary instead of mid-word.
        if end < len(text):
            # Look backwards from `end` for the nearest sentence end.
            # Search only the last 200 chars of the window, so a chunk
            # with no punctuation doesn't shrink to nothing.
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut > chunk_size - 200:      # found a boundary near the end
                end = start + cut + 1       # +1 keeps the "." inside the chunk

        chunk = text[start:end].strip()
        if chunk:                           # skip whitespace-only chunks
            chunks.append(chunk)

        start = end - overlap               # slide window back by `overlap`

    return chunks


chunks = []
for p in pages:
    for piece in chunk_text(p["text"]):
        chunks.append({
            "text": piece,
            "source": p["source"],
            "page": p["page"],
        })

print(f"Total chunks: {len(chunks)}")
print("-" * 60)
for c in chunks[::len(chunks)//3][:3]:      # 3 samples spread across the set
    print(f"[{c['source']} — page {c['page']}]")
    print(c["text"][:300])
    print("-" * 60)


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)


client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("pdf_chunks")
except Exception:
    pass  # didn't exist yet, create it
collection = client.get_or_create_collection("pdf_chunks")

collection.add(
    documents=[c["text"] for c in chunks],
    metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    embeddings=embeddings.tolist(),
    ids=[f"{c['source']}:{c['page']}:{i}" for i, c in enumerate(chunks)],
)

query_embedding = model.encode(
    ["How do I reset my password?"], show_progress_bar=False).tolist()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
)
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"[{meta['source']} — page {meta['page']}]")
    print(doc[:300])
    print("-" * 60)
