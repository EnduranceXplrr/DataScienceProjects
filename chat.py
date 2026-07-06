import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# --- Setup (runs once) ---
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("pdf_chunks")

SYSTEM_PROMPT = """You are a helpful assistant answering questions about \
Lizard Monitoring, a refrigeration monitoring system. Answer ONLY using the \
provided context. If the context does not contain the answer, say \
"I don't know based on the documents." Keep answers short and factual."""

print("Ask about the Lizard Monitoring docs. Type 'quit' to exit.\n")

# --- Chat loop ---
while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() in ("quit", "exit"):
        break

    # 1. Embed the question (same model as ingestion!)
    query_embedding = model.encode([question]).tolist()

    # 2. Retrieve top 4 chunks
    results = collection.query(query_embeddings=query_embedding, n_results=4)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Build the context block, each chunk labeled with its source
    context = "\n\n".join(
        f"[{m['source']} p.{dist:.2f} p.{m['page']}]\n{d}" for d, m, dist in zip(docs, metas, distances)
    )

    prompt = f"""Context:
{context}

Question: {question}

Answer using only the context above."""

    # 4. Generate
    response = ollama.chat(model="llama3.2:1b", messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    answer = response["message"]["content"]

    # 5. Print answer + ground-truth sources (from retrieval, not the model)
    print(f"\nBot: {answer}\n")
    sources = {f"{m['source']}p.{dist:.2f} p.{m['page']}" for m,
               dist in zip(metas, distances)}
    print(f"Sources: {', '.join(sorted(sources))}")
    print("-" * 60)
