"""
ingest.py — Document Ingestion & Embedding Pipeline
=====================================================
Loads PDFs/text files → chunks → embeds via Fireworks (nomic-embed)
→ stores in ChromaDB on disk.

Usage:
    python ingest.py --docs_dir ./docs
    python ingest.py --docs_dir ./docs --reset    # wipe + re-index
    python ingest.py --docs_dir ./docs --collection my_rag
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import argparse
import hashlib
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings


load_dotenv()

# ─── Config from .env ─────────────────────────────────────────────────────────
FIREWORKS_API_KEY  = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.fireworks.ai/inference/v1")
CHROMA_DIR         = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_SIZE         = 500
CHUNK_OVERLAP      = 50
BATCH_SIZE         = 20


# ─── Custom Fireworks Embeddings ──────────────────────────────────────────────
# Bypasses langchain_openai.OpenAIEmbeddings entirely to avoid the
# 'proxies' argument conflict between langchain-openai and openai>=1.35.
# Uses the openai client directly — same API, no wrapper issues.

class FireworksEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class that calls Fireworks AI directly
    using the openai client. Implements embed_documents and embed_query
    which is all ChromaDB needs.
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model  = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings with exponential backoff on rate limits."""
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if "429" in str(e) or "RATE_LIMIT" in str(e):
                    wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                    print(f"\n  ⏳ Rate limited — waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise
        raise Exception("Max retries exceeded on embedding request")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string — called during retrieval."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding


def get_embeddings() -> FireworksEmbeddings:
    if not FIREWORKS_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Add your Fireworks key to .env:\n"
            "  OPENAI_API_KEY=fw_your_key_here"
        )
    return FireworksEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=FIREWORKS_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )


# ─── Document Loaders ─────────────────────────────────────────────────────────

def load_documents(docs_dir: str) -> List[Document]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    files = list(docs_path.rglob("*.pdf")) + list(docs_path.rglob("*.txt"))
    if not files:
        raise ValueError(f"No PDF or .txt files found in {docs_dir}")

    print(f"\n📂  Found {len(files)} file(s) in {docs_dir}")
    all_docs: List[Document] = []

    for file_path in tqdm(files, desc="Loading files"):
        try:
            loader = PyPDFLoader(str(file_path)) if file_path.suffix == ".pdf" \
                     else TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path.relative_to(docs_path))
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ⚠️  Skipping {file_path.name}: {e}")

    print(f"✅  Loaded {len(all_docs)} page(s) from {len(files)} file(s)\n")
    return all_docs


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        chunk.metadata["chunk_id"]    = f"{chunk.metadata['source']}_{i}_{content_hash}"
        chunk.metadata["chunk_index"] = i

    avg = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    print(f"✂️   Split into {len(chunks)} chunks (avg {avg} chars each)\n")
    return chunks


# ─── Embedding + ChromaDB ──────────────────────────────────────────────────────

def build_vectorstore(
    chunks: List[Document],
    collection_name: str,
    reset: bool = False,
) -> Chroma:
    embeddings  = get_embeddings()
    chroma_path = Path(CHROMA_DIR)

    if reset and chroma_path.exists():
        import shutil
        shutil.rmtree(chroma_path)
        print(f"🗑️   Wiped existing ChromaDB at {CHROMA_DIR}\n")

    print(f"⚡  Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    print(f"    Batching {BATCH_SIZE} chunks per API call\n")

    import time as _time
    vectorstore = None
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = chunks[i : i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=CHROMA_DIR,
            )
        else:
            vectorstore.add_documents(batch)
        _time.sleep(2)  # 2s between every batch
        
    vectorstore.persist()
    count = vectorstore._collection.count()
    print(f"\n✅  ChromaDB ready — {count} vectors in '{collection_name}'")
    print(f"    Persisted at: {Path(CHROMA_DIR).resolve()}\n")
    return vectorstore


# ─── Smoke Test ───────────────────────────────────────────────────────────────

def smoke_test(vectorstore: Chroma, query: str = "What is this document about?"):
    print(f'🔍  Smoke test: "{query}"\n')
    results = vectorstore.similarity_search_with_relevance_scores(query, k=3)
    for rank, (doc, score) in enumerate(results, 1):
        source  = doc.metadata.get("source", "unknown")
        page    = doc.metadata.get("page", "?")
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  [{rank}] score={score:.3f}  source={source}  page={page}")
        print(f"       \"{snippet}...\"\n")


# ─── Stats ────────────────────────────────────────────────────────────────────

def print_stats(docs: List[Document], chunks: List[Document]):
    sources = set(c.metadata["source"] for c in chunks)
    avg     = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    print("=" * 55)
    print("  INGESTION SUMMARY")
    print("=" * 55)
    print(f"  Files indexed     : {len(sources)}")
    print(f"  Raw pages loaded  : {len(docs)}")
    print(f"  Chunks created    : {len(chunks)}")
    print(f"  Avg chunk length  : {avg} chars (~{avg//4} tokens)")
    print(f"  Embedding model   : {EMBEDDING_MODEL}")
    print(f"  Embedding provider: Fireworks AI")
    print(f"  ChromaDB path     : {Path(CHROMA_DIR).resolve()}")
    print("=" * 55)
    for src in sorted(sources):
        count = sum(1 for c in chunks if c.metadata["source"] == src)
        print(f"    • {src}  ({count} chunks)")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--docs_dir",   default="./docs",           help="Folder with PDFs/.txt files")
    parser.add_argument("--collection", default="rag_collection",   help="ChromaDB collection name")
    parser.add_argument("--reset",      action="store_true",        help="Wipe and rebuild collection")
    parser.add_argument("--test_query", default="What is this document about?")
    args = parser.parse_args()

    print("\n🚀  RAG Ingestion Pipeline (Fireworks AI)")
    print("─" * 45)

    docs        = load_documents(args.docs_dir)
    chunks      = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks, args.collection, args.reset)
    print_stats(docs, chunks)
    smoke_test(vectorstore, args.test_query)
    print("🎉  Done. ChromaDB is ready for RAG queries.\n")


if __name__ == "__main__":
    main()