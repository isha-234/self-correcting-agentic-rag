"""
retriever.py — ChromaDB Retrieval Layer
=========================================
Loads the persisted ChromaDB collection and exposes a clean
retrieval interface for all agents.

Usage (standalone test):
    python retriever.py
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

load_dotenv(Path(__file__).parent.parent / ".env")

# --- Config from .env ---------------------------------------------------------
EMBEDDING_API_KEY  = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
CHROMA_DIR         = os.getenv("CHROMA_DIR", "./chroma_db")
DEFAULT_TOP_K      = int(os.getenv("TOP_K", "5"))


# --- Embedding client ---------------------------------------------------------
class APIEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model  = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


@dataclass
class RetrievedChunk:
    text:        str
    source:      str
    page:        int
    chunk_index: int
    score:       float

    def __repr__(self):
        snippet = self.text[:80].replace("\n", " ")
        return (f"RetrievedChunk(score={self.score:.3f}, "
                f"source={self.source}, page={self.page})\n"
                f'  "{snippet}..."')


class Retriever:
    """
    Wraps ChromaDB for similarity search with metadata.
    Used by RetrieverAgent and injected into the LangGraph pipeline.
    """

    def __init__(self, collection_name: str = "rag_collection", top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self._load_vectorstore(collection_name)

    def _load_vectorstore(self, collection_name: str):
        if not Path(CHROMA_DIR).exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DIR}.\n"
                f"Run ingestion first:\n"
                f"  python ingest.py --docs_dir ./docs"
            )
        if not EMBEDDING_API_KEY:
            raise EnvironmentError("EMBEDDING_API_KEY not set in .env")

        embeddings = APIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
        )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        count = self.vectorstore._collection.count()
        print(f"✅  Loaded ChromaDB: {count} vectors in '{collection_name}'")

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        k       = top_k or self.top_k
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        chunks  = [
            RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=int(doc.metadata.get("page", 0)),
                chunk_index=int(doc.metadata.get("chunk_index", 0)),
                score=round(float(score), 4),
            )
            for doc, score in results
        ]
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def retrieve_as_context_string(
        self, query: str, top_k: int = None
    ) -> Tuple[str, List[RetrievedChunk]]:
        chunks = self.retrieve(query, top_k=top_k)
        lines  = [
            f"[{i}] source={c.source}, page={c.page}, score={c.score:.3f}\n{c.text}"
            for i, c in enumerate(chunks, 1)
        ]
        return "\n\n---\n\n".join(lines), chunks

    def get_collection_stats(self) -> dict:
        return {
            "total_chunks":  self.vectorstore._collection.count(),
            "chroma_dir":    str(Path(CHROMA_DIR).resolve()),
            "embed_model":   EMBEDDING_MODEL,
            "default_top_k": self.top_k,
        }


# --- Standalone test --------------------------------------------------------

if __name__ == "__main__":
    collection = os.getenv("CHROMA_COLLECTION", "rag_collection")
    print(f"\n🔍  Retriever Smoke Test  (collection: {collection})")
    print("─" * 45)

    retriever = Retriever(collection_name=collection)
    stats     = retriever.get_collection_stats()
    print(f"\n📊  {stats['total_chunks']} chunks | model: {stats['embed_model']}\n")

    for query in [
        "What is the main topic of these documents?",
        "What methods or techniques are described?",
        "What are the key findings or conclusions?",
    ]:
        print(f'Query: "{query}"')
        for chunk in retriever.retrieve(query, top_k=3):
            print(f"  {chunk}")
        print()