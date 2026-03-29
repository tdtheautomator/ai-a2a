# src/kb/kb_tools.py
"""
SAM-compatible tools for the KB Reader Agent.
Follows the same pattern as rag_tools.py — tools are async functions
using os.getenv() for config, returning plain dicts.
"""
import os
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI  # sync client — same pattern as rag_tools.py

# ── Config (from env / .env) ────────────────────────────────────────────────
_QDRANT_HOST  = os.getenv("QDRANT_HOST",              "localhost")
_QDRANT_PORT  = int(os.getenv("QDRANT_PORT",          "6333"))
_COLLECTION   = os.getenv("QDRANT_COLLECTION",        "documents")
_LLM_BASE     = f"http://{os.getenv('LITELLM_HOST','localhost')}:{os.getenv('LITELLM_PORT','4000')}"
_LLM_KEY      = os.getenv("LITELLM_API_KEY",          "sk-1234")
_EMBED_MODEL  = os.getenv("LITELLM_EMBEDDING_MODEL",  "qwen3-embedding:0.6b")

# FIX 1 — The payload key that holds the chunk text varies by ingestion pipeline.
# Common values: "text", "content", "page_content", "chunk_text".
# Set QDRANT_TEXT_KEY in .env to match whatever your indexer wrote.
# If unset, "text" is tried first; then a fallback scan is used at query time.
_TEXT_KEY     = os.getenv("QDRANT_TEXT_KEY", "text")

# FIX 2 — Raised from 20 to 50 so deeper-ranked relevant chunks are reachable.
_MAX_TOP_K    = int(os.getenv("QDRANT_MAX_TOP_K", "50"))

# ── Internal helpers ─────────────────────────────────────────────────────────
def _embedder() -> OpenAI:
    return OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE)

def _qdrant() -> QdrantClient:
    return QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)

def _embed(text: str) -> list[float]:
    """Sync embed — same pattern as rag_tools.py."""
    resp = _embedder().embeddings.create(model=_EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def _extract_text(payload: dict) -> str:
    """
    FIX 1 (core): Robustly extract the chunk text from a Qdrant payload.

    Strategy:
      1. Try the configured _TEXT_KEY first (env: QDRANT_TEXT_KEY, default "text").
      2. Fall back through other common key names used by popular ingestion libs.
      3. If nothing matches, return "" AND log the actual keys so the operator
         knows exactly what to set QDRANT_TEXT_KEY to — instead of silently
         sending empty chunks to the LLM.
    """
    # Primary key (configurable)
    value = payload.get(_TEXT_KEY)
    if value:
        return str(value)

    # Common fallbacks — covers LangChain, LlamaIndex, Haystack defaults
    for fallback_key in ("content", "page_content", "chunk_text", "body", "passage"):
        value = payload.get(fallback_key)
        if value:
            return str(value)

    # chunk_preview fallback — your indexer stores a truncated preview but not
    # the full text. This keeps the agent functional with existing indexed data.
    # Re-index with the full text stored to get complete answers.
    value = payload.get("chunk_preview")
    if value:
        return str(value)

    # Nothing found — log the real keys so the operator can fix _TEXT_KEY
    actual_keys = list(payload.keys())
    print(
        f"  [kb_tools WARN] Could not find text in payload. "
        f"Actual keys: {actual_keys}. "
        f"Set QDRANT_TEXT_KEY=<correct_key> in your .env."
    )
    return ""


# ── SAM tool functions ───────────────────────────────────────────────────────

async def search_knowledge_base(
    query: str,
    top_k: int = 5,
    file_name: str = None,
    tool_context: Optional[Any] = None,
    tool_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform semantic search over indexed documents in the knowledge base.

    Args:
        query:     The question or search query to find relevant documents for.
        top_k:     Number of results to return (default 5, max QDRANT_MAX_TOP_K=50).
        file_name: Optional — restrict results to a single source file name.

    Returns:
        A dictionary with ranked results including text, file name, and score.
    """
    try:
        vector = _embed(query)

        qfilter = None
        if file_name:
            qfilter = Filter(
                must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))]
            )

        response = _qdrant().query_points(
            collection_name=_COLLECTION,
            query=vector,
            limit=min(int(top_k), _MAX_TOP_K),  # FIX 2: uses raised cap
            query_filter=qfilter,
            with_payload=True,
        )

        if not response.points:
            msg = (
                f"No relevant documents found in '{file_name}'."
                if file_name else
                "No relevant documents found for this query."
            )
            return {"status": "success", "message": msg, "results": []}

        results = []
        for i, h in enumerate(response.points, 1):
            text = _extract_text(h.payload)  # FIX 1: robust extraction
            results.append({
                "rank":        i,
                "text":        text,
                "file_name":   h.payload.get("file_name", "unknown"),
                "score":       round(h.score, 4),
                "chunk_index": h.payload.get("chunk_index", 0),
            })

        return {"status": "success", "results": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}


async def list_knowledge_base_files(
    tool_context: Optional[Any] = None,
    tool_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    List all unique source file names currently indexed in the knowledge base.

    Returns:
        A dictionary with a sorted list of available file names.
    """
    try:
        client = _qdrant()
        file_names: set[str] = set()
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=["file_name"],
            )
            for point in results:
                fn = point.payload.get("file_name")
                if fn:
                    file_names.add(fn)
            if offset is None:
                break

        return {"status": "success", "file_names": sorted(file_names), "total": len(file_names)}
    except Exception as e:
        return {"status": "error", "message": str(e)}