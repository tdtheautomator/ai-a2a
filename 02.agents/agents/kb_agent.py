#!/usr/bin/env python3
# agents/kb_agent.py
"""
KB Reader Agent — A2A server on port 8001.

Responsibility: Qdrant semantic search ONLY.
All LLM calls (synthesis) are delegated to llm_agent via A2A.

Flow:
  1. Receive question
  2. Search Qdrant (or list files)
  3. POST hits to llm_agent → get plain-English answer
  4. Return answer

Run:
    uvicorn agents.kb_agent:app --port 8001 --reload
"""
import os
import sys
import time
import logging
import uvicorn
from typing import Any, Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from a2a.protocol import Message, AgentSkill, Task, TaskState
from a2a.client import A2AClient
from agents.base import BaseA2AAgent
from src.kb.kb_tools import search_knowledge_base, list_knowledge_base_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("kb_agent")

_PORT        = int(os.getenv("KB_AGENT_PORT",              "8001"))
_URL         = os.getenv("KB_AGENT_URL",                   f"http://localhost:{_PORT}")
_LLM_URL     = os.getenv("LLM_AGENT_URL",                 "http://localhost:8003")
_LLM_TIMEOUT = float(os.getenv("AGENT_READ_TIMEOUT_SECS", "120.0"))

# Synthesis system prompt — owned here, sent as metadata to llm_agent
_SYNTH_SYSTEM = """\
LANGUAGE REQUIREMENT: You MUST respond exclusively in English.

You are a Knowledge Base Agent. Read the retrieved document excerpts below and
write a clear, concise answer to the question using only that content.
End with a short "Sources:" line listing each file name.\
"""


class KBReaderAgent(BaseA2AAgent):

    def __init__(self):
        super().__init__(
            name        = "KB Reader Agent",
            description = (
                "Performs semantic search over indexed documents in Qdrant and "
                "returns an LLM-synthesised answer via the LLM Gateway Agent."
            ),
            url    = _URL,
            skills = [
                AgentSkill(
                    id          = "search_kb",
                    name        = "Search Knowledge Base",
                    description = "Semantic search over indexed PDF / text documents.",
                    examples    = [
                        "What is HAL?",
                        "What makes human conversations human?",
                        "What documents are available?",
                    ],
                ),
            ],
        )
        # No AsyncOpenAI client here — LLM calls go through llm_agent
        self._llm_client = A2AClient(
            _LLM_URL,
            timeout         = _LLM_TIMEOUT,
            connect_timeout = 10.0,
        )
        log.info("KB Agent init — url=%s  llm_agent=%s", _URL, _LLM_URL)

    async def _synthesise(self, question: str, hits: list, task_id: str) -> str:
        """
        Build the context string from Qdrant hits and delegate synthesis
        to llm_agent. Returns the answer text (or a fallback on failure).
        """
        context = "\n\n".join(
            f"[{h['rank']}] file={h['file_name']}  score={h['score']}\n{h['text']}"
            for h in hits
        )
        user_content = f"Answer in English:\n\nQuestion: {question}\n\nExcerpts:\n{context}"

        t0 = time.monotonic()
        log.info(
            "[KB-SYNTH] Delegating to llm_agent — hits=%d  context_chars=%d",
            len(hits), len(context),
        )

        llm_task = await self._llm_client.send_task(
            message  = user_content,
            metadata = {
                "system":     _SYNTH_SYSTEM,
                "max_tokens": 1000,
                "skill":      "synthesise_kb",
            },
        )
        ms = round((time.monotonic() - t0) * 1000)

        if llm_task.status.state == TaskState.COMPLETED:
            answer = llm_task.status.message.text() if llm_task.status.message else ""
            log.info("[KB-SYNTH] Done — elapsed=%d ms  chars=%d", ms, len(answer))
            return answer
        else:
            error = llm_task.status.message.text() if llm_task.status.message else "unknown"
            log.warning(
                "[KB-SYNTH] llm_agent FAILED after %d ms: %s — using fallback", ms, error
            )
            sources = ", ".join({h["file_name"] for h in hits})
            return (
                f"Found {len(hits)} relevant chunk(s) but could not synthesise "
                f"an answer (LLM error: {error}).\nSources: {sources}"
            )

    async def handle_task(
        self,
        task_id:  str,
        message:  Message,
        metadata: Dict[str, Any],
    ) -> Task:
        query = message.text().strip()
        if not query:
            return self.failed(task_id, message, "Empty query.")

        t0 = time.monotonic()
        log.info("[KB-TASK START]  id=%s  query=%r", task_id, query[:120])

        # ── list files if explicitly asked ────────────────────────────────────
        q_lower = query.lower()
        if any(kw in q_lower for kw in ("list", "available", "what files", "what documents")):
            log.info("[KB-TASK] Detected file-listing intent")
            t_list = time.monotonic()
            result = await list_knowledge_base_files()
            log.info(
                "[KB-TASK] list_knowledge_base_files elapsed=%d ms",
                round((time.monotonic() - t_list) * 1000),
            )
            if result["status"] == "error":
                return self.failed(task_id, message, result["message"])
            files = result.get("file_names", [])
            text  = (
                f"There are {result['total']} document(s) indexed:\n"
                + "\n".join(f"  • {f}" for f in files)
            )
            log.info(
                "[KB-TASK DONE]  id=%s  files=%d  elapsed=%d ms",
                task_id, result["total"], round((time.monotonic() - t0) * 1000),
            )
            return self.completed(task_id, message, text, data=result)

        # ── semantic search ───────────────────────────────────────────────────
        top_k  = int(metadata.get("top_k", 3))
        f_name = metadata.get("file_name")

        log.info("[KB-SEARCH] Querying Qdrant — top_k=%d  file_name=%s", top_k, f_name)
        t_search  = time.monotonic()
        result    = await search_knowledge_base(query=query, top_k=top_k, file_name=f_name)
        search_ms = round((time.monotonic() - t_search) * 1000)

        if result["status"] == "error":
            log.error("[KB-SEARCH] Qdrant error after %d ms: %s", search_ms, result["message"])
            return self.failed(task_id, message, result["message"])

        all_hits = result.get("results", [])
        hits     = [h for h in all_hits if h["score"] >= 0.5]
        log.info(
            "[KB-SEARCH] Done — elapsed=%d ms  total_hits=%d  above_threshold=%d  scores=%s",
            search_ms, len(all_hits), len(hits),
            [round(h["score"], 3) for h in all_hits],
        )

        if not hits:
            log.info("[KB-TASK] No hits above threshold — returning empty result")
            return self.completed(
                task_id, message,
                "No relevant documents found for that query.",
                data=result,
            )

        # ── delegate synthesis to llm_agent ───────────────────────────────────
        answer  = await self._synthesise(query, hits, task_id)
        elapsed = round((time.monotonic() - t0) * 1000)
        log.info("[KB-TASK DONE]  id=%s  hits=%d  elapsed=%d ms", task_id, len(hits), elapsed)

        return self.completed(task_id, message, answer, data=result)


agent = KBReaderAgent()
app   = agent.app

if __name__ == "__main__":
    uvicorn.run("agents.kb_agent:app", host="0.0.0.0", port=_PORT, reload=False)
