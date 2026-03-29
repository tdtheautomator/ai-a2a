#!/usr/bin/env python3
# agents/orchestrator.py
"""
Orchestrator Agent — A2A server on port 8000.
"""
import os
import sys
import re
import time
import logging
import asyncio
import uvicorn
from typing import Any, Dict, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from a2a.protocol import Message, AgentSkill, Task, TaskState
from a2a.client import A2AClient
from agents.base import BaseA2AAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("orchestrator")

# ── Config ────────────────────────────────────────────────────────────────────
_PORT    = int(os.getenv("ORCHESTRATOR_PORT",   "8000"))
_URL     = os.getenv("ORCHESTRATOR_URL",        f"http://localhost:{_PORT}")
_KB_URL  = os.getenv("KB_AGENT_URL",            "http://localhost:8001")
_DB_URL  = os.getenv("DB_AGENT_URL",            "http://localhost:8002")
_LLM_URL = os.getenv("LLM_AGENT_URL",           "http://localhost:8003")

# Route timeout: how long to wait for llm_agent routing decision.
# CPU-only Ollama with a reasoning model takes 60-120 s for a short reply.
_ROUTE_TIMEOUT = float(os.getenv("ROUTE_TIMEOUT_SECS", "180.0"))

_CACHE_TTL = float(os.getenv("ROUTE_CACHE_TTL", "300.0"))

# Agent call timeout: must cover the slowest possible agent chain.
# db_agent: SQL gen + exec + summarise ≈ 240 s on CPU.
# Set to 600 s (10 min) to be safe; lower once you have GPU.
_AGENT_READ_TIMEOUT    = float(os.getenv("AGENT_READ_TIMEOUT_SECS",    "600.0"))
_AGENT_CONNECT_TIMEOUT = float(os.getenv("AGENT_CONNECT_TIMEOUT_SECS", "10.0"))


# ─────────────────────────────────────────────────────────────────────────────
# Keyword fast-path
# ─────────────────────────────────────────────────────────────────────────────
_DB_RE = re.compile(
    r"""
    \b(
        how\s+many | count | total | sum | average | avg |
        top\s+\d+ | most | least | highest | lowest |
        per\s+(gender|category|merchant|month|day|year|week) |
        transaction[s]? | merchant[s]? | spend | spent | spending |
        transaction[_\s]?amount | revenue | sales |
        customer[_\s]?id | demodb | database | table | column |
        duplicate[s]? | skip[s]? | queue[s]? | from database |
        how\s+much | statistic[s]? | stats | report | breakdown |
        percent | ratio | rate | trend
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_KB_RE = re.compile(
    r"""
    \b(
        what\s+is | what\s+are | what\s+does |
        explain | describe | define | definition |
        how\s+does | how\s+do | tell\s+me\s+about |
        who\s+is | what\s+makes | from\s+(the\s+)?(documents|files|kb|knowledge\s+base) |
        list\s+(all\s+)?(document[s]?|file[s]?|kb|knowledge) |
        available\s+(document[s]?|file[s]?) |
        (document[s]?|file[s]?)\s+(available|indexed|in\s+the\s+(kb|knowledge\s+base)) |
        knowledge\s+base | paper[s]? | research |
        concept | overview | background | introduction | summary\s+of
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _keyword_route(question: str) -> Optional[str]:
    has_db = bool(_DB_RE.search(question))
    has_kb = bool(_KB_RE.search(question))
    if has_db and has_kb: return "both"
    if has_db:            return "db"
    if has_kb:            return "kb"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Route cache
# ─────────────────────────────────────────────────────────────────────────────
_route_cache: dict[str, tuple[str, float]] = {}

def _cache_get(question: str) -> Optional[str]:
    key   = question.strip().lower()
    entry = _route_cache.get(key)
    if entry:
        route, ts = entry
        if time.monotonic() - ts < _CACHE_TTL:
            return route
        del _route_cache[key]
    return None

def _cache_set(question: str, route: str) -> None:
    if len(_route_cache) >= 500:
        oldest = min(_route_cache, key=lambda k: _route_cache[k][1])
        del _route_cache[oldest]
    _route_cache[question.strip().lower()] = (route, time.monotonic())


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────
_ROUTER_SYSTEM = """
You are a routing agent. Given a user question, decide which agent(s) to call.

Available agents:
  "kb"   — Knowledge Base Reader: answers questions from indexed PDF/text documents.
  "db"   — Database Reader: queries PostgreSQL tables (transactions, files, stats).
  "both" — Use both agents and combine answers.

Reply with ONLY one word: kb  or  db  or  both

Examples:
  "What is HAL?" → kb
  "How many transactions?" → db
  "List available documents" → kb
  "Top merchants and what documents mention them?" → both
  "Average transaction amount per category" → db
  "What makes human conversation human?" → kb
"""

_SYNTHESISE_SYSTEM = """
You are a helpful assistant synthesising answers from one or more specialist agents.
Combine the information concisely and coherently.
If only one agent replied, present its answer clearly.
If both replied, blend the answers without duplication.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
class OrchestratorAgent(BaseA2AAgent):

    def __init__(self):
        super().__init__(
            name        = "Orchestrator Agent",
            description = (
                "Routes questions to KB/DB agents and synthesises answers via "
                "llm_agent. Makes zero direct LLM calls itself."
            ),
            url    = _URL,
            skills = [
                AgentSkill(
                    id          = "orchestrate",
                    name        = "Orchestrate",
                    description = "Route and synthesise answers from KB and DB agents.",
                    examples    = [
                        "What is HAL?",
                        "How many transactions are there?",
                        "Tell me about processed files and top merchants.",
                    ],
                ),
            ],
        )
        self._kb  = A2AClient(_KB_URL,  timeout=_AGENT_READ_TIMEOUT, connect_timeout=_AGENT_CONNECT_TIMEOUT)
        self._db  = A2AClient(_DB_URL,  timeout=_AGENT_READ_TIMEOUT, connect_timeout=_AGENT_CONNECT_TIMEOUT)
        self._llm = A2AClient(_LLM_URL, timeout=_AGENT_READ_TIMEOUT, connect_timeout=_AGENT_CONNECT_TIMEOUT)

        log.info(
            "Orchestrator init — kb=%s  db=%s  llm=%s  "
            "route_timeout=%.0fs  agent_read_timeout=%.0fs",
            _KB_URL, _DB_URL, _LLM_URL,
            _ROUTE_TIMEOUT, _AGENT_READ_TIMEOUT,
        )

    # ── Startup probe ─────────────────────────────────────────────────────────

    async def probe_agents(self) -> None:
        log.info("=== STARTUP PROBES ===")
        log.info("  KB Agent  : %s", _KB_URL)
        log.info("  DB Agent  : %s", _DB_URL)
        log.info("  LLM Agent : %s", _LLM_URL)
        log.info("  route_timeout      : %.0f s", _ROUTE_TIMEOUT)
        log.info("  agent_read_timeout : %.0f s", _AGENT_READ_TIMEOUT)
        kb_ok, db_ok, llm_ok = await asyncio.gather(
            self._kb.can_reach(),
            self._db.can_reach(),
            self._llm.can_reach(),
        )
        if not kb_ok:  log.error("KB Agent NOT reachable at %s",  _KB_URL)
        if not db_ok:  log.error("DB Agent NOT reachable at %s",  _DB_URL)
        if not llm_ok: log.error("LLM Agent NOT reachable at %s", _LLM_URL)
        log.info("=== STARTUP PROBES DONE ===")

    # ── LLM routing via llm_agent ─────────────────────────────────────────────

    async def _route_llm(self, question: str) -> str:
        t0 = time.monotonic()
        log.info("[ROUTE-LLM] Sending routing request to llm_agent")
        task = await self._llm.send_task(
            message  = question,
            metadata = {
                "system":     _ROUTER_SYSTEM,
                "max_tokens": 1000,
                "skill":      "route_question",
            },
        )
        ms = round((time.monotonic() - t0) * 1000)

        if task.status.state != TaskState.COMPLETED:
            err = task.status.message.text() if task.status.message else "unknown"
            log.error("[ROUTE-LLM] FAILED after %d ms: %s", ms, err)
            raise RuntimeError(f"llm_agent routing failed: {err}")

        raw = (task.status.message.text() if task.status.message else "").strip().lower()
        log.info("[ROUTE-LLM] Response=%r  elapsed=%d ms", raw, ms)

        if "both" in raw: return "both"
        if "db"   in raw: return "db"
        return "kb"

    # ── Synthesis via llm_agent ────────────────────────────────────────────────

    async def _synthesise(self, question: str, parts: dict[str, str]) -> str:
        t0 = time.monotonic()
        log.info("[SYNTHESIS] Delegating to llm_agent — agents=%s", list(parts.keys()))
        context      = "\n\n".join(
            f"=== {name.upper()} AGENT ===\n{text}" for name, text in parts.items()
        )
        task = await self._llm.send_task(
            message  = f"Question: {question}\n\n{context}",
            metadata = {
                "system":     _SYNTHESISE_SYSTEM,
                "max_tokens": 2000,
                "skill":      "synthesise_final",
            },
        )
        ms = round((time.monotonic() - t0) * 1000)
        if task.status.state == TaskState.COMPLETED:
            result = task.status.message.text() if task.status.message else ""
            log.info("[SYNTHESIS] Done — elapsed=%d ms  chars=%d", ms, len(result))
            return result
        log.warning("[SYNTHESIS] llm_agent failed after %d ms — returning raw concat", ms)
        return "\n\n---\n\n".join(parts.values())

    # ── Sub-agent helpers ─────────────────────────────────────────────────────

    async def _call_kb(self, question: str, metadata: dict, t0: float) -> str:
        log.info("[KB-CALL] → %s", _KB_URL)
        t_call = time.monotonic()
        task   = await self._kb.send_task(question, metadata=metadata)
        ms     = round((time.monotonic() - t_call) * 1000)
        text   = task.status.message.text() if task.status.message else ""
        if task.status.state == TaskState.COMPLETED:
            log.info("[KB-CALL] COMPLETED  elapsed=%d ms  total=%d ms",
                     ms, round((time.monotonic()-t0)*1000))
            return text
        log.error("[KB-CALL] FAILED  state=%s  elapsed=%d ms  msg=%r",
                  task.status.state.value, ms, text[:120])
        return f"KB Agent failed: {text}"

    async def _call_db(self, question: str, metadata: dict, t0: float) -> str:
        log.info("[DB-CALL] → %s", _DB_URL)
        t_call = time.monotonic()
        task   = await self._db.send_task(question, metadata=metadata)
        ms     = round((time.monotonic() - t_call) * 1000)
        text   = task.status.message.text() if task.status.message else ""
        if task.status.state == TaskState.COMPLETED:
            log.info("[DB-CALL] COMPLETED  elapsed=%d ms  total=%d ms",
                     ms, round((time.monotonic()-t0)*1000))
            return text
        log.error("[DB-CALL] FAILED  state=%s  elapsed=%d ms  msg=%r",
                  task.status.state.value, ms, text[:120])
        return f"DB Agent failed: {text}"

    # ── Task handler ──────────────────────────────────────────────────────────

    async def handle_task(
        self,
        task_id:  str,
        message:  Message,
        metadata: Dict[str, Any],
    ) -> Task:
        question = message.text().strip()
        if not question:
            return self.failed(task_id, message, "Empty question.")

        t0 = time.monotonic()
        log.info("=" * 60)
        log.info("[TASK START]  id=%s  q=%r", task_id, question[:120])

        # ── Step 1: keyword / cache ───────────────────────────────────────────
        route = _cache_get(question)
        if route:
            log.info("[STEP 1] cache  → %s  (%.0f ms)", route, (time.monotonic()-t0)*1000)
        else:
            route = _keyword_route(question)
            if route:
                log.info("[STEP 1] keyword → %s  (%.0f ms)", route, (time.monotonic()-t0)*1000)
            else:
                log.info("[STEP 1] ambiguous — will use llm_agent router")

        # ── Step 2: LLM routing (ambiguous questions only) ────────────────────
        answers: dict[str, str] = {}

        if route is None:
            log.info("[STEP 2] Speculative parallel + llm_agent routing  timeout=%.0fs",
                     _ROUTE_TIMEOUT)
            kb_task = asyncio.create_task(self._call_kb(question, metadata, t0))
            db_task = asyncio.create_task(self._call_db(question, metadata, t0))

            try:
                route = await asyncio.wait_for(self._route_llm(question), timeout=_ROUTE_TIMEOUT)
                log.info("[STEP 2] llm_agent → %s  (%.0f ms)", route, (time.monotonic()-t0)*1000)
            except asyncio.TimeoutError:
                route = "both"
                log.warning(
                    "[STEP 2] llm_agent timed out after %.0f s — using 'both'. "
                    "Raise ROUTE_TIMEOUT_SECS (current=%.0f).",
                    _ROUTE_TIMEOUT, _ROUTE_TIMEOUT,
                )
            except Exception as e:
                route = "both"
                log.warning("[STEP 2] llm_agent error (%s) — using 'both'", e)

            if route == "kb":
                db_task.cancel()
                answers = {"kb": await kb_task}
            elif route == "db":
                kb_task.cancel()
                answers = {"db": await db_task}
            else:
                kb_ans, db_ans = await asyncio.gather(kb_task, db_task)
                answers = {"kb": kb_ans, "db": db_ans}

            _cache_set(question, route)

        # ── Step 3: call agents (decisive route) ──────────────────────────────
        if not answers:
            log.info("[STEP 3] Calling agent(s) for route=%s", route)
            if route == "kb":
                answers = {"kb": await self._call_kb(question, metadata, t0)}
            elif route == "db":
                answers = {"db": await self._call_db(question, metadata, t0)}
            else:
                kb_ans, db_ans = await asyncio.gather(
                    self._call_kb(question, metadata, t0),
                    self._call_db(question, metadata, t0),
                )
                answers = {"kb": kb_ans, "db": db_ans}
            _cache_set(question, route)

        # ── Step 4: synthesise ────────────────────────────────────────────────
        if len(answers) == 1:
            log.info("[STEP 4] Single agent — skipping synthesis")
            final = next(iter(answers.values()))
        else:
            log.info("[STEP 4] Synthesising via llm_agent")
            final = await self._synthesise(question, answers)

        elapsed_ms = round((time.monotonic() - t0) * 1000)
        log.info("[TASK DONE]  id=%s  route=%s  elapsed=%d ms", task_id, route, elapsed_ms)
        log.info("=" * 60)

        return self.completed(
            task_id, message, final,
            data={"route": route, "agent_answers": answers, "elapsed_ms": elapsed_ms},
        )


# ── FastAPI app ───────────────────────────────────────────────────────────────
agent = OrchestratorAgent()
app   = agent.app

from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(application: FastAPI):
    await agent.probe_agents()
    yield

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run("agents.orchestrator:app", host="0.0.0.0", port=_PORT, reload=False)
