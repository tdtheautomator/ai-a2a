#!/usr/bin/env python3
"""
test_agents.py
==============
Structured end-to-end test suite for the A2A agent stack.

Test suites
-----------
1. HEALTH      — agent card reachability for all 4 agents
2. LLM         — llm_agent can summarise a provided text input
3. KB          — kb_agent retrieves + synthesises a KB answer (prints full response)
4. DB          — db_agent generates SQL + executes + summarises (prints full response)
5. ROUTING     — orchestrator keyword routing: given N questions, verify the
                  expected route (kb / db / both) without waiting for agent answers
6. FLOW-KB     — full chain: orchestrator → kb_agent → llm_agent
7. FLOW-DB     — full chain: orchestrator → db_agent → llm_agent (2× LLM calls)
8. FLOW-BOTH   — full chain: orchestrator → kb_agent + db_agent → llm_agent synthesis

Usage
-----
    python test_agents.py                   # all suites
    python test_agents.py --health          # health only
    python test_agents.py --llm             # LLM gateway only
    python test_agents.py --kb              # KB agent only
    python test_agents.py --db              # DB agent only
    python test_agents.py --routing         # routing decisions only (fast, no LLM)
    python test_agents.py --flow            # full orchestrator flows
    python test_agents.py --kb --db         # combine flags freely

Environment variables (or .env file)
-------------------------------------
    LLM_AGENT_URL       default http://localhost:8003
    KB_AGENT_URL        default http://localhost:8001
    DB_AGENT_URL        default http://localhost:8002
    ORCHESTRATOR_URL    default http://localhost:8000
    TEST_TIMEOUT_SECS   default 700  (generous for CPU-only Ollama)
"""
import sys
import os
import asyncio
import time
import argparse
from datetime import datetime
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2a.client import A2AClient
from a2a.protocol import TaskState

# ── URLs ──────────────────────────────────────────────────────────────────────
LLM_URL  = os.getenv("LLM_AGENT_URL",    "http://localhost:8003")
KB_URL   = os.getenv("KB_AGENT_URL",     "http://localhost:8001")
DB_URL   = os.getenv("DB_AGENT_URL",     "http://localhost:8002")
ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

TEST_TIMEOUT = float(os.getenv("TEST_TIMEOUT_SECS", "700.0"))

# ── Formatting ────────────────────────────────────────────────────────────────
W      = 72
SEP    = "─" * W
SEP2   = "═" * W
INDENT = "    "

def _ts() -> str:
    """Current timestamp string for log lines."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _header(title: str):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP2)

def _subheader(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def _pass(label: str, elapsed_s: float, note: str = ""):
    tag = f"  [{_ts()}]  ✓  PASS  {label}"
    timing = f"({elapsed_s:.1f}s)"
    suffix = f"  {note}" if note else ""
    print(f"{tag:<55} {timing}{suffix}")

def _fail(label: str, elapsed_s: float, reason: str = ""):
    tag = f"  [{_ts()}]  ✗  FAIL  {label}"
    timing = f"({elapsed_s:.1f}s)"
    print(f"{tag:<55} {timing}")
    if reason:
        for line in reason.splitlines()[:5]:
            print(f"{INDENT}      {line}")

def _info(msg: str):
    print(f"  [{_ts()}]  ℹ  {msg}")

def _print_response(text: str, max_lines: int = 25):
    """Pretty-print an agent response, truncating if too long."""
    if not text:
        print(f"{INDENT}  (empty response)")
        return
    lines = text.strip().splitlines()
    print()
    for line in lines[:max_lines]:
        print(f"{INDENT}  {line}")
    if len(lines) > max_lines:
        print(f"{INDENT}  … ({len(lines) - max_lines} more lines)")
    print()


# ── Result tracker ─────────────────────────────────────────────────────────────
class Results:
    def __init__(self):
        self._rows: list[tuple[str, bool, float, str]] = []

    def record(self, label: str, passed: bool, elapsed_s: float, note: str = ""):
        self._rows.append((label, passed, elapsed_s, note))
        if passed:
            _pass(label, elapsed_s, note)
        else:
            _fail(label, elapsed_s, note)

    def summary(self) -> bool:
        passed = [r for r in self._rows if r[1]]
        failed = [r for r in self._rows if not r[1]]
        print(f"\n{SEP2}")
        print("  SUMMARY")
        print(SEP2)
        for label, ok, elapsed, note in self._rows:
            icon = "✓" if ok else "✗"
            timing = f"({elapsed:.1f}s)"
            n = f"  — {note}" if note else ""
            print(f"  {icon}  {label:<45} {timing:>8}{n}")
        print(SEP)
        print(f"  Passed : {len(passed)} / {len(self._rows)}")
        if failed:
            print(f"  Failed : {len(failed)}")
            for label, _, elapsed, note in failed:
                print(f"    ✗  {label}  ({elapsed:.1f}s){' — '+note if note else ''}")
        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(SEP2)
        return len(failed) == 0


R = Results()


# ── Client factory ─────────────────────────────────────────────────────────────
def _client(url: str) -> A2AClient:
    return A2AClient(url, timeout=TEST_TIMEOUT, connect_timeout=10.0)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 1 — HEALTH
# ══════════════════════════════════════════════════════════════════════════════
async def suite_health() -> bool:
    _header("Suite 1 — HEALTH  (agent card reachability)")

    agents = [
        ("LLM Agent",    LLM_URL),
        ("KB Agent",     KB_URL),
        ("DB Agent",     DB_URL),
        ("Orchestrator", ORCH_URL),
    ]

    all_ok = True
    for name, url in agents:
        c  = _client(url)
        t0 = time.monotonic()
        try:
            card    = await c.get_agent_card()
            elapsed = time.monotonic() - t0
            skills  = [s.get("name", "?") for s in card.get("skills", [])]
            note    = f"v{card.get('version','?')}  skills={skills}"
            R.record(f"health/{name}", True, elapsed, note)
        except Exception as e:
            elapsed = time.monotonic() - t0
            R.record(f"health/{name}", False, elapsed, str(e)[:80])
            all_ok = False
        finally:
            await c.aclose()

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Suite 2 — LLM AGENT
# ══════════════════════════════════════════════════════════════════════════════
_LLM_SUMMARISE_SYSTEM = (
    "You are a text summariser. "
    "Read the text and return a single concise summary sentence."
)
_LLM_SUMMARISE_INPUT = (
    "The Amazon rainforest, often referred to as the 'lungs of the Earth', "
    "produces roughly 20% of the world's oxygen. It spans nine countries and "
    "covers over 5.5 million square kilometres. The forest is home to an "
    "estimated 10% of all species on Earth, many still undiscovered. "
    "Deforestation driven by agriculture, logging, and infrastructure has "
    "destroyed around 17% of the forest over the last 50 years, threatening "
    "biodiversity and accelerating climate change."
)

async def suite_llm() -> bool:
    _header("Suite 2 — LLM AGENT  (direct summarisation)")
    _subheader("Test: summarise a fixed paragraph")
    _info(f"URL        : {LLM_URL}")
    _info(f"Skill      : summarise_kb  (generic completion)")
    _info(f"Input chars: {len(_LLM_SUMMARISE_INPUT)}")
    _info(f"max_tokens : 512  (reasoning models need room for <think> block)")

    c  = _client(LLM_URL)
    t0 = time.monotonic()
    try:
        task    = await c.send_task(
            message  = _LLM_SUMMARISE_INPUT,
            metadata = {
                "system":     _LLM_SUMMARISE_SYSTEM,
                "max_tokens": 512,
                "skill":      "summarise_kb",
            },
        )
        elapsed = time.monotonic() - t0
        state   = task.status.state
        text    = task.status.message.text() if task.status.message else ""
        msg     = task.status.message.text() if task.status.message else "(no message)"

        _info(f"State   : {state.value}  ({elapsed:.1f}s)")

        if state != TaskState.COMPLETED:
            _info(f"Failure reason: {msg}")
            _info("Diagnosis:")
            if elapsed < 10:
                _info("  < 10s response — llm_agent returned an error without")
                _info("  calling Ollama. Check llm_agent logs:")
                _info("  docker logs demo-llm-agent --tail 30")
            else:
                _info("  Ollama was called but returned empty content.")
                _info("  Most likely: reasoning model exhausted max_tokens in <think> block.")
                _info("  The llm_agent retry logic should have handled this.")
                _info("  Check llm_agent logs for [LLM-EMPTY] or [LLM-RETRY]:")
                _info("  docker logs demo-llm-agent --tail 50")

        _info("Response:")
        _print_response(text)

        passed = state == TaskState.COMPLETED and bool(text.strip())
        if not passed and state == TaskState.COMPLETED and not text.strip():
            note = "COMPLETED but empty text — reasoning model token budget too low"
        elif not passed:
            note = f"state={state.value} — {msg[:80]}"
        else:
            note = ""
        R.record("llm/summarise", passed, elapsed, note)
        return passed
    except Exception as e:
        elapsed = time.monotonic() - t0
        R.record("llm/summarise", False, elapsed, str(e)[:100])
        return False
    finally:
        await c.aclose()


# ══════════════════════════════════════════════════════════════════════════════
# Suite 3 — KB AGENT  (direct)
# ══════════════════════════════════════════════════════════════════════════════
_KB_QUESTIONS = [
    ("What is HAL?",                        "kb/what-is-hal"),
    ("What makes human conversation human?", "kb/human-conversation"),
    ("List available documents",             "kb/list-docs"),
]

async def suite_kb() -> bool:
    _header("Suite 3 — KB AGENT  (Qdrant search → llm_agent synthesis)")
    _info(f"URL: {KB_URL}")

    all_ok = True
    for question, label in _KB_QUESTIONS:
        _subheader(f"Test: {label}")
        _info(f"Question: {question}")

        c  = _client(KB_URL)
        t0 = time.monotonic()
        try:
            task    = await c.send_task(question)
            elapsed = time.monotonic() - t0
            state   = task.status.state
            text    = task.status.message.text() if task.status.message else ""

            # Surface Qdrant hit metadata if present
            for artifact in (task.artifacts or []):
                for part in artifact.parts:
                    if hasattr(part, "data") and isinstance(part.data, dict):
                        hits = part.data.get("results", [])
                        if hits:
                            _info(f"Qdrant hits: {len(hits)}  "
                                  f"top score: {hits[0].get('score', '?'):.3f}")

            _info(f"State   : {state.value}  ({elapsed:.1f}s)")
            if state != TaskState.COMPLETED or not text.strip():
                hint = text[:120] if text else "(no message)"
                _info(f"  Detail : {hint}")
                if elapsed < 10:
                    _info("  Fast fail — Qdrant/llm_agent unreachable. docker logs demo-kb-agent --tail 30")
                else:
                    _info("  Slow fail — llm synthesis empty/timeout. docker logs demo-llm-agent --tail 40")
            _info("Response:")
            _print_response(text)

            passed = state == TaskState.COMPLETED and bool(text.strip())
            note   = f"state={state.value}  {text[:60]}" if state != TaskState.COMPLETED else ("COMPLETED but empty — check [LLM-EMPTY] in llm_agent logs" if not text.strip() else "")
            R.record(label, passed, elapsed, note)
            if not passed:
                all_ok = False
        except Exception as e:
            elapsed = time.monotonic() - t0
            R.record(label, False, elapsed, str(e)[:100])
            all_ok = False
        finally:
            await c.aclose()

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Suite 4 — DB AGENT  (direct)
# ══════════════════════════════════════════════════════════════════════════════
_DB_QUESTIONS = [
    ("How many transactions are there?",          "db/count"),
    ("Top 5 spending categories by total amount", "db/top5-categories"),
    ("Average transaction amount per gender",     "db/avg-by-gender"),
]

async def suite_db() -> bool:
    _header("Suite 4 — DB AGENT  (llm_agent SQL gen → Postgres → llm_agent summarise)")
    _info(f"URL: {DB_URL}")

    all_ok = True
    for question, label in _DB_QUESTIONS:
        _subheader(f"Test: {label}")
        _info(f"Question: {question}")

        c  = _client(DB_URL)
        t0 = time.monotonic()
        try:
            task    = await c.send_task(question)
            elapsed = time.monotonic() - t0
            state   = task.status.state
            text    = task.status.message.text() if task.status.message else ""

            # Surface SQL and row count from artifact metadata
            for artifact in (task.artifacts or []):
                for part in artifact.parts:
                    if hasattr(part, "data") and isinstance(part.data, dict):
                        sql       = part.data.get("sql", "")
                        row_count = part.data.get("row_count", "?")
                        if sql:
                            _info(f"SQL generated : {sql[:120].replace(chr(10),' ')}")
                            _info(f"Rows returned : {row_count}")

            _info(f"State   : {state.value}  ({elapsed:.1f}s)")
            if state != TaskState.COMPLETED or not text.strip():
                hint = text[:120] if text else "(no message)"
                _info(f"  Detail : {hint}")
                if elapsed < 10:
                    _info("  Fast fail — llm_agent/Postgres unreachable. docker logs demo-db-agent --tail 30")
                else:
                    _info("  Slow fail — SQL gen or summarise empty. docker logs demo-llm-agent --tail 40")
            _info("Response:")
            _print_response(text)

            passed = state == TaskState.COMPLETED and bool(text.strip())
            note   = ("COMPLETED but empty — see [LLM-EMPTY] in llm_agent logs"
                    if state == TaskState.COMPLETED and not text.strip()
                    else f"state={state.value}  {text[:60]}" if not passed else "")
            R.record(label, passed, elapsed, note)
            if not passed:
                all_ok = False
        except Exception as e:
            elapsed = time.monotonic() - t0
            R.record(label, False, elapsed, str(e)[:100])
            all_ok = False
        finally:
            await c.aclose()

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Suite 5 — ROUTING  (orchestrator keyword routing, no LLM wait)
# ══════════════════════════════════════════════════════════════════════════════
# These questions are designed to hit the keyword fast-path so the routing
# decision is made in <1 ms without waiting for Ollama.  The test verifies
# the route that was taken (from task.artifacts data) matches expected.
_ROUTING_CASES = [
    # (question,                                    expected_route, label)
    ("What is HAL?",                               "kb",   "routing/kb-concept"),
    ("Explain how embeddings work",                "kb",   "routing/kb-explain"),
    ("List available documents",                   "kb",   "routing/kb-list"),
    ("How many transactions are there?",           "db",   "routing/db-count"),
    ("Top 5 spending categories by total amount",  "db",   "routing/db-top5"),
    ("Average transaction amount per gender",      "db",   "routing/db-avg"),
    ("List KB documents and total transactions",   "both", "routing/both"),
]

async def suite_routing() -> bool:
    _header("Suite 5 — ROUTING  (orchestrator keyword routing, no LLM wait)")
    _info(f"URL: {ORCH_URL}")
    _info("These questions are chosen to hit the keyword fast-path (<1 ms).")
    _info("Test verifies the 'route' field in the orchestrator response.")

    # Use a short timeout here — keyword routing should be nearly instant.
    # If it takes >30 s the question fell through to LLM routing.
    routing_timeout = float(os.getenv("ROUTING_TEST_TIMEOUT_SECS", "60.0"))
    all_ok = True

    for question, expected, label in _ROUTING_CASES:
        _subheader(f"Test: {label}")
        _info(f"Question : {question}")
        _info(f"Expected : route={expected}")

        c  = A2AClient(ORCH_URL, timeout=routing_timeout, connect_timeout=10.0)
        t0 = time.monotonic()
        try:
            task    = await c.send_task(question)
            elapsed = time.monotonic() - t0
            state   = task.status.state

            # Extract the route from artifact data
            actual_route: Optional[str] = None
            for artifact in (task.artifacts or []):
                for part in artifact.parts:
                    if hasattr(part, "data") and isinstance(part.data, dict):
                        actual_route = part.data.get("route")

            route_ok = actual_route == expected
            passed   = state == TaskState.COMPLETED and route_ok
            note = f"route={actual_route}  expected={expected}"

            _info(f"State    : {state.value}  ({elapsed:.1f}s)")
            _info(f"Route    : {actual_route}  {'✓' if route_ok else '✗ MISMATCH'}")

            R.record(label, passed, elapsed, note)
            if not passed:
                all_ok = False
        except Exception as e:
            elapsed = time.monotonic() - t0
            R.record(label, False, elapsed, str(e)[:100])
            all_ok = False
        finally:
            await c.aclose()

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Suite 6 — FLOW-KB  (orchestrator → kb_agent → llm_agent)
# ══════════════════════════════════════════════════════════════════════════════
_FLOW_KB_QUESTIONS = [
    ("What is HAL?",                         "flow-kb/what-is-hal"),
    ("What makes human conversation human?",  "flow-kb/human-conversation"),
]

async def suite_flow_kb() -> bool:
    _header("Suite 6 — FLOW-KB  (orchestrator → kb_agent → llm_agent)")
    _info(f"URL: {ORCH_URL}")
    _info("Verifies: correct route=kb, non-empty answer, per-agent answer logged.")

    all_ok = True
    for question, label in _FLOW_KB_QUESTIONS:
        _subheader(f"Test: {label}")
        _info(f"Question: {question}")

        c  = _client(ORCH_URL)
        t0 = time.monotonic()
        try:
            task    = await c.send_task(question)
            elapsed = time.monotonic() - t0
            state   = task.status.state
            text    = task.status.message.text() if task.status.message else ""

            route        = None
            agent_answers: dict = {}
            for artifact in (task.artifacts or []):
                for part in artifact.parts:
                    if hasattr(part, "data") and isinstance(part.data, dict):
                        route         = part.data.get("route")
                        agent_answers = part.data.get("agent_answers", {})

            _info(f"State          : {state.value}  ({elapsed:.1f}s)")
            _info(f"Route taken    : {route}")

            for agent, answer in agent_answers.items():
                preview = (answer or "")[:120].replace("\n", " ")
                _info(f"  [{agent.upper()}] {preview}{'…' if len(answer or '')>120 else ''}")

            _info("Final response:")
            _print_response(text)

            route_ok = route == "kb"
            passed   = state == TaskState.COMPLETED and bool(text.strip()) and route_ok
            note     = f"route={route}" + ("" if route_ok else f"  expected=kb")
            R.record(label, passed, elapsed, note)
            if not passed:
                all_ok = False
        except Exception as e:
            elapsed = time.monotonic() - t0
            R.record(label, False, elapsed, str(e)[:100])
            all_ok = False
        finally:
            await c.aclose()

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Suite 7 — FLOW-DB  (orchestrator → db_agent → llm_agent × 2)
# ══════════════════════════════════════════════════════════════════════════════
_FLOW_DB_QUESTIONS = [
    ("How many transactions are there?",          "flow-db/count"),
    ("Top 5 spending categories by total amount", "flow-db/top5"),
]

async def suite_flow_db() -> bool:
    _header("Suite 7 — FLOW-DB  (orchestrator → db_agent → llm_agent SQL + summarise)")
    _info(f"URL: {ORCH_URL}")
    _info("Each question triggers 2 LLM calls inside db_agent (SQL gen + summarise).")

    all_ok = True
    for question, label in _FLOW_DB_QUESTIONS:
        _subheader(f"Test: {label}")
        _info(f"Question: {question}")

        c  = _client(ORCH_URL)
        t0 = time.monotonic()
        try:
            task    = await c.send_task(question)
            elapsed = time.monotonic() - t0
            state   = task.status.state
            text    = task.status.message.text() if task.status.message else ""

            route        = None
            agent_answers: dict = {}
            for artifact in (task.artifacts or []):
                for part in artifact.parts:
                    if hasattr(part, "data") and isinstance(part.data, dict):
                        route         = part.data.get("route")
                        agent_answers = part.data.get("agent_answers", {})

            _info(f"State          : {state.value}  ({elapsed:.1f}s)")
            _info(f"Route taken    : {route}")

            for agent, answer in agent_answers.items():
                preview = (answer or "")[:120].replace("\n", " ")
                _info(f"  [{agent.upper()}] {preview}{'…' if len(answer or '')>120 else ''}")

            _info("Final response:")
            _print_response(text)

            route_ok = route == "db"
            passed   = state == TaskState.COMPLETED and bool(text.strip()) and route_ok
            note     = f"route={route}" + ("" if route_ok else "  expected=db")
            R.record(label, passed, elapsed, note)
            if not passed:
                all_ok = False
        except Exception as e:
            elapsed = time.monotonic() - t0
            R.record(label, False, elapsed, str(e)[:100])
            all_ok = False
        finally:
            await c.aclose()

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Suite 8 — FLOW-BOTH  (orchestrator → kb + db → llm_agent synthesis)
# ══════════════════════════════════════════════════════════════════════════════
_FLOW_BOTH_QUESTIONS = [
    (
        "List KB documents and total transactions",
        "flow-both/docs-and-transactions",
    ),
]

async def suite_flow_both() -> bool:
    _header("Suite 8 — FLOW-BOTH  (orchestrator → kb + db in parallel → llm_agent synthesis)")
    _info(f"URL: {ORCH_URL}")
    _info("Verifies parallel fan-out and final synthesis step.")

    all_ok = True
    for question, label in _FLOW_BOTH_QUESTIONS:
        _subheader(f"Test: {label}")
        _info(f"Question: {question}")

        c  = _client(ORCH_URL)
        t0 = time.monotonic()
        try:
            task    = await c.send_task(question)
            elapsed = time.monotonic() - t0
            state   = task.status.state
            text    = task.status.message.text() if task.status.message else ""

            route        = None
            agent_answers: dict = {}
            orch_elapsed  = None
            for artifact in (task.artifacts or []):
                for part in artifact.parts:
                    if hasattr(part, "data") and isinstance(part.data, dict):
                        route         = part.data.get("route")
                        agent_answers = part.data.get("agent_answers", {})
                        orch_elapsed  = part.data.get("elapsed_ms")

            _info(f"State           : {state.value}  ({elapsed:.1f}s)")
            _info(f"Route taken     : {route}")
            if orch_elapsed:
                _info(f"Orchestrator ms : {orch_elapsed}")

            for agent, answer in agent_answers.items():
                preview = (answer or "")[:120].replace("\n", " ")
                _info(f"  [{agent.upper()}] {preview}{'…' if len(answer or '')>120 else ''}")

            _info("Synthesised response:")
            _print_response(text)

            route_ok = route == "both"
            both_answered = "kb" in agent_answers and "db" in agent_answers
            passed = (
                state == TaskState.COMPLETED
                and bool(text.strip())
                and route_ok
                and both_answered
            )
            note = f"route={route}  agents={list(agent_answers.keys())}"
            R.record(label, passed, elapsed, note)
            if not passed:
                all_ok = False
        except Exception as e:
            elapsed = time.monotonic() - t0
            R.record(label, False, elapsed, str(e)[:100])
            all_ok = False
        finally:
            await c.aclose()

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
async def main():
    parser = argparse.ArgumentParser(
        description="A2A agent stack test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--health",  action="store_true", help="Agent card health check")
    parser.add_argument("--llm",     action="store_true", help="LLM gateway summarisation")
    parser.add_argument("--kb",      action="store_true", help="KB agent direct tests")
    parser.add_argument("--db",      action="store_true", help="DB agent direct tests")
    parser.add_argument("--routing", action="store_true", help="Orchestrator routing decisions")
    parser.add_argument("--flow",    action="store_true", help="Full orchestrator flow tests")
    args = parser.parse_args()

    # If no flags given, run everything
    run_all = not any([args.health, args.llm, args.kb, args.db, args.routing, args.flow])

    print(f"\n{SEP2}")
    print("  A2A Agent Stack — Test Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  LLM Agent   : {LLM_URL}")
    print(f"  KB Agent    : {KB_URL}")
    print(f"  DB Agent    : {DB_URL}")
    print(f"  Orchestrator: {ORCH_URL}")
    print(f"  Timeout     : {TEST_TIMEOUT}s per call")
    print(SEP2)

    # Health always runs first; abort if agents are unreachable
    health_ok = await suite_health()
    if not health_ok:
        print(f"\n  ⚠  One or more agents are unreachable.")
        print("  Ensure both stacks are running:")
        print("    docker compose -f docker-compose-infra.yaml up -d")
        print("    docker compose -f docker-compose-agents.yaml up -d")
        print("  Or locally:")
        print("    uvicorn agents.llm_agent:app    --port 8003")
        print("    uvicorn agents.kb_agent:app     --port 8001")
        print("    uvicorn agents.db_agent:app     --port 8002")
        print("    uvicorn agents.orchestrator:app --port 8000")
        R.summary()
        sys.exit(1)

    if args.llm     or run_all: await suite_llm()
    if args.kb      or run_all: await suite_kb()
    if args.db      or run_all: await suite_db()
    if args.routing or run_all: await suite_routing()
    if args.flow    or run_all:
        await suite_flow_kb()
        await suite_flow_db()
        await suite_flow_both()

    all_passed = R.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())