#!/usr/bin/env python3
"""
ui_server.py
============
Demo UI server — port 8080.

Serves the HTML frontend and exposes API endpoints:

  GET  /                              → serves ui/index.html
  GET  /api/status                    → pings all agent cards, returns health JSON
  GET  /api/agents                    → full card details for all 4 agents
  POST /api/ask                       → proxies question to orchestrator
  GET  /api/logs                      → SSE stream of live log events (log panel)
  POST /api/tests/run                 → launches test_a2a_v2.py as subprocess
  GET  /api/tests/output/{job_id}     → SSE stream of test output for a job

Run:
    python ui_server.py
    uvicorn ui_server:app --port 8080 --reload
"""
import os
import sys
import json
import uuid
import asyncio
import logging
import httpx
import uvicorn
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Config ─────────────────────────────────────────────────────────────────
_PORT    = int(os.getenv("UI_PORT",          "8080"))
_LLM_URL = os.getenv("LLM_AGENT_URL",       "http://localhost:8003")
_KB_URL  = os.getenv("KB_AGENT_URL",        "http://localhost:8001")
_DB_URL  = os.getenv("DB_AGENT_URL",        "http://localhost:8002")
_OR_URL  = os.getenv("ORCHESTRATOR_URL",    "http://localhost:8000")

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("ui_server")

# ── Global SSE log queue ───────────────────────────────────────────────────
_sse_subscribers: list[asyncio.Queue] = []

async def _broadcast(event: dict):
    dead = []
    for q in _sse_subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _sse_subscribers.remove(q)
        except ValueError:
            pass

def _emit(level: str, agent: str, message: str, extra: dict | None = None):
    event = {
        "ts":      datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
        "level":   level,
        "agent":   agent,
        "message": message,
        "extra":   extra or {},
    }
    asyncio.create_task(_broadcast(event))


# ── Test job management ────────────────────────────────────────────────────
# job_id → list of subscriber queues
_test_job_queues: dict[str, list[asyncio.Queue]] = {}


async def _broadcast_test(job_id: str, event: dict):
    """Push an event to all SSE subscribers for a given test job."""
    dead = []
    for q in _test_job_queues.get(job_id, []):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _test_job_queues[job_id].remove(q)
        except ValueError:
            pass


async def _run_test_job(job_id: str, cmd: list[str]):
    """Run the test subprocess and stream its output to SSE subscribers."""

    async def emit(line: str, kind: str = "output"):
        await _broadcast_test(job_id, {"type": kind, "line": line})

    script_path = Path(__file__).parent / "test_a2a_v2.py"
    if not script_path.exists():
        await emit(f"ERROR: test_a2a_v2.py not found at {script_path}", "error")
        await emit("", "done")
        return

    # Replace placeholder with resolved path
    resolved_cmd = [c if c != "__SCRIPT__" else str(script_path) for c in cmd]

    await emit(f"$ {' '.join(resolved_cmd)}", "start")
    await emit("", "output")

    try:
        proc = await asyncio.create_subprocess_exec(
            *resolved_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            await emit(line)

        await proc.wait()
        await emit("", "output")
        await emit(
            f"─── process exited with code {proc.returncode} ───",
            "done" if proc.returncode == 0 else "error",
        )

    except Exception as e:
        await emit(f"ERROR launching subprocess: {e}", "error")

    # Final sentinel so the SSE generator knows to close
    await _broadcast_test(job_id, {"type": "eof"})


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="A2A Demo UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_DIR = Path(__file__).parent / "ui"
if _UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = _UI_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h1>UI not found. Place ui/index.html next to ui_server.py</h1>",
            status_code=404,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def agent_status():
    """Ping all four agent cards and return health (used for header pills)."""
    agents = {
        "orchestrator": _OR_URL,
        "kb":           _KB_URL,
        "db":           _DB_URL,
        "llm":          _LLM_URL,
    }
    results = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in agents.items():
            try:
                r = await client.get(f"{url}/.well-known/agent.json")
                r.raise_for_status()
                card = r.json()
                results[name] = {
                    "status": "ok",
                    "name":   card.get("name", name),
                    "url":    url,
                    "skills": [s["id"] for s in card.get("skills", [])],
                }
            except Exception as e:
                results[name] = {"status": "error", "url": url, "error": str(e)}
    return JSONResponse(results)


@app.get("/api/agents")
async def agent_cards():
    """
    Return full agent card details for all four agents.
    Used by the Agents tab in the UI.
    """
    agents = [
        ("llm",          "LLM Agent",    _LLM_URL),
        ("orchestrator", "Orchestrator", _OR_URL),
        ("kb",           "KB Agent",     _KB_URL),
        ("db",           "DB Agent",     _DB_URL),
    ]

    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for key, label, url in agents:
            t0 = asyncio.get_event_loop().time()
            try:
                r    = await client.get(f"{url}/.well-known/agent.json")
                ping = round((asyncio.get_event_loop().time() - t0) * 1000)
                r.raise_for_status()
                card = r.json()
                results[key] = {
                    "status":       "ok",
                    "ping_ms":      ping,
                    "url":          url,
                    "name":         card.get("name", label),
                    "version":      card.get("version", "?"),
                    "description":  card.get("description", ""),
                    "skills":       card.get("skills", []),
                    "capabilities": card.get("capabilities", {}),
                }
            except Exception as e:
                ping = round((asyncio.get_event_loop().time() - t0) * 1000)
                results[key] = {
                    "status":  "error",
                    "ping_ms": ping,
                    "url":     url,
                    "error":   str(e),
                }

    return JSONResponse(results)


@app.post("/api/ask")
async def ask(request: Request):
    """Proxy a question to the orchestrator and emit SSE log events."""
    body     = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    task_id = str(uuid.uuid4())
    t_start = asyncio.get_event_loop().time()

    _emit("info",  "ui",           "New question received",  {"question": question})
    _emit("info",  "orchestrator", "Deciding routing…")

    payload = {
        "jsonrpc": "2.0",
        "id":      str(uuid.uuid4()),
        "method":  "tasks/send",
        "params":  {
            "id":      task_id,
            "message": {
                "role":  "user",
                "parts": [{"type": "text", "text": question}],
            },
            "metadata": body.get("metadata", {}),
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                _OR_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            rpc = resp.json()
    except Exception as e:
        _emit("error", "orchestrator", f"Request failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)

    elapsed = round((asyncio.get_event_loop().time() - t_start) * 1000)
    result  = rpc.get("result", {})
    status  = result.get("status", {})
    state   = status.get("state", "unknown")

    route         = "unknown"
    agent_answers = {}
    for art in result.get("artifacts", []):
        for part in art.get("parts", []):
            if part.get("type") == "data":
                d             = part.get("data", {})
                route         = d.get("route", route)
                agent_answers = d.get("agent_answers", agent_answers)

    _emit("route", "orchestrator", f"Routed to: {route.upper()}", {"route": route})

    for agent_name, answer_text in agent_answers.items():
        level = "success" if answer_text and "unreachable" not in answer_text.lower() else "error"
        _emit(level, agent_name,
              (answer_text or "")[:200] + ("…" if len(answer_text or "") > 200 else ""))

    answer_text = ""
    if status.get("message"):
        for p in status["message"].get("parts", []):
            if p.get("type") == "text":
                answer_text = p["text"]
                break

    _emit(
        "success" if state == "completed" else "error",
        "orchestrator",
        f"Task {state} in {elapsed}ms",
        {"elapsed_ms": elapsed},
    )

    return JSONResponse({
        "task_id":       task_id,
        "state":         state,
        "answer":        answer_text,
        "route":         route,
        "agent_answers": agent_answers,
        "elapsed_ms":    elapsed,
    })


@app.get("/api/logs")
async def log_stream(request: Request):
    """SSE stream of live orchestrator log events."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_subscribers.append(queue)

    async def generate() -> AsyncGenerator[str, None]:
        yield "event: connected\ndata: {}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            try:
                _sse_subscribers.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/api/tests/run")
async def tests_run(request: Request):
    """
    Launch test_a2a_v2.py as a subprocess.

    Body: { "suites": ["health", "llm", "kb", "db", "routing", "flow"] }
    Returns: { "job_id": "<uuid>" }

    Stream the output with GET /api/tests/output/{job_id}.
    """
    body   = await request.json()
    suites = body.get("suites", [])

    # Build the command — "__SCRIPT__" is replaced with the resolved path inside _run_test_job
    suite_flag_map = {
        "health":  "--health",
        "llm":     "--llm",
        "kb":      "--kb",
        "db":      "--db",
        "routing": "--routing",
        "flow":    "--flow",
    }
    cmd = [sys.executable, "__SCRIPT__"]
    for s in suites:
        if s in suite_flag_map:
            cmd.append(suite_flag_map[s])

    job_id = str(uuid.uuid4())
    _test_job_queues[job_id] = []

    log.info("Launching test job %s: %s", job_id, suites)
    asyncio.create_task(_run_test_job(job_id, cmd))

    return JSONResponse({"job_id": job_id})


@app.get("/api/tests/output/{job_id}")
async def tests_output_stream(job_id: str, request: Request):
    """SSE stream of test output lines for a given job."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    if job_id not in _test_job_queues:
        _test_job_queues[job_id] = []
    _test_job_queues[job_id].append(queue)

    async def generate() -> AsyncGenerator[str, None]:
        yield "event: connected\ndata: {}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "eof":
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            try:
                _test_job_queues[job_id].remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


if __name__ == "__main__":
    uvicorn.run("ui_server:app", host="0.0.0.0", port=_PORT, reload=False)
