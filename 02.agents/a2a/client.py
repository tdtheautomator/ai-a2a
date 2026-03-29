# a2a/client.py
"""
Async A2A client.
Sends tasks to a remote A2A-compliant agent and returns the Task.

Changes vs original:
  - Per-request structured timing logs (connect, wait, total).
  - Configurable timeouts exposed as constructor args (defaults raised to
    support slow local Ollama models).
  - Startup probe: can_reach() does a lightweight GET on the agent card
    and logs latency — call this at agent init time to surface DNS/network
    failures before the first real request.
  - Detailed error messages distinguish connect vs read vs HTTP vs parse
    failures so the log immediately shows *which* phase timed out.
"""
from __future__ import annotations
import uuid
import time
import logging
import httpx
from typing import Any, Dict, Optional

from .protocol import (
    Task, Message, TaskStatus, TaskState,
    JsonRpcRequest, JsonRpcResponse, SendTaskParams,
    Artifact, TextPart,
)

log = logging.getLogger(__name__)


class A2AClient:
    """
    Async HTTP client for the A2A JSON-RPC protocol.

    Reuses a single httpx.AsyncClient across calls so TCP connections
    are pooled rather than opened/closed on every request.
    """

    def __init__(
        self,
        base_url:        str,
        timeout:         float = 120.0,   # read / total  — raised for slow Ollama models
        connect_timeout: float = 10.0,    # TCP connect   — raised for cross-compose DNS
        write_timeout:   float = 15.0,
        pool_timeout:    float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._label   = base_url  # shown in log lines

        _timeout = httpx.Timeout(
            connect = connect_timeout,
            read    = timeout,
            write   = write_timeout,
            pool    = pool_timeout,
        )

        self._http = httpx.AsyncClient(
            timeout = _timeout,
            limits  = httpx.Limits(
                max_keepalive_connections = 10,
                max_connections           = 20,
            ),
        )

        log.debug(
            "A2AClient created — url=%s  read_timeout=%.1fs  connect_timeout=%.1fs",
            self.base_url, timeout, connect_timeout,
        )

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "A2AClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── Startup probe ─────────────────────────────────────────────────────────

    async def can_reach(self) -> bool:
        """
        Lightweight reachability check — fetches the agent card.
        Returns True on success. Logs the outcome either way.
        Call this at startup to surface DNS / firewall / port issues early.
        """
        t0 = time.monotonic()
        try:
            resp = await self._http.get(
                f"{self.base_url}/.well-known/agent.json"
            )
            resp.raise_for_status()
            card = resp.json()
            ms = round((time.monotonic() - t0) * 1000)
            log.info(
                "PROBE OK  url=%-40s  agent=%r  latency=%d ms",
                self.base_url, card.get("name", "?"), ms,
            )
            return True
        except Exception as e:
            ms = round((time.monotonic() - t0) * 1000)
            log.error(
                "PROBE FAIL  url=%-40s  error=%s  after=%d ms",
                self.base_url, e, ms,
            )
            return False

    # ── Core API ──────────────────────────────────────────────────────────────

    async def send_task(
        self,
        message:  str | Message,
        task_id:  Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Send a task to the remote agent and return the resulting Task.
        Always returns a Task — network/parse errors become FAILED Tasks.
        """
        if metadata is None:
            metadata = {}

        if isinstance(message, str):
            message = Message.user(message)

        params = SendTaskParams(
            id       = task_id or str(uuid.uuid4()),
            message  = message,
            metadata = metadata,
        )
        req = JsonRpcRequest(
            id     = str(uuid.uuid4()),
            method = "tasks/send",
            params = params.model_dump(),
        )

        question_preview = message.text()[:80].replace("\n", " ")
        log.info(
            "SEND  url=%-40s  task_id=%s  q=%r",
            self.base_url, params.id, question_preview,
        )

        t0 = time.monotonic()

        # ── Network call ──────────────────────────────────────────────────────
        try:
            resp = await self._http.post(
                f"{self.base_url}/",
                json    = req.model_dump(),
                headers = {"Content-Type": "application/json"},
            )
            http_ms = round((time.monotonic() - t0) * 1000)
            log.info(
                "RECV  url=%-40s  task_id=%s  status=%d  elapsed=%d ms",
                self.base_url, params.id, resp.status_code, http_ms,
            )
            resp.raise_for_status()

        except httpx.ConnectError as e:
            ms = round((time.monotonic() - t0) * 1000)
            reason = (
                f"Cannot connect to {self.base_url} after {ms} ms — "
                f"check container name, network membership, and port. "
                f"Detail: {e}"
            )
            log.error("CONNECT_FAIL  url=%s  error=%s  elapsed=%d ms", self.base_url, e, ms)
            return self._error_task(params.id, message, reason)

        except httpx.TimeoutException as e:
            ms = round((time.monotonic() - t0) * 1000)
            phase = type(e).__name__  # ConnectTimeout / ReadTimeout / WriteTimeout / PoolTimeout
            reason = (
                f"Timeout ({phase}) calling {self.base_url} after {ms} ms. "
                f"The model inference inside the agent is likely slow. "
                f"Raise the read_timeout on A2AClient or add capacity. "
                f"Detail: {e}"
            )
            log.error(
                "TIMEOUT  url=%s  phase=%s  elapsed=%d ms",
                self.base_url, phase, ms,
            )
            return self._error_task(params.id, message, reason)

        except httpx.HTTPStatusError as e:
            ms = round((time.monotonic() - t0) * 1000)
            reason = f"HTTP {e.response.status_code} from {self.base_url}: {e.response.text[:200]}"
            log.error("HTTP_ERR  url=%s  status=%d  elapsed=%d ms  body=%s",
                      self.base_url, e.response.status_code, ms, e.response.text[:200])
            return self._error_task(params.id, message, reason)

        except httpx.RequestError as e:
            ms = round((time.monotonic() - t0) * 1000)
            reason = f"Connection error calling {self.base_url} after {ms} ms: {e}"
            log.error("REQUEST_ERR  url=%s  error=%s  elapsed=%d ms", self.base_url, e, ms)
            return self._error_task(params.id, message, reason)

        total_ms = round((time.monotonic() - t0) * 1000)

        # ── Parse JSON-RPC envelope ───────────────────────────────────────────
        try:
            rpc = JsonRpcResponse(**resp.json())
        except Exception as e:
            log.error("PARSE_FAIL  url=%s  error=%s  body=%s", self.base_url, e, resp.text[:300])
            return self._error_task(params.id, message, f"Invalid JSON-RPC response: {e}")

        if rpc.error:
            msg = rpc.error.get("message", "Unknown RPC error")
            log.error("RPC_ERR  url=%s  error=%s", self.base_url, msg)
            return self._error_task(params.id, message, msg)

        if rpc.result is None:
            log.error("RPC_EMPTY  url=%s", self.base_url)
            return self._error_task(params.id, message, "Agent returned an empty result")

        # ── Deserialise Task ──────────────────────────────────────────────────
        try:
            task = Task.model_validate(rpc.result)
        except Exception as e:
            log.error("DESER_FAIL  url=%s  error=%s", self.base_url, e)
            return self._error_task(params.id, message, f"Response parse error: {e}")

        log.info(
            "DONE  url=%-40s  task_id=%s  state=%s  total=%d ms",
            self.base_url, params.id, task.status.state.value, total_ms,
        )
        return task

    async def get_agent_card(self) -> dict:
        resp = await self._http.get(f"{self.base_url}/.well-known/agent.json")
        resp.raise_for_status()
        return resp.json()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _error_task(task_id: str, message: Message, reason: str) -> Task:
        return Task(
            id     = task_id,
            status = TaskStatus(
                state   = TaskState.FAILED,
                message = Message.agent_text(reason),
            ),
            history   = [message],
            artifacts = [Artifact(parts=[TextPart(text=reason)])],
        )
