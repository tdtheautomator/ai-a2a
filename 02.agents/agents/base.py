# agents/base.py
"""
BaseA2AAgent — a FastAPI app that implements the A2A protocol.
Subclass it and override `handle_task()`.

Optimisations over the original:
  - model_dump(mode="json") ensures TaskState enums serialise as their
    string values ("completed") rather than Python enum objects, which
    would cause JSONResponse to raise a serialisation error.
  - log.exception() in _handle_send prints the full traceback to the
    sub-agent's console so the real error is always visible.
  - Optional type annotation on skills parameter for cleaner imports.
"""
from __future__ import annotations
import uuid
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from a2a.protocol import (
    Task, Message, TaskStatus, TaskState,
    Artifact,
    AgentCard, AgentCapabilities, AgentSkill,
    JsonRpcRequest, JsonRpcResponse, SendTaskParams,
)

log = logging.getLogger(__name__)


class BaseA2AAgent:
    """
    Minimal A2A server. Subclasses implement handle_task().
    Exposes:
      GET  /.well-known/agent.json   — Agent Card
      POST /                         — JSON-RPC endpoint
    """

    def __init__(
        self,
        name:        str,
        description: str,
        url:         str,
        skills:      Optional[List[AgentSkill]] = None,
    ):
        self.card = AgentCard(
            name         = name,
            description  = description,
            url          = url,
            capabilities = AgentCapabilities(),
            skills       = skills or [],
        )
        self.app = FastAPI(title=name)
        self._register_routes()

    # ── Route registration ────────────────────────────────────────────────────

    def _register_routes(self):
        app = self.app

        @app.get("/.well-known/agent.json")
        async def agent_card():
            return self.card.model_dump()

        @app.post("/")
        async def jsonrpc_endpoint(request: Request):
            try:
                body = await request.json()
                rpc  = JsonRpcRequest(**body)
            except Exception as e:
                return JSONResponse(
                    JsonRpcResponse.err(None, -32700, f"Parse error: {e}").model_dump()
                )

            if rpc.method == "tasks/send":
                return await self._handle_send(rpc)
            else:
                return JSONResponse(
                    JsonRpcResponse.err(
                        rpc.id, -32601, f"Method not found: {rpc.method}"
                    ).model_dump()
                )

    # ── Internal send handler ─────────────────────────────────────────────────

    async def _handle_send(self, rpc: JsonRpcRequest) -> JSONResponse:
        task_id = rpc.params.get("id", str(uuid.uuid4()))
        try:
            params  = SendTaskParams(**rpc.params)
            message = params.message
            task    = await self.handle_task(params.id, message, params.metadata)

            # model_dump(mode="json") serialises enums as their string values
            # (e.g. "completed") rather than Python enum objects.
            # Without mode="json", JSONResponse raises a serialisation error
            # when it encounters TaskState.COMPLETED as a Python enum.
            return JSONResponse(
                JsonRpcResponse.ok(rpc.id, task.model_dump(mode="json")).model_dump()
            )

        except Exception as e:
            # log.exception prints the full traceback — critical for debugging
            # because the short str(e) in the response body loses the context
            # of where inside handle_task the failure occurred.
            log.exception("handle_task raised an unhandled exception (task_id=%s)", task_id)
            task = Task(
                id     = task_id,
                status = TaskStatus(
                    state   = TaskState.FAILED,
                    message = Message.agent_text(f"Internal error: {e}"),
                ),
            )
            return JSONResponse(
                JsonRpcResponse.ok(rpc.id, task.model_dump(mode="json")).model_dump()
            )

    # ── Subclass contract ─────────────────────────────────────────────────────

    @abstractmethod
    async def handle_task(
        self,
        task_id:  str,
        message:  Message,
        metadata: Dict[str, Any],
    ) -> Task:
        """Process one task. Must return a completed or failed Task."""
        ...

    # ── Helper builders ───────────────────────────────────────────────────────

    def completed(
        self,
        task_id:  str,
        user_msg: Message,
        text:     str,
        data:     Optional[dict] = None,
    ) -> Task:
        artifacts = [Artifact.text(text)]
        if data:
            artifacts.append(Artifact.data(data, name="raw_data"))
        return Task(
            id        = task_id,
            status    = TaskStatus(
                state   = TaskState.COMPLETED,
                message = Message.agent_text(text),
            ),
            history   = [user_msg],
            artifacts = artifacts,
        )

    def failed(
        self,
        task_id:  str,
        user_msg: Message,
        reason:   str,
    ) -> Task:
        return Task(
            id     = task_id,
            status = TaskStatus(
                state   = TaskState.FAILED,
                message = Message.agent_text(reason),
            ),
            history = [user_msg],
        )
