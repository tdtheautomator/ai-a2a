# a2a/protocol.py
"""
A2A Protocol models — implements the Google A2A spec.
https://google.github.io/A2A/

Core flow:
  Client  →  POST /  {"jsonrpc":"2.0","method":"tasks/send","params":{"id":"<uuid>","message":{...}}}
  Agent   →  responds with Task object (completed | failed | working)
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import uuid


# ── Parts (content atoms inside a Message) ──────────────────────────────────

class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: Dict[str, Any]

Part = Union[TextPart, DataPart]


# ── Message ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: List[Part]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", parts=[TextPart(text=text)])

    @classmethod
    def agent_text(cls, text: str) -> "Message":
        return cls(role="agent", parts=[TextPart(text=text)])

    @classmethod
    def agent_data(cls, data: dict) -> "Message":
        return cls(role="agent", parts=[DataPart(data=data)])

    def text(self) -> str:
        """Extract first text part, or empty string."""
        for p in self.parts:
            if isinstance(p, TextPart):
                return p.text
        return ""


# ── Task status ───────────────────────────────────────────────────────────────

class TaskState(str, Enum):
    SUBMITTED       = "submitted"
    WORKING         = "working"
    INPUT_REQUIRED  = "input-required"
    COMPLETED       = "completed"
    FAILED          = "failed"
    CANCELED        = "canceled"


class TaskStatus(BaseModel):
    state:   TaskState
    message: Optional[Message] = None


# ── Artifact (structured output from agent) ──────────────────────────────────

class Artifact(BaseModel):
    name:  Optional[str] = None
    parts: List[Part]
    index: int = 0

    @classmethod
    def text(cls, content: str, name: str = "result") -> "Artifact":
        return cls(name=name, parts=[TextPart(text=content)])

    @classmethod
    def data(cls, content: dict, name: str = "result") -> "Artifact":
        return cls(name=name, parts=[DataPart(data=content)])


# ── Task ──────────────────────────────────────────────────────────────────────

class Task(BaseModel):
    id:        str = Field(default_factory=lambda: str(uuid.uuid4()))
    status:    TaskStatus
    history:   List[Message]   = []
    artifacts: List[Artifact]  = []
    metadata:  Dict[str, Any]  = {}


# ── JSON-RPC envelope ─────────────────────────────────────────────────────────

class SendTaskParams(BaseModel):
    id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    message:  Message
    metadata: Dict[str, Any] = {}

class GetTaskParams(BaseModel):
    id: str

class CancelTaskParams(BaseModel):
    id: str

class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id:      Union[str, int, None] = None
    method:  str
    params:  Dict[str, Any] = {}

class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id:      Union[str, int, None] = None
    result:  Optional[Any]  = None
    error:   Optional[Dict[str, Any]] = None

    @classmethod
    def ok(cls, req_id: Any, result: Any) -> "JsonRpcResponse":
        return cls(id=req_id, result=result)

    @classmethod
    def err(cls, req_id: Any, code: int, message: str) -> "JsonRpcResponse":
        return cls(id=req_id, error={"code": code, "message": message})


# ── Agent Card (/.well-known/agent.json) ─────────────────────────────────────

class AgentCapabilities(BaseModel):
    streaming:          bool = False
    pushNotifications:  bool = False
    stateTransitionHistory: bool = True

class AgentSkill(BaseModel):
    id:          str
    name:        str
    description: str
    examples:    List[str] = []

class AgentCard(BaseModel):
    name:         str
    description:  str
    url:          str
    version:      str = "1.0.0"
    capabilities: AgentCapabilities = AgentCapabilities()
    skills:       List[AgentSkill]  = []
    defaultInputModes:  List[str] = ["text"]
    defaultOutputModes: List[str] = ["text"]
