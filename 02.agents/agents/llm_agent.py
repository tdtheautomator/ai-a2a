#!/usr/bin/env python3
# agents/llm_agent.py
"""
LLM Gateway Agent — A2A server on port 8003.

Fixes vs previous version
--------------------------
EMPTY REPLY (reasoning models):
  Qwen3 and other reasoning models wrap ALL output in <think>…</think>.
  With low max_tokens (e.g. 10) the model spends every token on thinking
  and returns empty visible content after the think-block is stripped.

  Fix 1 — skill-aware token budgets:
    route_question uses max_tokens=256 (enough for a think-block + "kb")
    All other skills keep their caller-supplied or default value.

  Fix 2 — empty-result retry with 2× tokens:
    If stripping think-blocks leaves an empty string AND the finish_reason
    is "length" (truncated), we retry once with doubled max_tokens.

  Fix 3 — think-only fallback:
    If result is still empty after retry, extract the last word from the
    think-block itself (works for routing decisions where the model
    reasons to an answer but never writes it outside the block).

TIMEOUT:
  _LLM_READ_TIMEOUT default raised from 120 s → 300 s (5 min).
  50 s was observed for a 3-token reply on CPU; a 512-token SQL-gen
  or summarise call can easily take 150–250 s on CPU-only Ollama.
  Override with LLM_READ_TIMEOUT_SECS env var.
"""
import os
import sys
import re
import time
import logging
import uvicorn
from typing import Any, Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import httpx
from openai import AsyncOpenAI
from a2a.protocol import Message, AgentSkill, Task
from agents.base import BaseA2AAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("llm_agent")

# ── Config ────────────────────────────────────────────────────────────────────
_PORT  = int(os.getenv("LLM_AGENT_PORT",     "8003"))
_URL   = os.getenv("LLM_AGENT_URL",          f"http://localhost:{_PORT}")
_HOST  = os.getenv("LITELLM_HOST",           "localhost")
_LPORT = os.getenv("LITELLM_PORT",           "4000")
_KEY   = os.getenv("LITELLM_API_KEY",        "sk-1234")
_MODEL = os.getenv("LITELLM_CHAT_MODEL",     "gpt-oss")
_BASE  = f"http://{_HOST}:{_LPORT}"

# Raised to 300 s: CPU-only Ollama takes ~50 s per 3 tokens.
# A 512-token completion can take 4–8 min on CPU.
# Set LLM_READ_TIMEOUT_SECS to override.
_LLM_READ_TIMEOUT = float(os.getenv("LLM_READ_TIMEOUT_SECS", "300.0"))

# Minimum token budget for the routing skill.
# Reasoning models need room to write a <think> block AND then output the
# answer word.  10 tokens is nowhere near enough — 256 is safe.
_ROUTE_MIN_TOKENS = int(os.getenv("ROUTE_MIN_TOKENS", "256"))

_THINK_RE      = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_STRIP_RE = re.compile(r"<think>.*?</think>",  re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks and normalise whitespace."""
    return _THINK_STRIP_RE.sub("", text).strip()


def _last_word_from_think(text: str) -> str:
    """
    Fallback for reasoning models that reason to an answer inside the think
    block but never write it outside.  Extracts the last meaningful word
    from the final think block — for routing this is typically 'kb'/'db'/'both'.
    """
    match = _THINK_RE.findall(text)
    if not match:
        return ""
    # Take the last think block, split on whitespace, return last token
    words = match[-1].strip().split()
    return words[-1].strip(".,;:\"'").lower() if words else ""


class LLMGatewayAgent(BaseA2AAgent):

    def __init__(self):
        super().__init__(
            name        = "LLM Gateway Agent",
            description = (
                "Single-responsibility LLM gateway. Accepts a user prompt and "
                "system prompt via A2A metadata and returns the model completion. "
                "All other agents delegate LLM calls here."
            ),
            url    = _URL,
            skills = [
                AgentSkill(
                    id          = "route_question",
                    name        = "Route Question",
                    description = "Decide which data agent(s) should handle a question.",
                    examples    = ["What is HAL?", "How many transactions?"],
                ),
                AgentSkill(
                    id          = "synthesise_kb",
                    name        = "Synthesise KB Answer",
                    description = "Generate a plain-English answer from Qdrant search hits.",
                    examples    = ["[Qdrant hits JSON] → answer with sources"],
                ),
                AgentSkill(
                    id          = "generate_sql",
                    name        = "Generate SQL",
                    description = "Translate a natural-language question into a PostgreSQL SELECT.",
                    examples    = ["Top 5 spending categories → SELECT ..."],
                ),
                AgentSkill(
                    id          = "summarise_db",
                    name        = "Summarise DB Result",
                    description = "Summarise SQL query results in plain English.",
                    examples    = ["[SQL + rows JSON] → concise answer"],
                ),
                AgentSkill(
                    id          = "synthesise_final",
                    name        = "Synthesise Final Answer",
                    description = "Blend answers from KB and/or DB agents into one response.",
                    examples    = ["[KB answer + DB answer] → unified response"],
                ),
            ],
        )

        self._llm = AsyncOpenAI(
            api_key  = _KEY,
            base_url = _BASE,
            timeout  = httpx.Timeout(
                connect = 10.0,
                read    = _LLM_READ_TIMEOUT,
                write   = 15.0,
                pool    = 10.0,
            ),
        )

        log.info(
            "LLM Agent init — url=%s  litellm=%s  model=%s  "
            "read_timeout=%.0fs  route_min_tokens=%d",
            _URL, _BASE, _MODEL, _LLM_READ_TIMEOUT, _ROUTE_MIN_TOKENS,
        )

    async def _complete(
        self,
        system:     str,
        user:       str,
        max_tokens: int,
        task_id:    str,
        skill:      str,
    ) -> tuple[str, str, object]:
        """
        Single LLM call.  Returns (raw_text, finish_reason, usage).
        Raises on network / API errors — caller handles.
        """
        resp = await self._llm.chat.completions.create(
            model      = _MODEL,
            max_tokens = max_tokens,
            messages   = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        raw    = (resp.choices[0].message.content or "").strip()
        finish = resp.choices[0].finish_reason
        usage  = resp.usage
        log.debug(
            "[LLM-RAW]  task=%s  skill=%s  finish=%s  raw_chars=%d  "
            "prompt_tok=%s  completion_tok=%s",
            task_id, skill, finish,
            len(raw),
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
        )
        return raw, finish, usage

    async def handle_task(
        self,
        task_id:  str,
        message:  Message,
        metadata: Dict[str, Any],
    ) -> Task:
        user_content = message.text().strip()
        if not user_content:
            return self.failed(task_id, message, "Empty user content.")

        system_prompt = metadata.get("system", "You are a helpful assistant.")
        skill_label   = metadata.get("skill", "completion")
        caller_tokens = int(metadata.get("max_tokens", 512))

        # For routing we enforce a minimum so reasoning models have room
        # to finish their think-block before outputting the answer word.
        if skill_label == "route_question":
            max_tokens = max(caller_tokens, _ROUTE_MIN_TOKENS)
        else:
            max_tokens = caller_tokens

        t0 = time.monotonic()
        log.info(
            "[LLM-CALL]  task=%s  skill=%s  model=%s  max_tokens=%d  "
            "system_chars=%d  user_chars=%d",
            task_id, skill_label, _MODEL, max_tokens,
            len(system_prompt), len(user_content),
        )

        try:
            raw, finish, usage = await self._complete(
                system_prompt, user_content, max_tokens, task_id, skill_label
            )
            result = _strip_think(raw)
            ms     = round((time.monotonic() - t0) * 1000)

            # ── Empty-result handling for reasoning models ────────────────────
            if not result:
                log.warning(
                    "[LLM-EMPTY]  task=%s  skill=%s  elapsed=%d ms  "
                    "finish=%s  raw_chars=%d  "
                    "Model appears to be reasoning-only (all output in <think> blocks).",
                    task_id, skill_label, ms, finish, len(raw),
                )

                # Strategy 1: retry with 2× tokens if we were truncated
                if finish == "length":
                    retry_tokens = max_tokens * 2
                    log.info(
                        "[LLM-RETRY]  task=%s  skill=%s  "
                        "Retrying with max_tokens=%d (was truncated)",
                        task_id, skill_label, retry_tokens,
                    )
                    try:
                        raw2, finish2, usage2 = await self._complete(
                            system_prompt, user_content,
                            retry_tokens, task_id, skill_label,
                        )
                        result = _strip_think(raw2)
                        ms     = round((time.monotonic() - t0) * 1000)
                        usage  = usage2
                        finish = finish2
                        if result:
                            log.info(
                                "[LLM-RETRY-OK]  task=%s  skill=%s  "
                                "elapsed=%d ms  chars=%d",
                                task_id, skill_label, ms, len(result),
                            )
                    except Exception as retry_err:
                        log.warning(
                            "[LLM-RETRY-FAIL]  task=%s  skill=%s  error=%s",
                            task_id, skill_label, retry_err,
                        )

                # Strategy 2: extract last word from think-block
                # Works for routing (model reasons to "kb"/"db"/"both" inside think)
                if not result:
                    fallback = _last_word_from_think(raw)
                    if fallback:
                        log.warning(
                            "[LLM-THINK-FALLBACK]  task=%s  skill=%s  "
                            "Extracted last word from think-block: %r",
                            task_id, skill_label, fallback,
                        )
                        result = fallback
                    else:
                        log.error(
                            "[LLM-EMPTY-FINAL]  task=%s  skill=%s  "
                            "No content after all recovery strategies.  raw=%r",
                            task_id, skill_label, raw[:200],
                        )
                        return self.failed(
                            task_id, message,
                            f"Model returned empty content for skill={skill_label}. "
                            f"finish_reason={finish}. "
                            f"The model may need a larger max_tokens budget. "
                            f"raw_chars={len(raw)}",
                        )

            log.info(
                "[LLM-DONE]  task=%s  skill=%s  elapsed=%d ms  finish=%s  "
                "prompt_tok=%s  completion_tok=%s  output_chars=%d",
                task_id, skill_label, ms, finish,
                usage.prompt_tokens if usage else "?",
                usage.completion_tokens if usage else "?",
                len(result),
            )
            return self.completed(task_id, message, result)

        except Exception as e:
            ms = round((time.monotonic() - t0) * 1000)
            log.error(
                "[LLM-FAIL]  task=%s  skill=%s  elapsed=%d ms  %s: %s  "
                "Check: LiteLLM at %s  Ollama running?",
                task_id, skill_label, ms, type(e).__name__, e, _BASE,
            )
            return self.failed(task_id, message, f"LLM call failed ({type(e).__name__}): {e}")


# ── FastAPI app ───────────────────────────────────────────────────────────────
agent = LLMGatewayAgent()
app   = agent.app

if __name__ == "__main__":
    uvicorn.run("agents.llm_agent:app", host="0.0.0.0", port=_PORT, reload=False)
