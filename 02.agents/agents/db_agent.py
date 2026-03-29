#!/usr/bin/env python3
# agents/db_agent.py
"""
DB Reader Agent — A2A server on port 8002.

Responsibility: Postgres query execution ONLY.
All LLM calls (SQL generation + result summarisation) are delegated
to llm_agent via A2A.

Flow:
  1. Receive natural-language question
  2. Ask llm_agent to generate SQL           (skill: generate_sql)
  3. Execute the SQL against Postgres
  4. Ask llm_agent to summarise the results  (skill: summarise_db)
  5. Return plain-English answer

Run:
    uvicorn agents.db_agent:app --port 8002 --reload
"""
import os
import sys
import re
import json
import time
import logging
import uvicorn
from typing import Any, Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from a2a.protocol import Message, AgentSkill, Task, TaskState
from a2a.client import A2AClient
from agents.base import BaseA2AAgent
from src.db.db_tools import execute_sql_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("db_agent")

_PORT        = int(os.getenv("DB_AGENT_PORT",              "8002"))
_URL         = os.getenv("DB_AGENT_URL",                   f"http://localhost:{_PORT}")
_LLM_URL     = os.getenv("LLM_AGENT_URL",                 "http://localhost:8003")
_LLM_TIMEOUT = float(os.getenv("AGENT_READ_TIMEOUT_SECS", "120.0"))

# ── Prompts owned by db_agent, sent to llm_agent as metadata ─────────────────
_SQL_GEN_SYSTEM = """
You are a Database Reader Agent. When a user asks a question about data,
you generate a PostgreSQL SELECT statement. Return ONLY the raw SQL —
no explanation, no markdown, no backticks, no <think> blocks.

DATABASE SCHEMA
---------------
Table: customer_transactions  (public schema - no prefix needed)
  "CustomerID"          BIGINT
  "Name"                TEXT
  "Surname"             TEXT
  "Gender"              TEXT
  "Birthdate"           TIMESTAMP
  "TransactionAmount"   DOUBLE PRECISION
  "Date"                TIMESTAMP
  "MerchantName"        TEXT
  "Category"            TEXT

SQL RULES
---------
1. Only SELECT - never INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE.
2. PostgreSQL is case-sensitive. ALWAYS double-quote every column from
   customer_transactions exactly as shown above.
3. DEMODB schema objects must be prefixed: DEMODB.chunked_files etc.
4. customer_transactions has NO schema prefix.
5. Always add LIMIT 50 unless the view already limits internally.
6. Return ONLY the raw SQL — no explanation, no markdown, no backticks.

EXAMPLES
--------
Q: average transaction amount by category
SQL: SELECT "Category", ROUND(AVG("TransactionAmount")::numeric, 2) AS avg_amount
     FROM customer_transactions GROUP BY "Category" ORDER BY avg_amount DESC LIMIT 50;
"""

_SUMMARISE_SYSTEM = (
    "You are a data analyst. Given a user question, a SQL query, and the query "
    "results, give a concise plain-English answer. "
    "Do not repeat the SQL. Do not dump raw JSON."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_sql(text: str) -> str:
    """Pull the first SELECT statement out of whatever llm_agent returned."""
    text = _THINK_RE.sub("", text)
    fence = re.search(r"```(?:sql)?\s*(.+?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    sel = re.search(r"(SELECT\b.+)", text, re.DOTALL | re.IGNORECASE)
    if sel:
        text = sel.group(1)
    if ";" in text:
        text = text[:text.index(";") + 1]
    return text.strip()


class DBReaderAgent(BaseA2AAgent):

    def __init__(self):
        super().__init__(
            name        = "DB Reader Agent",
            description = (
                "Translates natural-language questions into PostgreSQL SELECT "
                "statements via llm_agent, executes them, and returns "
                "plain-English answers via llm_agent."
            ),
            url    = _URL,
            skills = [
                AgentSkill(
                    id          = "query_db",
                    name        = "Query Database",
                    description = "NL → SQL (via llm_agent) → execute → summarise (via llm_agent).",
                    examples    = [
                        "How many transactions are there?",
                        "Top 5 spending categories?",
                        "Average transaction amount per gender?",
                        "Show me agent processing summary.",
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
        log.info("DB Agent init — url=%s  llm_agent=%s", _URL, _LLM_URL)

    async def _call_llm(
        self,
        user_content: str,
        skill:        str,
        max_tokens:   int,
        task_id:      str,
    ) -> tuple[bool, str]:
        """
        Delegate one LLM call to llm_agent.
        Returns (success: bool, text: str).
        """
        t0 = time.monotonic()
        log.info("[DB-LLM] Delegating — skill=%s  user_chars=%d", skill, len(user_content))

        llm_task = await self._llm_client.send_task(
            message  = user_content,
            metadata = {
                "system":     _SQL_GEN_SYSTEM if skill == "generate_sql" else _SUMMARISE_SYSTEM,
                "max_tokens": max_tokens,
                "skill":      skill,
            },
        )
        ms = round((time.monotonic() - t0) * 1000)

        if llm_task.status.state == TaskState.COMPLETED:
            text = llm_task.status.message.text() if llm_task.status.message else ""
            log.info("[DB-LLM] Done — skill=%s  elapsed=%d ms  chars=%d", skill, ms, len(text))
            return True, text
        else:
            err = llm_task.status.message.text() if llm_task.status.message else "unknown"
            log.error("[DB-LLM] FAILED — skill=%s  elapsed=%d ms  error=%s", skill, ms, err)
            return False, err

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
        log.info("[DB-TASK START]  id=%s  question=%r", task_id, question[:120])

        # ── Step 1: generate SQL via llm_agent ────────────────────────────────
        t_sql = time.monotonic()
        ok, raw_sql = await self._call_llm(
            user_content = question,
            skill        = "generate_sql",
            max_tokens   = 1000,
            task_id      = task_id,
        )
        sql_ms = round((time.monotonic() - t_sql) * 1000)

        if not ok:
            return self.failed(task_id, message, f"SQL generation failed: {raw_sql}")

        sql = _extract_sql(raw_sql)
        log.info("[DB-SQL-GEN] elapsed=%d ms  sql=%s", sql_ms, sql[:300])

        if not sql.upper().startswith("SELECT"):
            log.error("[DB-SQL-GEN] Non-SELECT returned: %r", sql[:120])
            return self.failed(
                task_id, message,
                f"LLM did not return a valid SELECT statement. Got: {sql[:120]!r}",
            )

        # ── Step 2: execute SQL ───────────────────────────────────────────────
        log.info("[DB-EXEC] Executing SQL against Postgres")
        t_exec    = time.monotonic()
        db_result = await execute_sql_query(sql=sql)
        exec_ms   = round((time.monotonic() - t_exec) * 1000)

        if db_result["status"] == "error":
            log.error("[DB-EXEC] DB error after %d ms: %s", exec_ms, db_result["message"])
            return self.failed(task_id, message, f"DB error: {db_result['message']}")

        log.info("[DB-EXEC] Done — elapsed=%d ms  rows=%d", exec_ms, db_result["row_count"])

        # ── Step 3: summarise results via llm_agent ────────────────────────────
        payload      = json.dumps(db_result, indent=2, default=str)
        user_content = (
            f"Question: {question}\n\n"
            f"SQL:\n{sql}\n\n"
            f"Result:\n{payload}"
        )
        t_sum = time.monotonic()
        ok, answer = await self._call_llm(
            user_content = user_content,
            skill        = "summarise_db",
            max_tokens   = 1000,
            task_id      = task_id,
        )
        sum_ms = round((time.monotonic() - t_sum) * 1000)

        if not ok:
            log.warning(
                "[DB-SUMMARISE] llm_agent failed after %d ms — falling back to raw rows", sum_ms
            )
            answer = (
                f"Query returned {db_result['row_count']} row(s).\n\n"
                + json.dumps(db_result["rows"], indent=2, default=str)
            )

        elapsed = round((time.monotonic() - t0) * 1000)
        log.info(
            "[DB-TASK DONE]  id=%s  elapsed=%d ms  "
            "(sql_gen=%d ms  db_exec=%d ms  summarise=%d ms)",
            task_id, elapsed, sql_ms, exec_ms, sum_ms,
        )

        meta = {"sql": sql, "row_count": db_result["row_count"], "rows": db_result["rows"]}
        return self.completed(task_id, message, answer, data=meta)


agent = DBReaderAgent()
app   = agent.app

if __name__ == "__main__":
    uvicorn.run("agents.db_agent:app", host="0.0.0.0", port=_PORT, reload=False)
