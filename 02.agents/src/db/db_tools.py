# src/db/db_tools.py
"""
SAM-compatible DB Reader tool.
Uses sync psycopg2 (same pattern as rag_tools.py) — asyncpg conflicts
with SAM's internal async event loop.
"""
import os
import re
import json
import psycopg2
import psycopg2.extras
from typing import Any, Dict, Optional

DB_HOST = os.getenv("DB_HOST",     "localhost")
DB_PORT = os.getenv("DB_PORT",     "5432")
DB_NAME = os.getenv("DB_NAME",     "DEMODB")
DB_USER = os.getenv("DB_USER",     "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "postgres")

# Also support a full connection string
_DSN = os.getenv("DB_CONNECTION_STRING", "")

_DESTRUCTIVE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|EXEC|EXECUTE|COPY)\b",
    re.IGNORECASE,
)

def _get_conn():
    if _DSN:
        return psycopg2.connect(_DSN)
    return psycopg2.connect(
        host=DB_HOST, port=int(DB_PORT),
        dbname=DB_NAME, user=DB_USER, password=DB_PASS,
    )

def _extract_sql(text: str) -> str:
    """Strip <think> blocks, fences, and preamble — leave only the SELECT."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:sql)?\s*(.+?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    sel = re.search(r"(SELECT\b.+)", text, re.DOTALL | re.IGNORECASE)
    if sel:
        text = sel.group(1)
    if ";" in text:
        text = text[:text.index(";") + 1]
    return text.strip()

def _check_safe(sql: str) -> tuple:
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return False, "Only SELECT statements are permitted."
    if _DESTRUCTIVE.search(sql):
        return False, "Forbidden keyword — destructive SQL is not allowed."
    return True, ""

async def execute_sql_query(
    sql: str,
    tool_context: Optional[Any] = None,
    tool_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a read-only SQL SELECT against PostgreSQL and return results.

    Args:
        sql: A valid PostgreSQL SELECT statement.
             Double-quote customer_transactions columns: "Category", "TransactionAmount" etc.
             Prefix DEMODB schema objects: DEMODB.chunked_files
             customer_transactions needs no schema prefix. Add LIMIT 50.

    Returns:
        status, columns, rows (list of dicts), row_count
    """
    try:
        sql = _extract_sql(sql)
        safe, reason = _check_safe(sql)
        if not safe:
            return {"status": "error", "message": reason}

        conn = _get_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchmany(50)]
            columns = [d.name for d in cur.description] if cur.description else []
            cur.close()
        finally:
            conn.close()

        return {
            "status":    "success",
            "columns":   columns,
            "row_count": len(rows),
            "rows":      rows,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
