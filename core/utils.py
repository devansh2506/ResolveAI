"""Small shared helpers."""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # .env sits in the project root, one level above core/


def get_database_url() -> str:
    """Return DATABASE_URL from the environment, or raise a clear error."""
    # SECRET CONSUMED HERE: DATABASE_URL (never printed)
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    return url


def run_query(sql: str, params: tuple = ()) -> list[tuple]:
    """Run one statement on a fresh connection (committed and closed on exit); return its rows, if any."""
    with psycopg.connect(get_database_url(), connect_timeout=10) as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall() if cur.description else []
