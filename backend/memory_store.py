# backend/memory_store.py
"""
Persistent Session Storage using SQLite
Replaces in-memory dictionary to prevent data loss on server restart.
"""
import os
import json
import sqlite3
import logging
from typing import Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database file path - configurable via environment variable
DB_PATH = os.getenv("SESSION_DB_PATH", os.path.join(os.path.dirname(__file__), "sessions.db"))


@contextmanager
def get_db_connection():
    """Context manager for database connections with proper cleanup."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _init_db():
    """Initialize the database schema if it doesn't exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                entities TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info(f"Session database initialized at: {DB_PATH}")


# Initialize database on module import
_init_db()


def _get_default_state() -> Dict[str, Any]:
    """Returns the default state for a new session."""
    return {
        "entities": {
            "income_monthly": None,
            "loan_amount": None,
            "tenure_months": None,
            "credit_score": None,
            "age": None
        },
        "summary": "Conversation started.\n"
    }


def load(session_id: str) -> Dict[str, Any]:
    """
    Load session state from SQLite database.
    Creates a new session with default state if not found.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT entities, summary FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    "entities": json.loads(row["entities"]),
                    "summary": row["summary"]
                }
            
            # Create new session with default state
            default_state = _get_default_state()
            conn.execute(
                "INSERT INTO sessions (session_id, entities, summary) VALUES (?, ?, ?)",
                (session_id, json.dumps(default_state["entities"]), default_state["summary"])
            )
            conn.commit()
            logger.info(f"New session created: {session_id}")
            return default_state
            
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}", exc_info=True)
        return _get_default_state()


def save(session_id: str, state: Dict[str, Any], new_user_msg: str = "", new_ai_msg: str = ""):
    """
    Save session state to SQLite database.
    Updates the state and maintains a running chat history summary.
    """
    try:
        if new_user_msg and new_ai_msg:
            # Keep a running log of the last few turns for Summary Memory
            current_summary = state.get("summary", "")
            updated_summary = current_summary + f"User: {new_user_msg}\nBank: {new_ai_msg}\n"
            
            # Prevent memory from blowing up by keeping only the last 1000 characters
            if len(updated_summary) > 1000:
                updated_summary = "..." + updated_summary[-1000:]
                
            state["summary"] = updated_summary
        
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, entities, summary, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    entities = excluded.entities,
                    summary = excluded.summary,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, json.dumps(state["entities"]), state["summary"])
            )
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}", exc_info=True)


def delete(session_id: str) -> bool:
    """Delete a session from the database."""
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            logger.info(f"Session deleted: {session_id}")
            return True
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        return False


def list_sessions() -> list:
    """List all active sessions (for admin/debugging purposes)."""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        return []
