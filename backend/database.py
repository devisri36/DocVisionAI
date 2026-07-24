import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from configs.config_loader import ConfigManager
from utils.logger import setup_logger

logger = setup_logger("database")

def get_db_path() -> Path:
    """Returns the database file path under outputs folder."""
    config = ConfigManager().config
    outputs_dir = Path(config.paths.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir / "docvision.db"

def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema and sets up tables if they do not exist."""
    db_path = get_db_path()
    logger.info(f"Initializing SQLite database at {db_path}...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create history logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                filename TEXT NOT NULL,
                category TEXT NOT NULL,
                fraud_score REAL NOT NULL,
                authenticity_score REAL NOT NULL,
                confidence_score REAL NOT NULL,
                extracted_fields TEXT NOT NULL, -- JSON string representation
                ocr_text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create uploads/files table to reference uploaded document paths
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info("Database schemas verified and initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database tables: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()

# --- DATABASE READ / WRITE HELPERS ---

def register_user(username: str, password_hash: str) -> bool:
    """Inserts a new user into the database. Returns True if successful, False if username exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"Registration failed: username '{username}' already exists.")
        return False
    finally:
        conn.close()

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves user info dictionary by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()

def register_file(file_id: str, filename: str, filepath: str, username: str) -> bool:
    """Logs an uploaded document file record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO files (file_id, filename, filepath, username) VALUES (?, ?, ?, ?)",
            (file_id, filename, str(filepath), username)
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to register uploaded file record: {e}")
        return False
    finally:
        conn.close()

def get_file_record(file_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves filepath record by unique file_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM files WHERE file_id = ?", (file_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()

def log_history_entry(
    username: str,
    filename: str,
    category: str,
    fraud_score: float,
    authenticity_score: float,
    confidence_score: float,
    extracted_fields: Dict[str, Any],
    ocr_text: str
) -> bool:
    """Logs the results of a document analysis run into the history table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        fields_json = json.dumps(extracted_fields)
        cursor.execute(
            """
            INSERT INTO history (
                username, filename, category, fraud_score, authenticity_score, confidence_score, extracted_fields, ocr_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (username, filename, category, fraud_score, authenticity_score, confidence_score, fields_json, ocr_text)
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to log verification history entry: {e}")
        return False
    finally:
        conn.close()

def fetch_history_entries(username: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches list of historical logs, optionally filtered by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if username:
            cursor.execute("SELECT * FROM history WHERE username = ? ORDER BY timestamp DESC", (username,))
        else:
            cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
            
        rows = cursor.fetchall()
        entries = []
        for r in rows:
            d = dict(r)
            try:
                d["extracted_fields"] = json.loads(d["extracted_fields"])
            except Exception:
                d["extracted_fields"] = {}
            entries.append(d)
        return entries
    finally:
        conn.close()
