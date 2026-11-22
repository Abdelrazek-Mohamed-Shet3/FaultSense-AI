"""SQLite persistence layer for FaultSense inspection history."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class InspectionDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    label_en TEXT NOT NULL,
                    label_ar TEXT NOT NULL,
                    severity_level TEXT NOT NULL,
                    severity_score INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    is_fault INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_inspection(self, filename: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        enriched = {**payload, "created_at": created_at, "filename": filename}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO inspections (
                    id, created_at, filename, media_type, domain, predicted_class,
                    label_en, label_ar, severity_level, severity_score, confidence,
                    is_fault, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    enriched["id"],
                    created_at,
                    filename,
                    enriched["media_type"],
                    enriched["domain"],
                    enriched["predicted_class"],
                    enriched["label_en"],
                    enriched["label_ar"],
                    enriched["severity_level"],
                    int(enriched["severity_score"]),
                    float(enriched["confidence"]),
                    1 if enriched["is_fault"] else 0,
                    json.dumps(enriched),
                ),
            )
            conn.commit()
        return enriched

    def list_inspections(
        self,
        domain: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT payload_json FROM inspections WHERE 1=1"
        params: List[Any] = []
        if domain and domain != "all":
            query += " AND domain = ?"
            params.append(domain)
        if severity and severity != "all":
            query += " AND severity_level = ?"
            params.append(severity)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def get_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM inspections WHERE id = ?",
                (inspection_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def delete_inspection(self, inspection_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
            conn.commit()
            return cur.rowcount > 0
