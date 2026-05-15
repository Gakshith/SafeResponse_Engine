from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class APIStateStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(
            db_path or os.getenv("SAFE_RESPONSE_STATE_DB", "artifacts/api_state.sqlite3")
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    request_id TEXT,
                    decision TEXT NOT NULL,
                    latency_seconds REAL NOT NULL,
                    risk_score REAL,
                    success INTEGER NOT NULL
                )
                """
            )

    @staticmethod
    def _job_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result_json = row["result_json"]
        job = {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "query": row["query"],
            "status": row["status"],
            "created_at": row["created_at"],
        }
        if row["started_at"] is not None:
            job["started_at"] = row["started_at"]
        if row["completed_at"] is not None:
            job["completed_at"] = row["completed_at"]
        if result_json:
            job["result"] = json.loads(result_json)
        if row["error"]:
            job["error"] = row["error"]
        return job

    def create_job(self, job_id: str, kind: str, query: str) -> dict[str, Any]:
        created_at = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (job_id, kind, query, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, kind, query, "queued", created_at),
            )
        return {
            "job_id": job_id,
            "kind": kind,
            "query": query,
            "status": "queued",
            "created_at": created_at,
        }

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "started_at",
            "completed_at",
            "result",
            "error",
        }
        assignments = []
        values = []
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"Unsupported job field: {key}")
            column = "result_json" if key == "result" else key
            assignments.append(f"{column} = ?")
            if key == "result":
                value = json.dumps(value, ensure_ascii=False)
            values.append(value)

        if not assignments:
            return

        values.append(job_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ?",
                values,
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._job_from_row(row)

    def prune_jobs(self, max_jobs: int, ttl_seconds: int, now: float | None = None) -> None:
        now = now or time.time()
        cutoff = now - ttl_seconds
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                DELETE FROM jobs
                WHERE status IN ('completed', 'failed')
                  AND COALESCE(completed_at, created_at) < ?
                """,
                (cutoff,),
            )
            overflow = connection.execute(
                "SELECT COUNT(*) - ? FROM jobs",
                (max_jobs,),
            ).fetchone()[0]
            if overflow <= 0:
                return
            old_rows = connection.execute(
                """
                SELECT job_id FROM jobs
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (overflow,),
            ).fetchall()
            connection.executemany(
                "DELETE FROM jobs WHERE job_id = ?",
                [(row["job_id"],) for row in old_rows],
            )

    def record_metric(
        self,
        *,
        latency_seconds: float,
        decision: str,
        risk_score: float | None,
        success: bool,
        request_id: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO metrics (
                    created_at,
                    request_id,
                    decision,
                    latency_seconds,
                    risk_score,
                    success
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    request_id,
                    decision,
                    latency_seconds,
                    risk_score,
                    int(success),
                ),
            )

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "mean": round(sum(values) / len(values), 6),
        }

    def metrics_summary(self, sample_limit: int = 100) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT decision, latency_seconds, risk_score, success
                FROM metrics
                ORDER BY id DESC
                LIMIT ?
                """,
                (sample_limit,),
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]

        decision_counts: dict[str, int] = {}
        latencies = []
        risk_scores = []
        success_count = 0
        for row in rows:
            decision = str(row["decision"])
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            latencies.append(float(row["latency_seconds"]))
            if row["risk_score"] is not None:
                risk_scores.append(float(row["risk_score"]))
            success_count += int(row["success"])

        accepted = decision_counts.get("ACCEPT", 0) + decision_counts.get("DIRECT", 0)
        rejected = decision_counts.get("REJECT", 0)
        sample_count = len(rows)
        return {
            "request_count": int(total),
            "sample_size": sample_count,
            "decision_counts": decision_counts,
            "average_latency_seconds": (
                round(sum(latencies) / len(latencies), 6) if latencies else None
            ),
            "accept_rate": round(accepted / sample_count, 6) if sample_count else None,
            "reject_rate": round(rejected / sample_count, 6) if sample_count else None,
            "success_rate": round(success_count / sample_count, 6) if sample_count else None,
            "risk_score_stats": self._stats(risk_scores),
            "latency_sample_size": sample_count,
            "state_db_path": str(self.db_path),
        }
