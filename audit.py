"""
audit.py — structured audit log + content store, backed by SQLite (stdlib only).

Two tables, deliberately separated:

  * audit_log : APPEND-ONLY. One immutable row per event (a classification or an
                appeal). This is the canonical trail graders read via GET /log.
  * content   : MUTABLE current state of each submission. Its `status` flips from
                "classified" to "under_review" when an appeal is filed.

Keeping the immutable trail apart from the mutable status record means an appeal
never overwrites the original decision — it is logged *beside* it.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

# DB lives in the repo root by default; overridable for tests via PROVENANCE_DB.
DB_PATH = os.environ.get(
    "PROVENANCE_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "provenance.db"),
)


def _now() -> str:
    """ISO-8601 UTC timestamp with millisecond precision, e.g. 2026-06-29T14:32:10.123Z."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS content (
                content_id          TEXT PRIMARY KEY,
                creator_id          TEXT,
                text                TEXT,
                created_at          TEXT,
                attribution         TEXT,
                confidence          REAL,
                ai_likelihood       REAL,
                llm_score           REAL,
                stylometric_score   REAL,
                stylometric_detail  TEXT,
                repetition_score    REAL,
                content_type        TEXT,
                label_variant       TEXT,
                status              TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id          TEXT,
                event_type          TEXT,   -- 'classified' | 'appeal'
                timestamp           TEXT,
                creator_id          TEXT,
                attribution         TEXT,
                confidence          REAL,
                ai_likelihood       REAL,
                llm_score           REAL,
                stylometric_score   REAL,
                repetition_score    REAL,
                content_type        TEXT,
                status              TEXT,
                appeal_reasoning    TEXT,
                details             TEXT
            );

            -- stretch S2: earned "verified human" credentials
            CREATE TABLE IF NOT EXISTS credentials (
                creator_id    TEXT PRIMARY KEY,
                credential_id TEXT,
                earned_at     TEXT,
                method        TEXT
            );

            -- stretch S2: one-time verification challenges
            CREATE TABLE IF NOT EXISTS verification_challenges (
                challenge_id TEXT PRIMARY KEY,
                creator_id   TEXT,
                prompt       TEXT,
                token        TEXT,
                issued_at    TEXT,
                status       TEXT   -- 'issued' | 'used'
            );
            """
        )
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema, for pre-existing databases."""
    for table, col, coltype in (
        ("content", "repetition_score", "REAL"),
        ("content", "content_type", "TEXT"),
        ("audit_log", "repetition_score", "REAL"),
        ("audit_log", "content_type", "TEXT"),
    ):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # column already exists


def record_submission(record: dict) -> str:
    """
    Persist a classification: write the mutable content row AND an immutable
    audit_log row (event_type='classified'). Returns the ISO timestamp used.

    `record` keys: content_id, creator_id, text, attribution, confidence,
    ai_likelihood, llm_score, stylometric_score, stylometric_detail (dict|None),
    label_variant.
    """
    ts = _now()
    detail_json = json.dumps(record.get("stylometric_detail")) if record.get(
        "stylometric_detail"
    ) is not None else None

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO content (
                content_id, creator_id, text, created_at, attribution, confidence,
                ai_likelihood, llm_score, stylometric_score, stylometric_detail,
                repetition_score, content_type, label_variant, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'classified')
            """,
            (
                record["content_id"],
                record.get("creator_id"),
                record.get("text"),
                ts,
                record.get("attribution"),
                record.get("confidence"),
                record.get("ai_likelihood"),
                record.get("llm_score"),
                record.get("stylometric_score"),
                detail_json,
                record.get("repetition_score"),
                record.get("content_type"),
                record.get("label_variant"),
            ),
        )
        conn.execute(
            """
            INSERT INTO audit_log (
                content_id, event_type, timestamp, creator_id, attribution,
                confidence, ai_likelihood, llm_score, stylometric_score,
                repetition_score, content_type, status, appeal_reasoning, details
            ) VALUES (?, 'classified', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'classified', NULL, ?)
            """,
            (
                record["content_id"],
                ts,
                record.get("creator_id"),
                record.get("attribution"),
                record.get("confidence"),
                record.get("ai_likelihood"),
                record.get("llm_score"),
                record.get("stylometric_score"),
                record.get("repetition_score"),
                record.get("content_type"),
                detail_json,
            ),
        )
    return ts


def record_appeal(content_id: str, creator_reasoning: str) -> dict | None:
    """
    File an appeal: flip the content's status to 'under_review' and append an
    immutable audit_log row (event_type='appeal') carrying the ORIGINAL decision
    forward alongside the creator's reasoning.

    Returns {"appeal_id": int, "original_decision": {...}}, or None if content_id unknown.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM content WHERE content_id = ?", (content_id,)
        ).fetchone()
        if row is None:
            return None

        conn.execute(
            "UPDATE content SET status = 'under_review' WHERE content_id = ?",
            (content_id,),
        )
        cur = conn.execute(
            """
            INSERT INTO audit_log (
                content_id, event_type, timestamp, creator_id, attribution,
                confidence, ai_likelihood, llm_score, stylometric_score, status,
                appeal_reasoning, details
            ) VALUES (?, 'appeal', ?, ?, ?, ?, ?, ?, ?, 'under_review', ?, NULL)
            """,
            (
                content_id,
                _now(),
                row["creator_id"],
                row["attribution"],
                row["confidence"],
                row["ai_likelihood"],
                row["llm_score"],
                row["stylometric_score"],
                creator_reasoning,
            ),
        )
        appeal_id = cur.lastrowid

    return {
        "appeal_id": appeal_id,
        "original_decision": {
            "attribution": row["attribution"],
            "confidence": row["confidence"],
            "ai_likelihood": row["ai_likelihood"],
            "llm_score": row["llm_score"],
            "stylometric_score": row["stylometric_score"],
            "classified_at": row["created_at"],
        },
    }


def get_recent_entries(limit: int = 20) -> list[dict]:
    """Most recent audit_log rows, newest first. Powers GET /log."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_clean(dict(r)) for r in rows]


def get_content(content_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM content WHERE content_id = ?", (content_id,)
        ).fetchone()
    return _content_dict(row) if row else None


def get_review_queue() -> list[dict]:
    """Content awaiting human review (status = under_review). Powers GET /review-queue."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM content WHERE status = 'under_review' ORDER BY created_at DESC"
        ).fetchall()
    return [_content_dict(r) for r in rows]


def _content_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("stylometric_detail"):
        try:
            d["stylometric_detail"] = json.loads(d["stylometric_detail"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _clean(d: dict) -> dict:
    """Drop NULL columns and parse the JSON details blob for tidy /log output."""
    if d.get("details"):
        try:
            d["details"] = json.loads(d["details"])
        except (json.JSONDecodeError, TypeError):
            pass
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Verified-human credentials + challenges  (stretch S2)
# ---------------------------------------------------------------------------

def get_credential(creator_id: str) -> dict | None:
    """Return the creator's verified-human credential, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT credential_id, earned_at, method FROM credentials WHERE creator_id = ?",
            (creator_id,),
        ).fetchone()
    return dict(row) if row else None


def grant_credential(creator_id: str, credential_id: str, method: str) -> dict:
    """Grant (or refresh) a verified-human credential for a creator."""
    earned_at = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO credentials (creator_id, credential_id, earned_at, method)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(creator_id) DO UPDATE SET
                credential_id = excluded.credential_id,
                earned_at     = excluded.earned_at,
                method        = excluded.method
            """,
            (creator_id, credential_id, earned_at, method),
        )
    return {"credential_id": credential_id, "earned_at": earned_at, "method": method}


def create_challenge(challenge_id: str, creator_id: str, prompt: str, token: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO verification_challenges
                (challenge_id, creator_id, prompt, token, issued_at, status)
            VALUES (?, ?, ?, ?, ?, 'issued')
            """,
            (challenge_id, creator_id, prompt, token, _now()),
        )


def get_challenge(challenge_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM verification_challenges WHERE challenge_id = ?", (challenge_id,)
        ).fetchone()
    return dict(row) if row else None


def mark_challenge_used(challenge_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE verification_challenges SET status = 'used' WHERE challenge_id = ?",
            (challenge_id,),
        )


# ---------------------------------------------------------------------------
# Analytics  (stretch S3)
# ---------------------------------------------------------------------------

def get_analytics() -> dict:
    """Aggregate metrics over the content store + audit log for GET /analytics."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT attribution, confidence FROM content"
        ).fetchall()
        total = len(rows)

        by_attr = {"likely_ai": 0, "likely_human": 0, "uncertain": 0}
        conf_sum = 0.0
        for r in rows:
            by_attr[r["attribution"]] = by_attr.get(r["attribution"], 0) + 1
            if r["confidence"] is not None:
                conf_sum += r["confidence"]

        appeals = conn.execute(
            "SELECT attribution FROM audit_log WHERE event_type = 'appeal'"
        ).fetchall()
        n_appeals = len(appeals)
        appeals_by_attr: dict = {}
        for a in appeals:
            appeals_by_attr[a["attribution"]] = appeals_by_attr.get(a["attribution"], 0) + 1

        # additional metric: average signal disagreement (spread among available signals)
        sig_rows = conn.execute(
            "SELECT llm_score, stylometric_score, repetition_score FROM content"
        ).fetchall()
        spreads = []
        for s in sig_rows:
            vals = [v for v in (s["llm_score"], s["stylometric_score"], s["repetition_score"]) if v is not None]
            if len(vals) > 1:
                spreads.append(max(vals) - min(vals))
        avg_disagreement = round(sum(spreads) / len(spreads), 3) if spreads else 0.0

        per_day = conn.execute(
            "SELECT substr(created_at,1,10) AS day, COUNT(*) AS n "
            "FROM content GROUP BY day ORDER BY day"
        ).fetchall()

        verified = conn.execute("SELECT COUNT(*) FROM credentials").fetchone()[0]

    def pct(n: int) -> float:
        return round(100 * n / total, 1) if total else 0.0

    return {
        "total_submissions": total,
        "attribution_breakdown": {k: {"count": v, "percent": pct(v)} for k, v in by_attr.items()},
        "appeal_rate": round(n_appeals / total, 3) if total else 0.0,
        "appeals": {"total": n_appeals, "by_original_attribution": appeals_by_attr},
        "average_confidence": round(conf_sum / total, 3) if total else 0.0,
        "average_signal_disagreement": avg_disagreement,
        "verified_human_creators": verified,
        "submissions_per_day": {r["day"]: r["n"] for r in per_day},
    }
