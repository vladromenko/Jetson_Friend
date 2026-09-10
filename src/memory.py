import json
import os
import re
import sqlite3
import threading
import time


class Memory:
    def __init__(self):
        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        data_dir = os.path.join(root, "data")
        os.makedirs(data_dir, exist_ok=True)

        self.path = os.path.join(data_dir, "memory.db")
        self.lock = threading.Lock()

        self._create_database()

    def _connect(self):
        connection = sqlite3.connect(
            self.path,
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _create_database(self):
        with self.lock:
            with self._connect() as db:
                db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS profile (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created REAL NOT NULL,
                        updated REAL NOT NULL,
                        kind TEXT NOT NULL,
                        text TEXT NOT NULL,
                        emotion TEXT,
                        importance REAL NOT NULL DEFAULT 0.5,
                        source TEXT,
                        active INTEGER NOT NULL DEFAULT 1
                    );

                    CREATE TABLE IF NOT EXISTS objects (
                        name TEXT PRIMARY KEY,
                        first_seen REAL NOT NULL,
                        last_seen REAL NOT NULL,
                        seen_count INTEGER NOT NULL DEFAULT 1,
                        location TEXT,
                        confidence REAL
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created REAL NOT NULL,
                        kind TEXT NOT NULL,
                        text TEXT NOT NULL,
                        importance REAL NOT NULL DEFAULT 0.3
                    );

                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )

    def set_profile(self, key, value):
        now = time.time()

        with self.lock:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO profile(key, value, updated)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated = excluded.updated
                    """,
                    (key, str(value), now),
                )

    def get_profile(self, key, default=None):
        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    "SELECT value FROM profile WHERE key = ?",
                    (key,),
                ).fetchone()

        if row is None:
            return default

        return row["value"]

    def remember(
        self,
        text,
        kind="fact",
        emotion="neutral",
        importance=0.5,
        source="conversation",
    ):
        text = str(text).strip()

        if not text:
            return None

        importance = max(
            0.0,
            min(1.0, float(importance)),
        )

        now = time.time()

        with self.lock:
            with self._connect() as db:
                existing = db.execute(
                    """
                    SELECT id
                    FROM memories
                    WHERE active = 1
                      AND lower(trim(text)) = lower(trim(?))
                    ORDER BY updated DESC
                    LIMIT 1
                    """,
                    (text[:1000],),
                ).fetchone()

                if existing is not None:
                    db.execute(
                        """
                        UPDATE memories
                        SET updated = ?,
                            importance = MAX(importance, ?)
                        WHERE id = ?
                        """,
                        (
                            now,
                            importance,
                            existing["id"],
                        ),
                    )
                    return existing["id"]

                cursor = db.execute(
                    """
                    INSERT INTO memories(
                        created,
                        updated,
                        kind,
                        text,
                        emotion,
                        importance,
                        source,
                        active
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        now,
                        now,
                        kind,
                        text[:1000],
                        emotion,
                        importance,
                        source,
                    ),
                )

                return cursor.lastrowid

    def forget_last(self):
        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT id, text
                    FROM memories
                    WHERE active = 1
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()

                if row is None:
                    return None

                db.execute(
                    """
                    UPDATE memories
                    SET active = 0,
                        updated = ?
                    WHERE id = ?
                    """,
                    (time.time(), row["id"]),
                )

                return row["text"]

    def search(self, query, limit=6):
        words = {
            word
            for word in re.findall(
                r"[a-zA-Z0-9']+",
                query.lower(),
            )
            if len(word) >= 3
        }

        with self.lock:
            with self._connect() as db:
                rows = db.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE active = 1
                    ORDER BY importance DESC, updated DESC
                    LIMIT 150
                    """
                ).fetchall()

        scored = []

        for row in rows:
            memory_words = set(
                re.findall(
                    r"[a-zA-Z0-9']+",
                    row["text"].lower(),
                )
            )

            overlap = len(words & memory_words)

            age_days = max(
                0.0,
                (time.time() - row["updated"]) / 86400.0,
            )

            recency = 1.0 / (1.0 + age_days / 30.0)

            score = (
                overlap * 2.0
                + float(row["importance"]) * 2.0
                + recency
            )

            if overlap > 0:
                scored.append(
                    (
                        score,
                        dict(row),
                    )
                )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            item[1]
            for item in scored[:limit]
        ]

    def context(self, query, limit=6):
        memories = self.search(
            query,
            limit=limit,
        )

        if not memories:
            return "No relevant long-term memories."

        lines = []

        for item in memories:
            emotion = item.get("emotion") or "neutral"

            lines.append(
                "- "
                + item["text"]
                + f" [type={item['kind']}, emotion={emotion}]"
            )

        return "\n".join(lines)

    def describe_person(self, limit=20):
        with self.lock:
            with self._connect() as db:
                rows = db.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE active = 1
                    ORDER BY importance DESC, updated DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        if not rows:
            return "I do not have any long-term memories about you yet."

        return "\n".join(
            "- " + row["text"]
            for row in rows
        )

    def see_object(
        self,
        name,
        location="camera view",
        confidence=None,
    ):
        name = str(name).strip().lower()

        if not name:
            return

        now = time.time()

        with self.lock:
            with self._connect() as db:
                existing = db.execute(
                    """
                    SELECT name
                    FROM objects
                    WHERE name = ?
                    """,
                    (name,),
                ).fetchone()

                if existing is None:
                    db.execute(
                        """
                        INSERT INTO objects(
                            name,
                            first_seen,
                            last_seen,
                            seen_count,
                            location,
                            confidence
                        )
                        VALUES (?, ?, ?, 1, ?, ?)
                        """,
                        (
                            name,
                            now,
                            now,
                            location,
                            confidence,
                        ),
                    )
                else:
                    db.execute(
                        """
                        UPDATE objects
                        SET last_seen = ?,
                            seen_count = seen_count + 1,
                            location = ?,
                            confidence = ?
                        WHERE name = ?
                        """,
                        (
                            now,
                            location,
                            confidence,
                            name,
                        ),
                    )

    def last_seen(self, name):
        name = str(name).strip().lower()

        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT *
                    FROM objects
                    WHERE name LIKE ?
                    ORDER BY last_seen DESC
                    LIMIT 1
                    """,
                    ("%" + name + "%",),
                ).fetchone()

        if row is None:
            return None

        return dict(row)

    def event(
        self,
        kind,
        text,
        importance=0.3,
    ):
        with self.lock:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO events(
                        created,
                        kind,
                        text,
                        importance
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        time.time(),
                        kind,
                        text[:1000],
                        float(importance),
                    ),
                )

    def set_setting(self, key, value):
        encoded = json.dumps(value)

        with self.lock:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO settings(key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value
                    """,
                    (key, encoded),
                )

    def get_setting(self, key, default=None):
        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT value
                    FROM settings
                    WHERE key = ?
                    """,
                    (key,),
                ).fetchone()

        if row is None:
            return default

        try:
            return json.loads(row["value"])
        except Exception:
            return default
