import json
import os
import re
import sqlite3
import threading
import time
import uuid


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "ours", "say", "she",
    "should", "so", "tell", "than", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your", "yours",
}


LONG_TERM_KINDS = {
    "fact",
    "identity",
    "preference",
    "goal",
    "episode",
    "relationship",
    "visual",
    "object",
}


TEMPORARY_KINDS = {
    "emotion",
    "mood",
    "temporary",
    "state",
}


class Memory:
    """
    Persistent local memory for MILO.

    The old single-user API still works, but memory can now be isolated
    by person_id.

    Temporary emotional state is kept separately from long-term memory.
    """

    def __init__(self):
        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            ),
        )

        data_dir = os.path.join(
            root,
            "data",
        )

        os.makedirs(
            data_dir,
            exist_ok=True,
        )

        self.path = os.path.join(
            data_dir,
            "memory.db",
        )

        self.lock = threading.RLock()

        self._create_database()
        self._migrate_database()
        self._migrate_legacy_person()

    def _connect(self):
        db = sqlite3.connect(
            self.path,
            timeout=10,
        )

        db.row_factory = sqlite3.Row

        db.execute(
            "PRAGMA foreign_keys = ON"
        )

        db.execute(
            "PRAGMA journal_mode = WAL"
        )

        db.execute(
            "PRAGMA synchronous = NORMAL"
        )

        return db

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

                    CREATE TABLE IF NOT EXISTS persons (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        created REAL NOT NULL,
                        updated REAL NOT NULL,
                        last_seen REAL,
                        active INTEGER NOT NULL DEFAULT 1
                    );

                    CREATE TABLE IF NOT EXISTS person_profiles (
                        person_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated REAL NOT NULL,
                        PRIMARY KEY(person_id, key)
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
                        active INTEGER NOT NULL DEFAULT 1,
                        person_id TEXT,
                        normalized_text TEXT,
                        expires_at REAL,
                        confidence REAL NOT NULL DEFAULT 1.0
                    );

                    CREATE TABLE IF NOT EXISTS temporary_states (
                        person_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 1.0,
                        source TEXT,
                        created REAL NOT NULL,
                        updated REAL NOT NULL,
                        expires_at REAL,
                        PRIMARY KEY(person_id, key)
                    );

                    CREATE TABLE IF NOT EXISTS objects (
                        name TEXT PRIMARY KEY,
                        first_seen REAL NOT NULL,
                        last_seen REAL NOT NULL,
                        seen_count INTEGER NOT NULL DEFAULT 1,
                        location TEXT,
                        confidence REAL,
                        owner_person_id TEXT
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created REAL NOT NULL,
                        kind TEXT NOT NULL,
                        text TEXT NOT NULL,
                        importance REAL NOT NULL DEFAULT 0.3,
                        person_id TEXT
                    );

                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )

    def _columns(
        self,
        db,
        table,
    ):
        rows = db.execute(
            f"PRAGMA table_info({table})"
        )

        return {
            row["name"]
            for row in rows
        }

    def _migrate_database(self):
        """
        Upgrade databases produced by the previous Jetson_Friend version.
        Existing memories are preserved.
        """

        with self.lock:
            with self._connect() as db:
                memory_columns = self._columns(
                    db,
                    "memories",
                )

                additions = {
                    "person_id": "TEXT",
                    "normalized_text": "TEXT",
                    "expires_at": "REAL",
                    "confidence": "REAL NOT NULL DEFAULT 1.0",
                }

                for column, definition in additions.items():
                    if column not in memory_columns:
                        db.execute(
                            f"ALTER TABLE memories "
                            f"ADD COLUMN {column} {definition}"
                        )

                object_columns = self._columns(
                    db,
                    "objects",
                )

                if "owner_person_id" not in object_columns:
                    db.execute(
                        "ALTER TABLE objects "
                        "ADD COLUMN owner_person_id TEXT"
                    )

                event_columns = self._columns(
                    db,
                    "events",
                )

                if "person_id" not in event_columns:
                    db.execute(
                        "ALTER TABLE events "
                        "ADD COLUMN person_id TEXT"
                    )

                rows = db.execute(
                    """
                    SELECT id, text
                    FROM memories
                    WHERE normalized_text IS NULL
                       OR normalized_text = ''
                    """
                ).fetchall()

                for row in rows:
                    normalized = self._normalize_text(
                        row["text"]
                    )

                    db.execute(
                        """
                        UPDATE memories
                        SET normalized_text = ?
                        WHERE id = ?
                        """,
                        (
                            normalized,
                            row["id"],
                        ),
                    )

                db.executescript(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_person_active
                    ON memories(
                        person_id,
                        active,
                        updated DESC
                    );

                    CREATE INDEX IF NOT EXISTS idx_memories_normalized
                    ON memories(
                        person_id,
                        normalized_text
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_person_created
                    ON events(
                        person_id,
                        created DESC
                    );

                    CREATE INDEX IF NOT EXISTS idx_temp_person
                    ON temporary_states(
                        person_id,
                        updated DESC
                    );
                    """
                )

    def _migrate_legacy_person(self):
        """
        Convert old profile.name into the first real person.

        Existing memories and events are assigned to this person.
        """

        with self.lock:
            with self._connect() as db:
                current_id = self._get_setting_db(
                    db,
                    "current_person_id",
                )

                if current_id:
                    row = db.execute(
                        """
                        SELECT id
                        FROM persons
                        WHERE id = ?
                          AND active = 1
                        """,
                        (
                            current_id,
                        ),
                    ).fetchone()

                    if row is not None:
                        return

                row = db.execute(
                    """
                    SELECT value
                    FROM profile
                    WHERE key = 'name'
                    """
                ).fetchone()

                if row is None:
                    return

                name = self._normalize_name(
                    row["value"]
                )

                if not name:
                    return

                person = db.execute(
                    """
                    SELECT id
                    FROM persons
                    WHERE lower(name) = lower(?)
                      AND active = 1
                    LIMIT 1
                    """,
                    (
                        name,
                    ),
                ).fetchone()

                if person is None:
                    person_id = str(
                        uuid.uuid4()
                    )

                    now = time.time()

                    db.execute(
                        """
                        INSERT INTO persons(
                            id,
                            name,
                            created,
                            updated,
                            last_seen,
                            active
                        )
                        VALUES (?, ?, ?, ?, ?, 1)
                        """,
                        (
                            person_id,
                            name,
                            now,
                            now,
                            now,
                        ),
                    )

                else:
                    person_id = person["id"]

                db.execute(
                    """
                    UPDATE memories
                    SET person_id = ?
                    WHERE person_id IS NULL
                    """,
                    (
                        person_id,
                    ),
                )

                db.execute(
                    """
                    UPDATE events
                    SET person_id = ?
                    WHERE person_id IS NULL
                    """,
                    (
                        person_id,
                    ),
                )

                self._set_setting_db(
                    db,
                    "current_person_id",
                    person_id,
                )

    @staticmethod
    def _normalize_name(
        name,
    ):
        return " ".join(
            str(name)
            .strip()
            .split()
        )

    @staticmethod
    def _normalize_text(
        text,
    ):
        text = str(
            text
        ).strip().lower()

        text = re.sub(
            r"[^\w\s']",
            " ",
            text,
            flags=re.UNICODE,
        )

        return " ".join(
            text.split()
        )

    @staticmethod
    def _keywords(
        text,
    ):
        words = re.findall(
            r"[a-zA-Z0-9']+",
            str(text).lower(),
        )

        return {
            word
            for word in words
            if (
                len(word) >= 3
                and word not in STOPWORDS
            )
        }

    @staticmethod
    def _clamp(
        value,
        default=0.5,
    ):
        try:
            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            number = default

        return max(
            0.0,
            min(
                1.0,
                number,
            ),
        )

    def _set_setting_db(
        self,
        db,
        key,
        value,
    ):
        db.execute(
            """
            INSERT INTO settings(
                key,
                value
            )
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value
            """,
            (
                str(key),
                json.dumps(
                    value
                ),
            ),
        )

    def _get_setting_db(
        self,
        db,
        key,
        default=None,
    ):
        row = db.execute(
            """
            SELECT value
            FROM settings
            WHERE key = ?
            """,
            (
                str(key),
            ),
        ).fetchone()

        if row is None:
            return default

        try:
            return json.loads(
                row["value"]
            )

        except Exception:
            return default

    # ============================================================
    # PEOPLE
    # ============================================================

    def create_person(
        self,
        name,
        make_current=True,
    ):
        name = self._normalize_name(
            name
        )

        if not name:
            return None

        now = time.time()

        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT id
                    FROM persons
                    WHERE lower(name) = lower(?)
                      AND active = 1
                    LIMIT 1
                    """,
                    (
                        name,
                    ),
                ).fetchone()

                if row is None:
                    person_id = str(
                        uuid.uuid4()
                    )

                    db.execute(
                        """
                        INSERT INTO persons(
                            id,
                            name,
                            created,
                            updated,
                            last_seen,
                            active
                        )
                        VALUES (?, ?, ?, ?, ?, 1)
                        """,
                        (
                            person_id,
                            name,
                            now,
                            now,
                            now,
                        ),
                    )

                else:
                    person_id = row["id"]

                    db.execute(
                        """
                        UPDATE persons
                        SET name = ?,
                            updated = ?,
                            last_seen = ?
                        WHERE id = ?
                        """,
                        (
                            name,
                            now,
                            now,
                            person_id,
                        ),
                    )

                if make_current:
                    self._set_setting_db(
                        db,
                        "current_person_id",
                        person_id,
                    )

                    db.execute(
                        """
                        INSERT INTO profile(
                            key,
                            value,
                            updated
                        )
                        VALUES ('name', ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            value = excluded.value,
                            updated = excluded.updated
                        """,
                        (
                            name,
                            now,
                        ),
                    )

                return person_id

    def get_person(
        self,
        person_id,
    ):
        if not person_id:
            return None

        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT *
                    FROM persons
                    WHERE id = ?
                      AND active = 1
                    """,
                    (
                        person_id,
                    ),
                ).fetchone()

        if row is None:
            return None

        return dict(
            row
        )

    def find_person_by_name(
        self,
        name,
    ):
        name = self._normalize_name(
            name
        )

        if not name:
            return None

        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT *
                    FROM persons
                    WHERE lower(name) = lower(?)
                      AND active = 1
                    LIMIT 1
                    """,
                    (
                        name,
                    ),
                ).fetchone()

        if row is None:
            return None

        return dict(
            row
        )

    def list_persons(
        self,
    ):
        with self.lock:
            with self._connect() as db:
                rows = db.execute(
                    """
                    SELECT *
                    FROM persons
                    WHERE active = 1
                    ORDER BY
                        last_seen DESC,
                        name ASC
                    """
                ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def set_current_person(
        self,
        person_id,
    ):
        person = self.get_person(
            person_id
        )

        if person is None:
            return False

        now = time.time()

        with self.lock:
            with self._connect() as db:
                self._set_setting_db(
                    db,
                    "current_person_id",
                    person_id,
                )

                db.execute(
                    """
                    UPDATE persons
                    SET updated = ?,
                        last_seen = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        now,
                        person_id,
                    ),
                )

                db.execute(
                    """
                    INSERT INTO profile(
                        key,
                        value,
                        updated
                    )
                    VALUES ('name', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated = excluded.updated
                    """,
                    (
                        person["name"],
                        now,
                    ),
                )

        return True

    def get_current_person_id(
        self,
    ):
        with self.lock:
            with self._connect() as db:
                person_id = self._get_setting_db(
                    db,
                    "current_person_id",
                )

                if not person_id:
                    return None

                row = db.execute(
                    """
                    SELECT id
                    FROM persons
                    WHERE id = ?
                      AND active = 1
                    """,
                    (
                        person_id,
                    ),
                ).fetchone()

        if row is None:
            return None

        return person_id

    def get_current_person(
        self,
    ):
        return self.get_person(
            self.get_current_person_id()
        )

    def touch_person(
        self,
        person_id,
    ):
        if not person_id:
            return False

        now = time.time()

        with self.lock:
            with self._connect() as db:
                result = db.execute(
                    """
                    UPDATE persons
                    SET last_seen = ?,
                        updated = ?
                    WHERE id = ?
                      AND active = 1
                    """,
                    (
                        now,
                        now,
                        person_id,
                    ),
                )

                return (
                    result.rowcount > 0
                )

    def forget_person(
        self,
        person_id,
    ):
        if not person_id:
            return False

        now = time.time()

        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT id
                    FROM persons
                    WHERE id = ?
                      AND active = 1
                    """,
                    (
                        person_id,
                    ),
                ).fetchone()

                if row is None:
                    return False

                db.execute(
                    """
                    UPDATE memories
                    SET active = 0,
                        updated = ?
                    WHERE person_id = ?
                    """,
                    (
                        now,
                        person_id,
                    ),
                )

                db.execute(
                    """
                    DELETE FROM temporary_states
                    WHERE person_id = ?
                    """,
                    (
                        person_id,
                    ),
                )

                db.execute(
                    """
                    DELETE FROM person_profiles
                    WHERE person_id = ?
                    """,
                    (
                        person_id,
                    ),
                )

                db.execute(
                    """
                    UPDATE persons
                    SET active = 0,
                        updated = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        person_id,
                    ),
                )

                current_id = self._get_setting_db(
                    db,
                    "current_person_id",
                )

                if current_id == person_id:
                    self._set_setting_db(
                        db,
                        "current_person_id",
                        None,
                    )

                    db.execute(
                        """
                        DELETE FROM profile
                        WHERE key = 'name'
                        """
                    )

        return True

    def _resolve_person_id(
        self,
        person_id=None,
    ):
        if person_id:
            return person_id

        return self.get_current_person_id()

    # ============================================================
    # PROFILE
    # ============================================================

    def set_profile(
        self,
        key,
        value,
        person_id=None,
    ):
        key = str(
            key
        ).strip()

        if not key:
            return

        resolved = self._resolve_person_id(
            person_id
        )

        if (
            key == "name"
            and not resolved
        ):
            resolved = self.create_person(
                value,
                make_current=True,
            )

        now = time.time()

        with self.lock:
            with self._connect() as db:
                if resolved:
                    db.execute(
                        """
                        INSERT INTO person_profiles(
                            person_id,
                            key,
                            value,
                            updated
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(person_id, key) DO UPDATE SET
                            value = excluded.value,
                            updated = excluded.updated
                        """,
                        (
                            resolved,
                            key,
                            str(value),
                            now,
                        ),
                    )

                db.execute(
                    """
                    INSERT INTO profile(
                        key,
                        value,
                        updated
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated = excluded.updated
                    """,
                    (
                        key,
                        str(value),
                        now,
                    ),
                )

                if (
                    key == "name"
                    and resolved
                ):
                    db.execute(
                        """
                        UPDATE persons
                        SET name = ?,
                            updated = ?
                        WHERE id = ?
                        """,
                        (
                            self._normalize_name(
                                value
                            ),
                            now,
                            resolved,
                        ),
                    )

    def get_profile(
        self,
        key,
        default=None,
        person_id=None,
    ):
        key = str(
            key
        ).strip()

        resolved = self._resolve_person_id(
            person_id
        )

        with self.lock:
            with self._connect() as db:
                if resolved:
                    row = db.execute(
                        """
                        SELECT value
                        FROM person_profiles
                        WHERE person_id = ?
                          AND key = ?
                        """,
                        (
                            resolved,
                            key,
                        ),
                    ).fetchone()

                    if row is not None:
                        return row["value"]

                    if key == "name":
                        row = db.execute(
                            """
                            SELECT name
                            FROM persons
                            WHERE id = ?
                              AND active = 1
                            """,
                            (
                                resolved,
                            ),
                        ).fetchone()

                        if row is not None:
                            return row["name"]

                row = db.execute(
                    """
                    SELECT value
                    FROM profile
                    WHERE key = ?
                    """,
                    (
                        key,
                    ),
                ).fetchone()

        if row is None:
            return default

        return row["value"]

    # ============================================================
    # LONG-TERM MEMORY
    # ============================================================

    def remember(
        self,
        text,
        kind="fact",
        emotion="neutral",
        importance=0.5,
        source="conversation",
        person_id=None,
        confidence=1.0,
        expires_at=None,
    ):
        text = str(
            text
        ).strip()

        if not text:
            return None

        kind = str(
            kind or "fact"
        ).strip().lower()[:30]

        emotion = str(
            emotion or "neutral"
        ).strip().lower()[:30]

        importance = self._clamp(
            importance,
            0.5,
        )

        confidence = self._clamp(
            confidence,
            1.0,
        )

        resolved = self._resolve_person_id(
            person_id
        )

        # Emotions and moods should not normally become permanent facts.
        if kind in TEMPORARY_KINDS:
            if resolved:
                self.set_temporary_state(
                    "emotion",
                    {
                        "text": text[:500],
                        "emotion": emotion,
                    },
                    person_id=resolved,
                    confidence=confidence,
                    source=source,
                    ttl_seconds=3 * 3600,
                )

            return None

        if kind not in LONG_TERM_KINDS:
            kind = "fact"

        normalized = self._normalize_text(
            text
        )

        if not normalized:
            return None

        now = time.time()

        with self.lock:
            with self._connect() as db:
                if resolved:
                    existing = db.execute(
                        """
                        SELECT
                            id,
                            importance,
                            confidence
                        FROM memories
                        WHERE active = 1
                          AND person_id = ?
                          AND normalized_text = ?
                        LIMIT 1
                        """,
                        (
                            resolved,
                            normalized,
                        ),
                    ).fetchone()

                else:
                    existing = db.execute(
                        """
                        SELECT
                            id,
                            importance,
                            confidence
                        FROM memories
                        WHERE active = 1
                          AND person_id IS NULL
                          AND normalized_text = ?
                        LIMIT 1
                        """,
                        (
                            normalized,
                        ),
                    ).fetchone()

                if existing is not None:
                    db.execute(
                        """
                        UPDATE memories
                        SET updated = ?,
                            kind = ?,
                            emotion = ?,
                            importance = ?,
                            source = ?,
                            confidence = ?,
                            expires_at = ?
                        WHERE id = ?
                        """,
                        (
                            now,
                            kind,
                            emotion,
                            max(
                                importance,
                                float(
                                    existing[
                                        "importance"
                                    ]
                                ),
                            ),
                            str(source)[:50],
                            max(
                                confidence,
                                float(
                                    existing[
                                        "confidence"
                                    ]
                                    or 0.0
                                ),
                            ),
                            expires_at,
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
                        active,
                        person_id,
                        normalized_text,
                        expires_at,
                        confidence
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        1, ?, ?, ?, ?
                    )
                    """,
                    (
                        now,
                        now,
                        kind,
                        text[:1000],
                        emotion,
                        importance,
                        str(source)[:50],
                        resolved,
                        normalized,
                        expires_at,
                        confidence,
                    ),
                )

                return cursor.lastrowid

    def forget_last(
        self,
        person_id=None,
    ):
        resolved = self._resolve_person_id(
            person_id
        )

        with self.lock:
            with self._connect() as db:
                if resolved:
                    row = db.execute(
                        """
                        SELECT id, text
                        FROM memories
                        WHERE active = 1
                          AND person_id = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (
                            resolved,
                        ),
                    ).fetchone()

                else:
                    row = db.execute(
                        """
                        SELECT id, text
                        FROM memories
                        WHERE active = 1
                          AND person_id IS NULL
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
                    (
                        time.time(),
                        row["id"],
                    ),
                )

                return row["text"]

    def search(
        self,
        query,
        limit=4,
        person_id=None,
    ):
        """
        Fast lexical retrieval for edge hardware.

        If a known person is active:
        - that person's memories may be returned;
        - global memories may be returned;
        - other people's memories cannot be returned.

        Importance only ranks already relevant memories.
        It cannot force an unrelated memory into the prompt.
        """

        resolved = self._resolve_person_id(
            person_id
        )

        query_words = self._keywords(
            query
        )

        if not query_words:
            return []

        now = time.time()

        with self.lock:
            with self._connect() as db:
                if resolved:
                    rows = db.execute(
                        """
                        SELECT *
                        FROM memories
                        WHERE active = 1
                          AND (
                              person_id = ?
                              OR person_id IS NULL
                          )
                          AND (
                              expires_at IS NULL
                              OR expires_at > ?
                          )
                        ORDER BY updated DESC
                        LIMIT 120
                        """,
                        (
                            resolved,
                            now,
                        ),
                    ).fetchall()

                else:
                    rows = db.execute(
                        """
                        SELECT *
                        FROM memories
                        WHERE active = 1
                          AND person_id IS NULL
                          AND (
                              expires_at IS NULL
                              OR expires_at > ?
                          )
                        ORDER BY updated DESC
                        LIMIT 120
                        """,
                        (
                            now,
                        ),
                    ).fetchall()

        scored = []

        for row in rows:
            memory_words = self._keywords(
                row["text"]
            )

            overlap = (
                query_words
                & memory_words
            )

            if overlap:
                union = (
                    query_words
                    | memory_words
                )

                lexical = (
                    len(overlap)
                    / max(
                        1,
                        len(union),
                    )
                )

                age_days = max(
                    0.0,
                    (
                        now
                        - float(
                            row["updated"]
                        )
                    )
                    / 86400.0,
                )

                recency = (
                    1.0
                    / (
                        1.0
                        + age_days / 30.0
                    )
                )

                importance = self._clamp(
                    row["importance"],
                    0.5,
                )

                confidence = self._clamp(
                    row["confidence"],
                    1.0,
                )

                if row["kind"] in {
                    "identity",
                    "preference",
                    "goal",
                }:
                    type_bonus = 0.15

                else:
                    type_bonus = 0.0

                score = (
                    len(overlap) * 3.0
                    + lexical * 2.0
                    + importance * 0.8
                    + recency * 0.4
                    + confidence * 0.3
                    + type_bonus
                )

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
            for item in scored[
                :max(
                    1,
                    int(limit),
                )
            ]
        ]

    def context(
        self,
        query,
        limit=4,
        person_id=None,
    ):
        memories = self.search(
            query,
            limit=limit,
            person_id=person_id,
        )

        if not memories:
            return (
                "No relevant long-term memories."
            )

        return "\n".join(
            (
                f"- {item['text']} "
                f"[type={item['kind']}]"
            )
            for item in memories
        )

    def describe_person(
        self,
        limit=20,
        person_id=None,
    ):
        resolved = self._resolve_person_id(
            person_id
        )

        if not resolved:
            return (
                "I do not know which person "
                "you mean yet."
            )

        now = time.time()

        with self.lock:
            with self._connect() as db:
                rows = db.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE active = 1
                      AND person_id = ?
                      AND (
                          expires_at IS NULL
                          OR expires_at > ?
                      )
                    ORDER BY
                        importance DESC,
                        updated DESC
                    LIMIT ?
                    """,
                    (
                        resolved,
                        now,
                        max(
                            1,
                            int(limit),
                        ),
                    ),
                ).fetchall()

        if not rows:
            return (
                "I do not have any long-term "
                "memories about you yet."
            )

        return "\n".join(
            "- " + row["text"]
            for row in rows
        )

    # ============================================================
    # TEMPORARY STATE / EMOTION
    # ============================================================

    def set_temporary_state(
        self,
        key,
        value,
        person_id=None,
        confidence=1.0,
        source="conversation",
        ttl_seconds=3 * 3600,
    ):
        resolved = self._resolve_person_id(
            person_id
        )

        key = str(
            key
        ).strip()

        if (
            not resolved
            or not key
        ):
            return False

        now = time.time()

        if ttl_seconds is None:
            expires_at = None

        else:
            expires_at = (
                now
                + max(
                    1.0,
                    float(
                        ttl_seconds
                    ),
                )
            )

        encoded = json.dumps(
            value,
            ensure_ascii=False,
        )

        with self.lock:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO temporary_states(
                        person_id,
                        key,
                        value,
                        confidence,
                        source,
                        created,
                        updated,
                        expires_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(person_id, key) DO UPDATE SET
                        value = excluded.value,
                        confidence = excluded.confidence,
                        source = excluded.source,
                        updated = excluded.updated,
                        expires_at = excluded.expires_at
                    """,
                    (
                        resolved,
                        key,
                        encoded,
                        self._clamp(
                            confidence,
                            1.0,
                        ),
                        str(source)[:50],
                        now,
                        now,
                        expires_at,
                    ),
                )

        return True

    def get_temporary_state(
        self,
        key,
        default=None,
        person_id=None,
        min_confidence=0.0,
    ):
        resolved = self._resolve_person_id(
            person_id
        )

        if not resolved:
            return default

        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT *
                    FROM temporary_states
                    WHERE person_id = ?
                      AND key = ?
                    """,
                    (
                        resolved,
                        str(key),
                    ),
                ).fetchone()

                if row is None:
                    return default

                expired = (
                    row["expires_at"]
                    is not None
                    and row["expires_at"]
                    <= time.time()
                )

                if expired:
                    db.execute(
                        """
                        DELETE FROM temporary_states
                        WHERE person_id = ?
                          AND key = ?
                        """,
                        (
                            resolved,
                            str(key),
                        ),
                    )

                    return default

                if (
                    float(
                        row["confidence"]
                    )
                    < float(
                        min_confidence
                    )
                ):
                    return default

        try:
            return json.loads(
                row["value"]
            )

        except Exception:
            return row["value"]

    def clear_temporary_state(
        self,
        key=None,
        person_id=None,
    ):
        resolved = self._resolve_person_id(
            person_id
        )

        if not resolved:
            return

        with self.lock:
            with self._connect() as db:
                if key is None:
                    db.execute(
                        """
                        DELETE FROM temporary_states
                        WHERE person_id = ?
                        """,
                        (
                            resolved,
                        ),
                    )

                else:
                    db.execute(
                        """
                        DELETE FROM temporary_states
                        WHERE person_id = ?
                          AND key = ?
                        """,
                        (
                            resolved,
                            str(key),
                        ),
                    )

    def cleanup_expired(
        self,
    ):
        now = time.time()

        with self.lock:
            with self._connect() as db:
                db.execute(
                    """
                    DELETE FROM temporary_states
                    WHERE expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (
                        now,
                    ),
                )

                db.execute(
                    """
                    UPDATE memories
                    SET active = 0,
                        updated = ?
                    WHERE active = 1
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (
                        now,
                        now,
                    ),
                )

    # ============================================================
    # OBJECT / ENVIRONMENT MEMORY
    # ============================================================

    def see_object(
        self,
        name,
        location="camera view",
        confidence=None,
        owner_person_id=None,
    ):
        name = str(
            name
        ).strip().lower()

        if not name:
            return

        now = time.time()

        if confidence is None:
            confidence_value = None

        else:
            confidence_value = self._clamp(
                confidence,
                0.5,
            )

        with self.lock:
            with self._connect() as db:
                existing = db.execute(
                    """
                    SELECT name
                    FROM objects
                    WHERE name = ?
                    """,
                    (
                        name,
                    ),
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
                            confidence,
                            owner_person_id
                        )
                        VALUES (?, ?, ?, 1, ?, ?, ?)
                        """,
                        (
                            name,
                            now,
                            now,
                            str(location)[:200],
                            confidence_value,
                            owner_person_id,
                        ),
                    )

                else:
                    db.execute(
                        """
                        UPDATE objects
                        SET last_seen = ?,
                            seen_count = seen_count + 1,
                            location = ?,
                            confidence = ?,
                            owner_person_id = COALESCE(
                                ?,
                                owner_person_id
                            )
                        WHERE name = ?
                        """,
                        (
                            now,
                            str(location)[:200],
                            confidence_value,
                            owner_person_id,
                            name,
                        ),
                    )

    def last_seen(
        self,
        name,
        owner_person_id=None,
    ):
        name = str(
            name
        ).strip().lower()

        if not name:
            return None

        with self.lock:
            with self._connect() as db:
                if owner_person_id:
                    row = db.execute(
                        """
                        SELECT *
                        FROM objects
                        WHERE name LIKE ?
                          AND (
                              owner_person_id = ?
                              OR owner_person_id IS NULL
                          )
                        ORDER BY
                            CASE
                                WHEN owner_person_id = ?
                                THEN 0
                                ELSE 1
                            END,
                            last_seen DESC
                        LIMIT 1
                        """,
                        (
                            "%" + name + "%",
                            owner_person_id,
                            owner_person_id,
                        ),
                    ).fetchone()

                else:
                    row = db.execute(
                        """
                        SELECT *
                        FROM objects
                        WHERE name LIKE ?
                        ORDER BY last_seen DESC
                        LIMIT 1
                        """,
                        (
                            "%" + name + "%",
                        ),
                    ).fetchone()

        if row is None:
            return None

        return dict(
            row
        )

    # ============================================================
    # EVENTS
    # ============================================================

    def event(
        self,
        kind,
        text,
        importance=0.3,
        person_id=None,
    ):
        resolved = self._resolve_person_id(
            person_id
        )

        with self.lock:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO events(
                        created,
                        kind,
                        text,
                        importance,
                        person_id
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(),
                        str(kind)[:50],
                        str(text)[:1000],
                        self._clamp(
                            importance,
                            0.3,
                        ),
                        resolved,
                    ),
                )

    # ============================================================
    # SETTINGS
    # ============================================================

    def set_setting(
        self,
        key,
        value,
    ):
        with self.lock:
            with self._connect() as db:
                self._set_setting_db(
                    db,
                    key,
                    value,
                )

    def get_setting(
        self,
        key,
        default=None,
    ):
        with self.lock:
            with self._connect() as db:
                return self._get_setting_db(
                    db,
                    key,
                    default,
                )