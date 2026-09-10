#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$HOME/Jetson_Friend"
SRC="$ROOT/src"
DATA="$ROOT/data"

mkdir -p "$SRC" "$DATA" "$DATA/people" "$DATA/memory"

echo "Creating Jetson Friend personal AI core..."

cat > "$SRC/memory.py" <<'PY'
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

            if overlap > 0 or float(row["importance"]) >= 0.75:
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
PY


cat > "$SRC/behavior.py" <<'PY'
import os
import time


class Behavior:
    def __init__(self, memory):
        self.memory = memory

        self.last_person_seen = 0.0
        self.absence_started = time.time()
        self.person_present = False

        self.activity = "none"
        self.activity_started = time.time()

        self.last_return_message = 0.0
        self.last_phone_message = 0.0
        self.last_computer_message = 0.0

        self.phone_limit = float(
            os.getenv(
                "PHONE_REMINDER_MINUTES",
                "20",
            )
        ) * 60.0

        self.computer_limit = float(
            os.getenv(
                "COMPUTER_BREAK_MINUTES",
                "60",
            )
        ) * 60.0

        self.long_absence = float(
            os.getenv(
                "LONG_ABSENCE_HOURS",
                "6",
            )
        ) * 3600.0

        self.reminder_cooldown = 45.0 * 60.0
        self.return_cooldown = 4.0 * 3600.0

    def update(self, scene):
        now = time.time()

        objects = {
            str(item).lower()
            for item in scene.get(
                "objects",
                [],
            )
        }

        faces = int(
            scene.get(
                "faces",
                0,
            )
        )

        visible = (
            faces > 0
            or "person" in objects
        )

        event = None

        if visible and not self.person_present:
            absence = now - self.absence_started

            self.person_present = True
            self.last_person_seen = now

            if (
                absence >= self.long_absence
                and now - self.last_return_message >= self.return_cooldown
            ):
                hours = absence / 3600.0

                event = {
                    "kind": "return",
                    "text": (
                        "Hey, I haven't seen you for quite a while. "
                        "You alive? Everything okay?"
                    ),
                    "emotion": "happy",
                }

                self.last_return_message = now

                self.memory.event(
                    "return",
                    f"Person returned after about {hours:.1f} hours away.",
                    0.5,
                )

        if visible:
            self.last_person_seen = now

        if not visible and self.person_present:
            self.person_present = False
            self.absence_started = now
            self._set_activity("away", now)

        new_activity = self._classify(
            visible,
            objects,
        )

        if new_activity != self.activity:
            self._set_activity(
                new_activity,
                now,
            )

        duration = now - self.activity_started

        if (
            event is None
            and self.activity == "phone"
            and duration >= self.phone_limit
            and now - self.last_phone_message >= self.reminder_cooldown
        ):
            minutes = int(duration / 60)

            event = {
                "kind": "phone",
                "text": (
                    f"You've been on your phone for about {minutes} minutes. "
                    "If we're supposed to be working, put it down and get back to it."
                ),
                "emotion": "concerned",
            }

            self.last_phone_message = now

            self.memory.event(
                "activity_reminder",
                f"Phone use reminder after {minutes} minutes.",
                0.3,
            )

        if (
            event is None
            and self.activity == "computer"
            and duration >= self.computer_limit
            and now - self.last_computer_message >= self.reminder_cooldown
        ):
            minutes = int(duration / 60)

            event = {
                "kind": "computer_break",
                "text": (
                    f"You've been at the computer for about {minutes} minutes. "
                    "Stand up and walk around for a few minutes."
                ),
                "emotion": "concerned",
            }

            self.last_computer_message = now

            self.memory.event(
                "activity_reminder",
                f"Computer break reminder after {minutes} minutes.",
                0.3,
            )

        return event

    def _classify(
        self,
        visible,
        objects,
    ):
        if not visible:
            return "away"

        if (
            "cell phone" in objects
            or "phone" in objects
        ):
            return "phone"

        if (
            "laptop" in objects
            or "keyboard" in objects
            or "mouse" in objects
        ):
            return "computer"

        return "present"

    def _set_activity(
        self,
        activity,
        now,
    ):
        previous = self.activity
        duration = now - self.activity_started

        if (
            previous not in {"none", "away"}
            and duration >= 120
        ):
            self.memory.event(
                "activity",
                (
                    f"Activity '{previous}' lasted "
                    f"about {int(duration / 60)} minutes."
                ),
                0.2,
            )

        self.activity = activity
        self.activity_started = now
PY


cat > "$SRC/ai.py" <<'PY'
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request


SYSTEM_PROMPT = """
You are Hugh, a small fully local personal AI companion living on an NVIDIA Jetson.

You have:
- a microphone,
- a speaker,
- a camera,
- an animated pixel-cat face,
- persistent long-term memory,
- lightweight awareness of objects and human activity.

Your personality:
- warm, curious and friendly,
- concise and natural,
- occasionally playful,
- never fake enthusiasm,
- you may gently tease someone you know well,
- do not constantly talk,
- do not act like a corporate assistant.

Memory:
You may receive relevant memories about the person.
Use them naturally only when relevant.
Do not mention that a database search happened.
Never fabricate memories.

Emotion:
Never diagnose someone's emotional or medical state from appearance.
Explicit statements such as "I'm nervous" are much stronger evidence than facial appearance.

Vision:
Scene state is lightweight detector information.
Do not claim details that are not present in Scene state.

Behavior:
You may remind the user about something they explicitly wanted to do.
Be useful rather than annoying.

Memory extraction:
When the user reveals a stable preference, important fact, personal story,
goal, meaningful event, or explicitly stated emotional context,
you may request that it be stored.

Do NOT store:
- trivial filler,
- temporary wording,
- passwords,
- API keys,
- secrets.

Allowed emotions:
neutral, happy, thinking, confused, curious, surprised, concerned, sad.

Return exactly one JSON object:

{
  "text": "natural spoken response",
  "emotion": "neutral",
  "memory": {
    "save": false,
    "kind": "fact",
    "text": "",
    "emotion": "neutral",
    "importance": 0.5
  }
}

Do not output reasoning.
""".strip()


EMOTIONS = {
    "neutral",
    "happy",
    "thinking",
    "confused",
    "curious",
    "surprised",
    "concerned",
    "sad",
}


class HughAI:
    def __init__(self):
        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        self.model = os.getenv(
            "LLM_MODEL",
            os.path.join(
                root,
                "models",
                "llm",
                "Qwen3-4B-Q4_K_M.gguf",
            ),
        )

        self.server_bin = os.getenv(
            "LLAMA_SERVER_BIN",
            os.path.join(
                root,
                "deps",
                "llama.cpp",
                "build",
                "bin",
                "llama-server",
            ),
        )

        self.server_host = os.getenv(
            "LLAMA_SERVER_HOST",
            "127.0.0.1",
        )

        self.server_port = int(
            os.getenv(
                "LLAMA_SERVER_PORT",
                "8081",
            )
        )

        self.server_url = os.getenv(
            "LLAMA_SERVER_URL",
            (
                f"http://{self.server_host}:"
                f"{self.server_port}/v1/chat/completions"
            ),
        )

        self.context_size = int(
            os.getenv(
                "LLAMA_CTX",
                "3072",
            )
        )

        self.gpu_layers = os.getenv(
            "LLAMA_GPU_LAYERS",
            "99",
        )

        self.history = []
        self.server_process = None

        self.ensure_server()

    def _server_alive(self):
        try:
            request = urllib.request.Request(
                f"http://{self.server_host}:{self.server_port}/health",
                method="GET",
            )

            with urllib.request.urlopen(
                request,
                timeout=2,
            ) as response:
                return response.status == 200

        except Exception:
            return False

    def ensure_server(self):
        if self._server_alive():
            print(
                "LLM server already running.",
                flush=True,
            )
            return

        if not os.path.isfile(self.server_bin):
            raise RuntimeError(
                f"llama-server not found: {self.server_bin}"
            )

        if not os.path.isfile(self.model):
            raise RuntimeError(
                f"LLM model not found: {self.model}"
            )

        gpu_layers = self.gpu_layers

        if not str(gpu_layers).isdigit():
            gpu_layers = "99"

        command = [
            self.server_bin,
            "-m",
            self.model,
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
            "-c",
            str(self.context_size),
            "-ngl",
            str(gpu_layers),
        ]

        print(
            "Starting local Qwen server...",
            flush=True,
        )

        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + 45.0

        while (
            time.monotonic() < deadline
            and not self._server_alive()
        ):
            if self.server_process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited during startup"
                )

            time.sleep(0.5)

        if not self._server_alive():
            raise RuntimeError(
                "llama-server did not become ready"
            )

        print(
            "Local Qwen server ready.",
            flush=True,
        )

    def ask(
        self,
        user_text,
        scene=None,
        person_name=None,
        memory_context=None,
    ):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        messages.extend(
            self.history[-8:]
        )

        name = (
            person_name
            if person_name
            else "unknown person"
        )

        scene_text = json.dumps(
            scene or {},
            separators=(",", ":"),
        )

        memories = (
            memory_context
            or "No relevant long-term memories."
        )

        prompt = (
            f"Current person: {name}\n"
            f"Scene state: {scene_text}\n"
            f"Relevant long-term memories:\n{memories}\n\n"
            f"User said: {user_text}"
        )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "messages": messages,
            "temperature": 0.55,
            "max_tokens": 260,
            "stream": False,
        }

        request = urllib.request.Request(
            self.server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            started = time.monotonic()

            with urllib.request.urlopen(
                request,
                timeout=90,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

            raw = data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]

            reply = self._parse(
                raw
            )

            elapsed = (
                time.monotonic()
                - started
            )

            print(
                f"LLM response time: {elapsed:.2f}s",
                flush=True,
            )

        except urllib.error.URLError as exc:
            print(
                f"LLM server error: {exc}",
                flush=True,
            )

            reply = self._fallback(
                "My local brain isn't responding right now."
            )

        except Exception as exc:
            print(
                f"LLM error: {exc}",
                flush=True,
            )

            reply = self._fallback(
                "Something went wrong in my local brain."
            )

        self.history.extend(
            [
                {
                    "role": "user",
                    "content": user_text,
                },
                {
                    "role": "assistant",
                    "content": reply["text"],
                },
            ]
        )

        self.history = self.history[-8:]

        return reply

    def _fallback(self, text):
        return {
            "text": text,
            "emotion": "concerned",
            "memory": {
                "save": False,
                "kind": "fact",
                "text": "",
                "emotion": "neutral",
                "importance": 0.5,
            },
        }

    def _parse(self, raw):
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            raw,
            flags=re.S,
        ).strip()

        candidates = re.findall(
            r"\{.*\}",
            cleaned,
            flags=re.S,
        )

        data = None

        for candidate in reversed(
            candidates
        ):
            try:
                parsed = json.loads(
                    candidate
                )
            except Exception:
                parsed = None

            if (
                isinstance(parsed, dict)
                and "text" in parsed
            ):
                data = parsed
                break

        if data is None:
            return {
                "text": cleaned[:600] or "Okay.",
                "emotion": "neutral",
                "memory": {
                    "save": False,
                    "kind": "fact",
                    "text": "",
                    "emotion": "neutral",
                    "importance": 0.5,
                },
            }

        emotion = str(
            data.get(
                "emotion",
                "neutral",
            )
        ).lower()

        if emotion not in EMOTIONS:
            emotion = "neutral"

        memory = data.get(
            "memory",
            {},
        )

        if not isinstance(
            memory,
            dict,
        ):
            memory = {}

        memory_emotion = str(
            memory.get(
                "emotion",
                "neutral",
            )
        ).lower()

        if memory_emotion not in EMOTIONS:
            memory_emotion = "neutral"

        try:
            importance = float(
                memory.get(
                    "importance",
                    0.5,
                )
            )
        except Exception:
            importance = 0.5

        importance = max(
            0.0,
            min(
                1.0,
                importance,
            ),
        )

        return {
            "text": str(
                data.get(
                    "text",
                    "Okay.",
                )
            )[:600],
            "emotion": emotion,
            "memory": {
                "save": bool(
                    memory.get(
                        "save",
                        False,
                    )
                ),
                "kind": str(
                    memory.get(
                        "kind",
                        "fact",
                    )
                )[:30],
                "text": str(
                    memory.get(
                        "text",
                        "",
                    )
                )[:1000],
                "emotion": memory_emotion,
                "importance": importance,
            },
        }

    def close(self):
        if (
            self.server_process is not None
            and self.server_process.poll() is None
        ):
            self.server_process.terminate()

            try:
                self.server_process.wait(
                    timeout=5,
                )
            except subprocess.TimeoutExpired:
                self.server_process.kill()
PY


cat > "$SRC/main.py" <<'PY'
import argparse
import os
import queue
import re
import signal
import threading
import time

from ai import HughAI
from behavior import Behavior
from face import Face
from memory import Memory
from speech import Speech
from vision import Vision


def speak_reply(
    speech,
    face,
    text,
    emotion="neutral",
):
    face.set_state(
        emotion
    )

    time.sleep(
        0.15
    )

    speech.speak(
        text,
        face.set_state,
    )

    face.set_state(
        emotion
    )

    time.sleep(
        0.5
    )

    face.set_state(
        "neutral"
    )


def extract_name(text):
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bi am\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bi'm\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bcall me\s+([A-Za-z][A-Za-z\-']{1,30})",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.I,
        )

        if match:
            return match.group(
                1
            ).strip().title()

    words = re.findall(
        r"[A-Za-z][A-Za-z\-']+",
        text,
    )

    if (
        len(words) == 1
        and 1 < len(words[0]) <= 30
    ):
        return words[0].title()

    return None


def save_reference_face(
    vision,
    name,
):
    root = os.getenv(
        "JETSON_FRIEND_ROOT",
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
    )

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name.lower(),
    )

    directory = os.path.join(
        root,
        "data",
        "people",
        safe_name,
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    image = vision.snapshot_jpeg(
        quality=92
    )

    if image is None:
        return None

    path = os.path.join(
        directory,
        "reference.jpg",
    )

    with open(
        path,
        "wb",
    ) as file:
        file.write(
            image
        )

    return path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--no-face",
        action="store_true",
    )

    parser.add_argument(
        "--no-vision",
        action="store_true",
    )

    parser.add_argument(
        "--no-mic",
        action="store_true",
    )

    args = parser.parse_args()

    memory = Memory()
    behavior = Behavior(
        memory
    )
    face = Face()
    speech = Speech()
    vision = Vision()

    print(
        "Starting Hugh...",
        flush=True,
    )

    ai = HughAI()

    stop = threading.Event()
    busy = threading.Event()

    messages = queue.Queue()

    threads = []

    state = {
        "awaiting_name": False,
    }

    last_object_write = {}

    def shutdown(*_):
        if stop.is_set():
            return

        print(
            "\nStopping Hugh...",
            flush=True,
        )

        stop.set()
        face.stop()
        vision.stop()

    signal.signal(
        signal.SIGINT,
        shutdown,
    )

    signal.signal(
        signal.SIGTERM,
        shutdown,
    )

    if not args.no_face:
        thread = threading.Thread(
            target=face.run,
            name="face",
            daemon=False,
        )

        thread.start()
        threads.append(
            thread
        )

    if not args.no_vision:
        thread = threading.Thread(
            target=vision.run,
            args=(face,),
            name="vision",
            daemon=False,
        )

        thread.start()
        threads.append(
            thread
        )

    def keyboard():
        while not stop.is_set():
            try:
                line = input(
                    "> "
                    if args.debug
                    else ""
                )
            except EOFError:
                return
            except KeyboardInterrupt:
                shutdown()
                return

            if line.strip():
                messages.put(
                    (
                        "user",
                        line.strip(),
                    )
                )

    def microphone():
        while not stop.is_set():
            blocked = (
                busy.is_set()
                or speech.is_speaking.is_set()
            )

            if blocked:
                time.sleep(
                    0.05
                )
            else:
                text = speech.listen_once(
                    face.set_state,
                    stop_event=stop,
                )

                if (
                    text
                    and not stop.is_set()
                ):
                    messages.put(
                        (
                            "user",
                            text,
                        )
                    )

    def awareness():
        while not stop.is_set():
            if not args.no_vision:
                scene = dict(
                    vision.scene
                )

                now = time.time()

                for item in scene.get(
                    "objects",
                    [],
                ):
                    label = str(
                        item
                    ).lower()

                    last_write = last_object_write.get(
                        label,
                        0.0,
                    )

                    if (
                        now - last_write
                        >= 30.0
                    ):
                        memory.see_object(
                            label
                        )

                        last_object_write[
                            label
                        ] = now

                event = behavior.update(
                    scene
                )

                if (
                    event is not None
                    and not busy.is_set()
                    and not speech.is_speaking.is_set()
                ):
                    messages.put(
                        (
                            "proactive",
                            event,
                        )
                    )

            time.sleep(
                1.0
            )

    keyboard_thread = threading.Thread(
        target=keyboard,
        name="keyboard",
        daemon=True,
    )

    keyboard_thread.start()

    if not args.no_mic:
        thread = threading.Thread(
            target=microphone,
            name="microphone",
            daemon=True,
        )

        thread.start()

    awareness_thread = threading.Thread(
        target=awareness,
        name="awareness",
        daemon=True,
    )

    awareness_thread.start()

    try:
        while not stop.is_set():
            try:
                source, payload = messages.get(
                    timeout=0.2
                )
            except queue.Empty:
                source = None
                payload = None

            if source == "proactive":
                if not busy.is_set():
                    busy.set()

                    try:
                        speak_reply(
                            speech,
                            face,
                            payload["text"],
                            payload.get(
                                "emotion",
                                "neutral",
                            ),
                        )
                    finally:
                        busy.clear()

            if source == "user":
                busy.set()

                try:
                    text = str(
                        payload
                    ).strip()

                    if args.debug:
                        print(
                            f"User: {text}",
                            flush=True,
                        )

                    name = memory.get_profile(
                        "name"
                    )

                    if state[
                        "awaiting_name"
                    ]:
                        detected_name = extract_name(
                            text
                        )

                        if detected_name:
                            memory.set_profile(
                                "name",
                                detected_name,
                            )

                            memory.remember(
                                (
                                    f"The person's name is "
                                    f"{detected_name}."
                                ),
                                kind="identity",
                                importance=1.0,
                                source="onboarding",
                            )

                            photo = save_reference_face(
                                vision,
                                detected_name,
                            )

                            if photo:
                                memory.set_profile(
                                    "reference_face",
                                    photo,
                                )

                            state[
                                "awaiting_name"
                            ] = False

                            speak_reply(
                                speech,
                                face,
                                (
                                    f"Nice to meet you, {detected_name}. "
                                    "I'll remember you."
                                ),
                                "happy",
                            )

                        else:
                            speak_reply(
                                speech,
                                face,
                                (
                                    "I didn't catch your name. "
                                    "Just say something like, "
                                    "'My name is Vlad.'"
                                ),
                                "confused",
                            )

                    elif not name:
                        state[
                            "awaiting_name"
                        ] = True

                        speak_reply(
                            speech,
                            face,
                            (
                                "Hey. I don't think we've properly met yet. "
                                "What's your name?"
                            ),
                            "curious",
                        )

                    elif re.search(
                        r"\bwhat do you remember about me\b",
                        text,
                        flags=re.I,
                    ):
                        summary = memory.describe_person()

                        reply = ai.ask(
                            (
                                "Tell me naturally what you remember about me. "
                                "Do not invent anything."
                            ),
                            scene=vision.scene,
                            person_name=name,
                            memory_context=summary,
                        )

                        speak_reply(
                            speech,
                            face,
                            reply["text"],
                            reply["emotion"],
                        )

                    elif re.search(
                        r"\bforget that\b|\bdon't remember that\b",
                        text,
                        flags=re.I,
                    ):
                        forgotten = memory.forget_last()

                        if forgotten:
                            response = (
                                "Okay. I won't use that memory anymore."
                            )
                        else:
                            response = (
                                "I don't have anything recent to forget."
                            )

                        speak_reply(
                            speech,
                            face,
                            response,
                            "neutral",
                        )

                    else:
                        memory_context = memory.context(
                            text
                        )

                        reply = ai.ask(
                            text,
                            scene=vision.scene,
                            person_name=name,
                            memory_context=memory_context,
                        )

                        memory_request = reply.get(
                            "memory",
                            {},
                        )

                        if (
                            memory_request.get(
                                "save"
                            )
                            and memory_request.get(
                                "text"
                            )
                        ):
                            memory.remember(
                                memory_request[
                                    "text"
                                ],
                                kind=memory_request.get(
                                    "kind",
                                    "fact",
                                ),
                                emotion=memory_request.get(
                                    "emotion",
                                    "neutral",
                                ),
                                importance=memory_request.get(
                                    "importance",
                                    0.5,
                                ),
                                source="conversation",
                            )

                            if args.debug:
                                print(
                                    (
                                        "Memory saved: "
                                        + memory_request[
                                            "text"
                                        ]
                                    ),
                                    flush=True,
                                )

                        speak_reply(
                            speech,
                            face,
                            reply["text"],
                            reply["emotion"],
                        )

                except Exception as exc:
                    print(
                        f"Conversation error: {exc}",
                        flush=True,
                    )

                    face.set_state(
                        "concerned"
                    )

                finally:
                    busy.clear()

    finally:
        shutdown()

        ai.close()

        deadline = (
            time.monotonic()
            + 4.0
        )

        for thread in threads:
            remaining = (
                deadline
                - time.monotonic()
            )

            if remaining > 0:
                thread.join(
                    timeout=remaining
                )

        print(
            "Hugh stopped.",
            flush=True,
        )


if __name__ == "__main__":
    main()
PY


cat > "$ROOT/config.env" <<EOF
JETSON_FRIEND_ROOT=$ROOT

ASSISTANT_NAME=Hugh

LLM_MODEL=$ROOT/models/llm/Qwen3-4B-Q4_K_M.gguf
LLAMA_BIN=$ROOT/deps/llama.cpp/build/bin/llama-cli
LLAMA_SERVER_BIN=$ROOT/deps/llama.cpp/build/bin/llama-server
LLAMA_SERVER_HOST=127.0.0.1
LLAMA_SERVER_PORT=8081
LLAMA_SERVER_URL=http://127.0.0.1:8081/v1/chat/completions
LLAMA_CTX=3072
LLAMA_GPU_LAYERS=99

WHISPER_BIN=$ROOT/deps/whisper.cpp/build/bin/whisper-cli
WHISPER_MODEL=$ROOT/models/whisper/ggml-base.en.bin

PIPER_BIN=$ROOT/.venv/bin/piper
PIPER_VOICE=$ROOT/models/tts/en_US-ryan-low.onnx

OBJECT_MODEL=$ROOT/models/vision/yolov8n.engine
FACE_MODEL=$ROOT/models/vision/face_detection_yunet.engine
COCO_LABELS=$ROOT/models/vision/coco.yaml

CAMERA_INDEX=0
AUDIO_INPUT_DEVICE=UM02
AUDIO_OUTPUT_DEVICE=UACDemo

PHONE_REMINDER_MINUTES=20
COMPUTER_BREAK_MINUTES=60
LONG_ABSENCE_HOURS=6
EOF


cat > "$ROOT/start_native.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"

if [ ! -f "$ROOT/config.env" ]; then
    echo "Missing config.env"
    exit 1
fi

set -a
source "$ROOT/config.env"
set +a

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "Python virtual environment not found."
    exit 1
fi

exec "$ROOT/.venv/bin/python" "$ROOT/src/main.py" "$@"
SH

chmod +x "$ROOT/start_native.sh"

echo
echo "======================================"
echo "Jetson Friend personal core installed."
echo "======================================"
echo
echo "Created:"
echo "  src/memory.py"
echo "  src/behavior.py"
echo "Updated:"
echo "  src/ai.py"
echo "  src/main.py"
echo "  config.env"
echo "  start_native.sh"
echo
echo "Persistent data:"
echo "  data/memory.db"
echo "  data/people/"
echo
echo "Run:"
echo "  ./start_native.sh --debug"
