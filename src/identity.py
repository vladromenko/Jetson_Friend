import json
import os
import sqlite3
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class IdentityResult:
    person_id: str | None
    name: str | None
    confidence: float
    similarity: float
    status: str

    def as_dict(self):
        return {
            "person_id": self.person_id,
            "name": self.name,
            "confidence": self.confidence,
            "similarity": self.similarity,
            "status": self.status,
        }


class IdentityManager:
    """
    Lightweight multi-user face identity manager for MILO.

    - recognition is event-driven;
    - several embeddings can belong to one person;
    - weak matches remain UNKNOWN;
    - all biometric data stays local;
    - integrates with Memory and Vision.
    """

    def __init__(self, memory, model_path=None):
        self.memory = memory
        self.lock = memory.lock

        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        self.model_path = model_path or os.getenv(
            "FACE_RECOGNITION_MODEL",
            os.path.join(
                root,
                "models",
                "vision",
                "face_recognition_sface.onnx",
            ),
        )

        self.match_threshold = float(
            os.getenv(
                "FACE_MATCH_THRESHOLD",
                "0.46",
            )
        )

        self.strong_match_threshold = float(
            os.getenv(
                "FACE_STRONG_MATCH_THRESHOLD",
                "0.58",
            )
        )

        self.ambiguity_margin = float(
            os.getenv(
                "FACE_AMBIGUITY_MARGIN",
                "0.06",
            )
        )

        self.min_face_quality = float(
            os.getenv(
                "FACE_MIN_QUALITY",
                "0.55",
            )
        )

        self.max_embeddings_per_person = int(
            os.getenv(
                "FACE_MAX_EMBEDDINGS",
                "12",
            )
        )

        self.recognition_cooldown = float(
            os.getenv(
                "FACE_RECOGNITION_COOLDOWN_SEC",
                "1.5",
            )
        )

        self.identity_hold_seconds = float(
            os.getenv(
                "FACE_IDENTITY_HOLD_SEC",
                "3.0",
            )
        )

        self.recognizer = None
        self.available = False
        self.backend = "disabled"

        self.track_results = {}
        self.last_recognition_at = 0.0

        self.last_result = IdentityResult(
            None,
            None,
            0.0,
            0.0,
            "UNKNOWN",
        )

        self.last_result_at = 0.0

        self._create_schema()
        with self._connect() as db:
            columns = {
                row[1]
                for row in db.execute("PRAGMA table_info(person_face_embeddings)")
            }
            if "representation" not in columns:
                db.execute(
                    "ALTER TABLE person_face_embeddings ADD COLUMN representation TEXT NOT NULL DEFAULT 'legacy-unaligned'"
                )
        self._load_recognizer()

    def _connect(self):
        db = sqlite3.connect(
            self.memory.path,
            timeout=10,
        )

        db.row_factory = sqlite3.Row

        db.execute("PRAGMA foreign_keys = ON")

        db.execute("PRAGMA journal_mode = WAL")

        db.execute("PRAGMA synchronous = NORMAL")

        return db

    def _create_schema(self):
        with self.lock:
            with self._connect() as db:
                db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS person_face_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        dimensions INTEGER NOT NULL,
                        quality REAL NOT NULL DEFAULT 1.0,
                        source TEXT,
                        created REAL NOT NULL,
                        last_used REAL,
                        active INTEGER NOT NULL DEFAULT 1
                    );

                    CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
                    ON person_face_embeddings(
                        person_id,
                        active,
                        quality DESC
                    );

                    CREATE INDEX IF NOT EXISTS idx_face_embeddings_active
                    ON person_face_embeddings(
                        active,
                        created DESC
                    );
                    """
                )

    def _load_recognizer(self):
        if not hasattr(
            cv2,
            "FaceRecognizerSF_create",
        ):
            print(
                "Identity disabled: OpenCV SFace API is unavailable.",
                flush=True,
            )
            return

        if not os.path.isfile(self.model_path):
            print(
                "Identity waiting for SFace model: " + self.model_path,
                flush=True,
            )
            return

        try:
            self.recognizer = cv2.FaceRecognizerSF_create(
                self.model_path,
                "",
            )

            self.available = True
            self.backend = "OpenCV SFace"

            print(
                f"Identity backend ready: {self.backend}",
                flush=True,
            )

        except Exception as exc:
            self.recognizer = None
            self.available = False
            self.backend = "disabled"

            print(
                f"Identity initialization failed: {exc}",
                flush=True,
            )

    @staticmethod
    def _normalize_embedding(
        embedding,
    ):
        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        norm = float(np.linalg.norm(vector))

        if not np.isfinite(vector).all() or norm <= 1e-8:
            return None

        return np.ascontiguousarray(
            vector / norm,
            dtype=np.float32,
        )

    def extract_embedding(self, face_crop, landmarks=None):
        if (
            not self.available
            or self.recognizer is None
            or face_crop is None
            or face_crop.size == 0
        ):
            return None
        if landmarks is None or len(landmarks) != 5:
            return None
        try:
            h, w = face_crop.shape[:2]
            row = np.asarray(
                [0, 0, w, h, *np.asarray(landmarks).reshape(-1), 1.0], dtype=np.float32
            )
            with self.lock:
                aligned = self.recognizer.alignCrop(face_crop, row)
                feature = self.recognizer.feature(aligned)
            return self._normalize_embedding(feature)
        except Exception as exc:
            print(f"Face embedding warning: {exc}", flush=True)
            return None

    @staticmethod
    def _blob_to_embedding(
        blob,
        dimensions,
    ):
        vector = np.frombuffer(
            blob,
            dtype=np.float32,
        )

        if vector.size != int(dimensions):
            return None

        return np.ascontiguousarray(
            vector.copy(),
            dtype=np.float32,
        )

    def embedding_count(
        self,
        person_id,
    ):
        if not person_id:
            return 0

        with self.lock:
            with self._connect() as db:
                row = db.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM person_face_embeddings
                    WHERE person_id = ?
                      AND active = 1 AND representation = 'sface-aligned-v1'
                    """,
                    (person_id,),
                ).fetchone()

        if row is None:
            return 0

        return int(row["count"])

    def add_embedding(
        self,
        person_id,
        embedding,
        quality=1.0,
        source="camera",
    ):
        if not person_id:
            return False

        person = self.memory.get_person(person_id)

        if person is None:
            return False

        normalized = self._normalize_embedding(embedding)

        if normalized is None:
            return False

        quality = max(
            0.0,
            min(
                1.0,
                float(quality),
            ),
        )

        if quality < self.min_face_quality:
            return False

        if self._is_duplicate_embedding(
            person_id,
            normalized,
        ):
            return False

        now = time.time()

        with self.lock:
            with self._connect() as db:
                if not db.execute(
                    "SELECT 1 FROM persons WHERE id=? AND active=1", (person_id,)
                ).fetchone():
                    return False
                db.execute(
                    """
                    INSERT INTO person_face_embeddings(
                        person_id,
                        embedding,
                        dimensions,
                        quality,
                        source,
                        created,
                        last_used,
                        active,
                        representation
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'sface-aligned-v1')
                    """,
                    (
                        person_id,
                        normalized.tobytes(),
                        int(normalized.size),
                        quality,
                        str(source)[:50],
                        now,
                        now,
                    ),
                )

                self._trim_embeddings_db(
                    db,
                    person_id,
                )

        return True

    def add_face(
        self,
        person_id,
        face_crop,
        quality=1.0,
        source="camera",
    ):
        embedding = self.extract_embedding(face_crop)

        if embedding is None:
            return False

        return self.add_embedding(
            person_id,
            embedding,
            quality=quality,
            source=source,
        )

    def _is_duplicate_embedding(
        self,
        person_id,
        candidate,
    ):
        rows = self._load_embeddings(person_id=person_id)

        duplicate = False

        for row in rows:
            similarity = self._cosine(
                candidate,
                row["embedding"],
            )

            if similarity >= 0.98:
                duplicate = True
                break

        return duplicate

    def _trim_embeddings_db(
        self,
        db,
        person_id,
    ):
        rows = db.execute(
            """
            SELECT id
            FROM person_face_embeddings
            WHERE person_id = ?
              AND active = 1 AND representation = 'sface-aligned-v1'
            ORDER BY
                quality DESC,
                created DESC
            """,
            (person_id,),
        ).fetchall()

        keep = max(
            3,
            self.max_embeddings_per_person,
        )

        if len(rows) > keep:
            remove_ids = [row["id"] for row in rows[keep:]]

            placeholders = ",".join("?" for _ in remove_ids)

            db.execute(
                f"""
                UPDATE person_face_embeddings
                SET active = 0
                WHERE id IN ({placeholders})
                """,
                tuple(remove_ids),
            )

    def _load_embeddings(
        self,
        person_id=None,
    ):
        with self.lock:
            with self._connect() as db:
                if person_id:
                    rows = db.execute(
                        """
                        SELECT
                            id,
                            person_id,
                            embedding,
                            dimensions,
                            quality,
                            created,
                            last_used
                        FROM person_face_embeddings
                        WHERE active = 1 AND representation = 'sface-aligned-v1'
                          AND person_id = ?
                        ORDER BY
                            quality DESC,
                            created DESC
                        """,
                        (person_id,),
                    ).fetchall()

                else:
                    rows = db.execute(
                        """
                        SELECT
                            id,
                            person_id,
                            embedding,
                            dimensions,
                            quality,
                            created,
                            last_used
                        FROM person_face_embeddings
                        WHERE active = 1 AND representation = 'sface-aligned-v1'
                        ORDER BY
                            person_id,
                            quality DESC,
                            created DESC
                        """
                    ).fetchall()

        result = []

        for row in rows:
            vector = self._blob_to_embedding(
                row["embedding"],
                row["dimensions"],
            )

            if vector is not None:
                result.append(
                    {
                        "id": row["id"],
                        "person_id": (row["person_id"]),
                        "embedding": vector,
                        "quality": float(row["quality"]),
                        "created": float(row["created"]),
                        "last_used": (row["last_used"]),
                    }
                )

        return result

    @staticmethod
    def _cosine(
        a,
        b,
    ):
        a = np.asarray(
            a,
            dtype=np.float32,
        ).reshape(-1)

        b = np.asarray(
            b,
            dtype=np.float32,
        ).reshape(-1)

        if a.size != b.size or a.size == 0:
            return -1.0

        a_norm = float(np.linalg.norm(a))

        b_norm = float(np.linalg.norm(b))

        if a_norm <= 1e-8 or b_norm <= 1e-8:
            return -1.0

        return float(
            np.dot(
                a,
                b,
            )
            / (a_norm * b_norm)
        )

    def _person_scores(
        self,
        candidate,
    ):
        rows = self._load_embeddings()

        grouped = {}

        for row in rows:
            person_id = row["person_id"]

            similarity = self._cosine(
                candidate,
                row["embedding"],
            )

            if person_id not in grouped:
                grouped[person_id] = []

            grouped[person_id].append(
                (
                    similarity,
                    row["quality"],
                    row["id"],
                )
            )

        scores = []

        for (
            person_id,
            samples,
        ) in grouped.items():
            if len(samples) < 3:
                continue
            ordered = sorted(
                samples,
                key=lambda item: item[0],
                reverse=True,
            )

            best = ordered[0][0]

            top = ordered[
                : min(
                    3,
                    len(ordered),
                )
            ]

            weighted_sum = 0.0
            weight_total = 0.0

            for (
                similarity,
                quality,
                embedding_id,
            ) in top:
                weight = max(
                    0.25,
                    float(quality),
                )

                weighted_sum += similarity * weight

                weight_total += weight

            average = weighted_sum / max(
                weight_total,
                1e-8,
            )

            score = best * 0.70 + average * 0.30

            scores.append(
                {
                    "person_id": (person_id),
                    "score": float(score),
                    "best": float(best),
                    "embedding_id": int(ordered[0][2]),
                }
            )

        scores.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scores

    def recognize_embedding(
        self,
        embedding,
    ):
        candidate = self._normalize_embedding(embedding)

        if candidate is None:
            return IdentityResult(
                None,
                None,
                0.0,
                0.0,
                "UNKNOWN",
            )

        scores = self._person_scores(candidate)

        if not scores:
            return IdentityResult(
                None,
                None,
                0.0,
                0.0,
                "UNKNOWN",
            )

        best = scores[0]

        if len(scores) > 1:
            second_score = scores[1]["score"]
        else:
            second_score = -1.0

        margin = best["score"] - second_score

        accepted = (
            best["score"] >= self.match_threshold and margin >= self.ambiguity_margin
        )

        if not accepted:
            confidence = self._similarity_confidence(best["score"])

            return IdentityResult(
                None,
                None,
                confidence,
                best["score"],
                "UNKNOWN",
            )

        person = self.memory.get_person(best["person_id"])

        if person is None:
            return IdentityResult(
                None,
                None,
                0.0,
                best["score"],
                "UNKNOWN",
            )

        confidence = self._similarity_confidence(best["score"])

        self._mark_embedding_used(best["embedding_id"])

        return IdentityResult(
            best["person_id"],
            person["name"],
            confidence,
            best["score"],
            "KNOWN",
        )

    def recognize_face(
        self,
        face_crop,
        quality=1.0,
        force=False,
    ):
        now = time.monotonic()

        if float(quality) < self.min_face_quality:
            return IdentityResult(
                None,
                None,
                0.0,
                0.0,
                "LOW_QUALITY",
            )

        if not self.available:
            return IdentityResult(
                None,
                None,
                0.0,
                0.0,
                "UNAVAILABLE",
            )

        if not force and (now - self.last_recognition_at < self.recognition_cooldown):
            if self.last_result.status == "KNOWN" and (
                now - self.last_result_at <= self.identity_hold_seconds
            ):
                return self.last_result

            return IdentityResult(
                None,
                None,
                0.0,
                0.0,
                "COOLDOWN",
            )

        self.last_recognition_at = now

        embedding = self.extract_embedding(face_crop)

        if embedding is None:
            result = IdentityResult(
                None,
                None,
                0.0,
                0.0,
                "UNKNOWN",
            )

        else:
            result = self.recognize_embedding(embedding)

        self.last_result = result
        self.last_result_at = now

        return result

    def recognize_from_vision(self, vision, force=False, track_id=None):
        crop, meta = vision.get_face_crop(
            min_quality=self.min_face_quality, track_id=track_id
        )
        unknown = IdentityResult(None, None, 0.0, 0.0, "UNKNOWN")
        if crop is None or meta is None:
            self.track_results.pop(track_id, None)
            return unknown
        track_id = meta["track_id"]
        now = time.monotonic()
        cached = self.track_results.get(track_id)
        if cached and not force and now - cached[0] < self.recognition_cooldown:
            return cached[1]
        embedding = self.extract_embedding(crop, meta.get("local_landmarks"))
        result = (
            self.recognize_embedding(embedding) if embedding is not None else unknown
        )
        self.track_results = {
            tid: value
            for tid, value in self.track_results.items()
            if now - value[0] < 30
        }
        self.track_results[track_id] = (now, result)
        self.last_result, self.last_result_at = result, now
        return result

    def _similarity_confidence(
        self,
        similarity,
    ):
        low = self.match_threshold - 0.10

        high = max(
            self.strong_match_threshold,
            self.match_threshold + 0.10,
        )

        normalized = (float(similarity) - low) / max(
            high - low,
            1e-6,
        )

        return round(
            max(
                0.0,
                min(
                    1.0,
                    normalized,
                ),
            ),
            3,
        )

    def _mark_embedding_used(
        self,
        embedding_id,
    ):
        with self.lock:
            with self._connect() as db:
                db.execute(
                    """
                    UPDATE person_face_embeddings
                    SET last_used = ?
                    WHERE id = ?
                    """,
                    (
                        time.time(),
                        int(embedding_id),
                    ),
                )

    def enroll_person_from_vision(
        self, name, vision, samples=5, timeout=8.0, person_id=None
    ):
        if not self.available or not person_id or not self.memory.get_person(person_id):
            return {"ok": False, "samples": 0, "reason": "profile_or_model_unavailable"}
        target = max(3, int(samples))
        deadline = time.monotonic() + max(3.0, float(timeout))
        track_id = None
        last_capture = None
        candidates = []
        while time.monotonic() < deadline and len(candidates) < target:
            crop, meta = vision.get_face_crop(min_quality=self.min_face_quality)
            if crop is None or meta is None:
                time.sleep(0.1)
                continue
            if track_id is None:
                track_id = meta["track_id"]
            if meta["track_id"] != track_id:
                return {"ok": False, "samples": 0, "reason": "person_changed"}
            if meta["captured"] != last_capture:
                last_capture = meta["captured"]
                embedding = self.extract_embedding(crop, meta.get("local_landmarks"))
                if embedding is not None:
                    if (
                        candidates
                        and self._cosine(embedding, candidates[0][0])
                        < self.match_threshold
                    ):
                        return {
                            "ok": False,
                            "samples": 0,
                            "reason": "inconsistent_faces",
                        }
                    if not any(
                        self._cosine(embedding, old[0]) >= 0.98 for old in candidates
                    ):
                        candidates.append((embedding, meta["quality"]))
            time.sleep(0.35)
        if len(candidates) < 3:
            return {"ok": False, "samples": 0, "reason": "not_enough_good_face_samples"}
        # Only commit after a consistent collection; no partial cross-person enrollment.
        for embedding, quality in candidates:
            self.add_embedding(
                person_id, embedding, quality=quality, source="onboarding"
            )
        count = self.embedding_count(person_id)
        return {
            "ok": count >= 3,
            "samples": count,
            "person_id": person_id,
            "name": name,
        }

    def reinforce_identity(
        self,
        person_id,
        vision,
    ):
        """
        Add a new high-quality reference only
        occasionally after a strong match.
        """

        crop, meta = vision.get_face_crop(
            min_quality=max(
                self.min_face_quality,
                0.65,
            )
        )

        if crop is None or meta is None:
            return False

        return self.add_face(
            person_id,
            crop,
            quality=float(
                meta.get(
                    "quality",
                    1.0,
                )
            ),
            source="reinforcement",
        )

    def delete_face_data(
        self,
        person_id,
    ):
        if not person_id:
            return False

        with self.lock:
            with self._connect() as db:
                result = db.execute(
                    """
                    DELETE FROM person_face_embeddings
                    WHERE person_id = ?
                    """,
                    (person_id,),
                )

        if self.last_result.person_id == person_id:
            self.clear_current_identity()

        return result.rowcount > 0

    def clear_current_identity(self):
        self.track_results.clear()
        self.last_result = IdentityResult(
            None,
            None,
            0.0,
            0.0,
            "UNKNOWN",
        )

        self.last_result_at = 0.0

    def status(self):
        people = self.memory.list_persons()

        return {
            "available": self.available,
            "backend": self.backend,
            "model_path": self.model_path,
            "known_people": len(people),
            "match_threshold": (self.match_threshold),
            "strong_match_threshold": (self.strong_match_threshold),
            "ambiguity_margin": (self.ambiguity_margin),
            "min_face_quality": (self.min_face_quality),
            "current_identity": (self.last_result.as_dict()),
        }

    def export_debug_summary(
        self,
    ):
        data = self.status()
        data["people"] = []

        for person in self.memory.list_persons():
            data["people"].append(
                {
                    "person_id": (person["id"]),
                    "name": (person["name"]),
                    "embeddings": (self.embedding_count(person["id"])),
                }
            )

        return json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
