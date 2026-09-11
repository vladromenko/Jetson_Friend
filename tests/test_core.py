import io
import json
import threading
import time

import numpy as np
import pytest

from ai import HughAI
from behavior import Behavior
from identity import IdentityManager
from memory import Memory
from memory_policy import candidate, private_turn
from presence import Presence
from streaming import deltas, Phrases
from tracking import FaceTracks
from vision import Vision


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("JETSON_FRIEND_ROOT", str(tmp_path))
    return Memory()


def test_unknown_does_not_inherit_current_person(memory):
    pid = memory.create_person("Vlad")
    memory.remember("My robotics exam is Friday", person_id=pid)
    assert memory.search("robotics exam", person_id=None) == []
    assert memory.remember("Unknown person's robotics secret", person_id=None) is None
    assert memory.get_profile("name", person_id=None) is None
    assert memory.forget_last(person_id=None) is None
    assert len(memory.search("robotics exam", person_id=pid)) == 1


def test_same_name_is_not_identity(memory):
    a, b = memory.create_person("Anna"), memory.create_person("Anna")
    assert a != b
    memory.remember("I prefer coffee", person_id=a)
    memory.remember("I prefer tea", person_id=b)
    assert "coffee" not in memory.context("prefer coffee", person_id=b)


def test_global_is_explicit_and_legacy_is_not_public(memory):
    with memory._connect() as db:
        db.execute(
            "INSERT INTO memories(created,updated,kind,text,importance,active) VALUES (0,0,'fact','secret coffee',1,1)"
        )
    memory.remember("Charging station near wall", source="global")
    assert not memory.search("secret coffee")
    assert memory.search("charging station")


def test_restart_does_not_restore_person(memory):
    memory.create_person("Vlad")
    restarted = Memory()
    assert restarted.get_current_person_id() is None


def test_delete_erases_all_person_rows(memory):
    pid = memory.create_person("Vlad")
    identity = IdentityManager(memory)
    memory.remember("I prefer coffee", person_id=pid)
    memory.event("episode", "private", person_id=pid)
    memory.set_temporary_state("emotion", "sad", person_id=pid)
    identity.add_embedding(pid, np.eye(1, 128)[0])
    assert memory.forget_person(pid)
    with memory._connect() as db:
        for table in (
            "memories",
            "events",
            "temporary_states",
            "person_profiles",
            "person_face_embeddings",
        ):
            assert (
                db.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE person_id=?", (pid,)
                ).fetchone()[0]
                == 0
            )
    assert memory.get_person(pid) is None


def test_dedup_and_temporary_expiry(memory):
    pid = memory.create_person("Vlad")
    a = memory.remember("I prefer coffee.", person_id=pid)
    b = memory.remember("  I prefer COFFEE! ", person_id=pid)
    assert a == b
    memory.set_temporary_state("emotion", "sad", person_id=pid, ttl_seconds=-1)
    with memory._connect() as db:
        db.execute(
            "UPDATE temporary_states SET expires_at=? WHERE person_id=?",
            (time.time() - 1, pid),
        )
    assert memory.get_temporary_state("emotion", person_id=pid) is None


def test_presence_hysteresis_and_real_return():
    presence = Presence(confirm=1, absence=5)
    seen = [{"person_id": "a", "track_id": 1}]
    assert presence.update(seen, 0) == []
    assert presence.update(seen, 1)[0]["kind"] == "known_person_seen"
    assert presence.update([], 2) == []
    assert presence.people["a"].status == "TEMPORARILY_LOST"
    assert presence.update(seen, 3) == []
    assert presence.update([], 9)[0]["kind"] == "person_left"
    assert presence.update(seen, 100) == []
    event = presence.update(seen, 101)[0]
    assert event["kind"] == "person_returned" and event["absence"] == 98


def test_people_have_separate_presence():
    presence = Presence(confirm=0, absence=2)
    presence.update([{"person_id": "a"}, {"person_id": "b"}], 0)
    events = presence.update([{"person_id": "b"}], 3)
    assert [e["person_id"] for e in events] == ["a"]
    assert presence.people["b"].status == "VISIBLE"


def test_behavior_silence_and_delivery_cooldown(memory):
    pid = memory.create_person("Vlad")
    behavior = Behavior(memory)
    behavior.presence.confirm = 0
    behavior.long_absence = 5
    scene = {"updated": time.time(), "people": [{"person_id": pid}]}
    assert behavior.update(scene, now=0) is None
    assert behavior.update({"updated": time.time(), "people": []}, now=6) is None
    assert behavior.update(scene, now=20, busy=True) is None
    assert behavior.update(scene, now=21, listening=True) is None
    event = behavior.update(scene, now=22)
    assert event and event["person_id"] == pid
    behavior.acknowledge(event)
    assert behavior.update(scene, now=23) is None


def test_phone_detection_does_not_imply_use(memory):
    behavior = Behavior(memory)
    for tick in (0, 60, 1800, 10000):
        assert (
            behavior.update(
                {"updated": time.time(), "objects": ["phone"], "people": []}, now=tick
            )
            is None
        )


def test_tracks_reset_on_miss_and_ambiguity():
    tracks = FaceTracks()
    box = {"box": [0, 0, 1, 1]}
    a = tracks.update([box], 0)[0]["track_id"]
    assert tracks.update([box], 0.2)[0]["track_id"] == a
    crossing = tracks.update([box, box], 0.4)
    assert all(d["track_id"] != a for d in crossing)
    tracks.update([], 0.5)
    assert tracks.update([box], 0.6)[0]["track_id"] not in [
        d["track_id"] for d in crossing
    ]


def test_stale_or_multiple_face_crops_are_not_enrollment_samples():
    vision = Vision.__new__(Vision)
    vision.frame_lock = threading.RLock()
    sample = (
        np.ones((80, 80, 3), dtype=np.uint8),
        {"quality": 1, "captured": time.time() - 3},
    )
    vision.face_samples = {1: sample}
    assert vision.get_face_crop() == (None, None)
    vision.face_samples = {1: sample, 2: sample}
    assert vision.get_face_crop() == (None, None)


def test_blank_face_quality_is_rejected():
    assert (
        Vision._face_quality(
            np.zeros((120, 120, 3), dtype=np.uint8), {"confidence": 1, "area": 1}
        )
        == 0
    )


def test_ambiguous_strong_matches_remain_unknown(memory, monkeypatch):
    identity = IdentityManager(memory)
    a, b = memory.create_person("a"), memory.create_person("b")
    monkeypatch.setattr(
        identity,
        "_person_scores",
        lambda _: [
            {"person_id": a, "score": 0.90, "embedding_id": 1},
            {"person_id": b, "score": 0.89, "embedding_id": 2},
        ],
    )
    assert identity.recognize_embedding(np.ones(128)).status == "UNKNOWN"
    assert identity._normalize_embedding(np.full(128, np.nan)) is None


def test_histories_are_uuid_scoped_and_unknown_is_ephemeral():
    ai = HughAI.__new__(HughAI)
    ai.history_lock = threading.RLock()
    ai.histories = {}
    ai.history_turns = 2
    ai._append_history("uuid-a", "secret", "okay")
    ai._append_history(None, "anonymous secret", "okay")
    assert ai._get_history("uuid-b") == []
    assert ai._get_history(None) == []
    ai.clear_history("uuid-a")
    assert ai._get_history("uuid-a") == []


@pytest.mark.parametrize(
    "raw", ['{"text":"unfinished', "<think>hidden reasoning", '{"memory":']
)
def test_broken_json_is_never_spoken(raw):
    ai = HughAI.__new__(HughAI)
    reply = ai._parse(raw)
    assert "unfinished" not in reply["text"]
    assert "hidden reasoning" not in reply["text"]
    assert not reply["text"].startswith("{")


def test_sse_utf8_and_truncation():
    event = {"choices": [{"delta": {"content": "Hi — Milo."}}]}
    wire = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
    assert list(deltas(io.BytesIO(wire))) == [event]
    with pytest.raises(RuntimeError):
        list(deltas(io.BytesIO(wire.split(b"data: [DONE]")[0])))


def test_phrase_boundary_and_reasoning_guard():
    phrases = Phrases()
    assert phrases.feed("Hi there.") == []
    assert phrases.feed(" How are you?") == ["Hi there."]
    assert phrases.feed("", final=True) == ["How are you?"]
    with pytest.raises(ValueError):
        Phrases().feed("<think>let me think")


@pytest.mark.parametrize(
    "text",
    [
        "Okay.",
        "I'm tired right now.",
        "Don't remember this: I prefer tea.",
        "That's funny.",
    ],
)
def test_selective_memory_ignores_transient_or_private(text):
    assert candidate(text) is None


def test_selective_memory_accepts_preference():
    assert candidate("I prefer coffee without sugar.")["kind"] == "preference"
    assert private_turn("Do not remember this")


def test_schedule_update_replaces_old_fact(memory):
    pid = memory.create_person("Vlad")
    for text in ("My exam is on Friday.", "My exam was moved to Monday."):
        memory.remember(**candidate(text), person_id=pid)
    rows = memory.search("exam", person_id=pid)
    assert len(rows) == 1 and "Monday" in rows[0]["text"]


def test_new_embeddings_are_separate_from_legacy(memory):
    pid = memory.create_person("Vlad")
    identity = IdentityManager(memory)
    embedding = np.eye(1, 128)[0]
    assert identity.add_embedding(pid, embedding)
    assert identity.embedding_count(pid) == 1
    assert len(identity._load_embeddings()) == 1
    with memory._connect() as db:
        db.execute(
            "UPDATE person_face_embeddings SET representation='legacy-unaligned'"
        )
    assert identity._load_embeddings() == []
    assert identity.embedding_count(pid) == 0


@pytest.mark.parametrize(
    "text",
    ["See you later.", "I look forward to Friday.", "My camera exam is tomorrow."],
)
def test_nonvisual_language_does_not_load_vlm(text):
    from intents import wants_visual

    assert not wants_visual(text)


def test_explicit_visual_question():
    from intents import wants_visual

    assert wants_visual("What am I holding?")


def test_microphone_negotiates_supported_rate_and_caches(monkeypatch):
    import speech as module

    speech = module.Speech.__new__(module.Speech)
    checked = []
    monkeypatch.setattr(
        module.sd, "query_devices", lambda *args: {"default_samplerate": 48000}
    )

    def check(**settings):
        checked.append(settings["samplerate"])
        if settings["samplerate"] == 44100:
            raise module.sd.PortAudioError("unsupported rate")

    monkeypatch.setattr(module.sd, "check_input_settings", check)
    assert speech._capture_rate(0, 44100) == 16000
    assert speech._capture_rate(0, 44100) == 16000
    assert checked == [44100, 16000]


def test_speaker_uses_matching_shared_sink_when_desktop_owns_audio(monkeypatch):
    import speech as module
    from types import SimpleNamespace

    speaker = module.Speech()
    speaker.output_hint = "UACDemo"
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="51 alsa_output.hdmi PipeWire\n53 alsa_output.usb-UACDemo.stereo PipeWire\n",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    assert speaker._find_output_device() == "pipewire:alsa_output.usb-UACDemo.stereo"
    assert speaker._find_output_device() == "pipewire:alsa_output.usb-UACDemo.stereo"
    assert calls == [["pactl", "list", "short", "sinks"]]


def test_speaker_falls_back_to_alsa_without_audio_server(monkeypatch):
    import speech as module
    from types import SimpleNamespace

    speaker = module.Speech()
    speaker.output_hint = "UACDemo"
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/" + name)

    def run(command, **kwargs):
        if command[0] == "pactl":
            return SimpleNamespace(returncode=1, stdout="")
        return SimpleNamespace(returncode=0, stdout="card 1: UACDemo [USB Audio]")

    monkeypatch.setattr(module.subprocess, "run", run)
    assert speaker._find_output_device() == "plughw:1,0"
    speaker.output_hint = "hw:2,0"
    assert speaker._find_output_device(refresh=True) == "hw:2,0"


def test_matching_requires_multiple_references_and_returns_uuid(memory):
    pid = memory.create_person("Anna")
    identity = IdentityManager(memory)
    reference = np.zeros(128, dtype=np.float32)
    reference[0] = 1
    identity.add_embedding(pid, reference)
    assert identity.recognize_embedding(reference).status == "UNKNOWN"
    for axis in (1, 2):
        variant = reference.copy()
        variant[axis] = 0.5
        assert identity.add_embedding(pid, variant)
    result = identity.recognize_embedding(reference)
    assert result.status == "KNOWN"
    assert result.person_id == pid
