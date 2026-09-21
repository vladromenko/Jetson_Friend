import tempfile
import unittest
from pathlib import Path

from milo.intents import visual_intent
from milo.llm import LocalModel
from milo.memory import ObjectMemory
from milo.ros_runtime import RosRuntime


class FakeModel(LocalModel):
    def __init__(self, answer):
        super().__init__("http://localhost:1", "test-model")
        self.answers = answer if isinstance(answer, list) else [answer]
        self.payload = None

    @property
    def answer(self):
        return self.answers[-1]

    @answer.setter
    def answer(self, value):
        self.answers = [value]

    def _post(self, payload, timeout=None):
        self.payload = payload
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return {"choices": [{"message": {"content": answer}}]}


class AssistantTests(unittest.TestCase):
    def test_j1_sign_from_rc9_log(self):
        # With J1 falling from 104 to 80, face x rose from 0.54 to 0.90.
        from milo.tracking import FaceFollower
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            pan_sign=1, pan_joint=1, target_x=0.5, target_y=0.5,
            deadband_x=0.08, deadband_y=0.09, max_step_deg=2,
            tilt_joint_a=3, tilt_sign_a=1, tilt_joint_b=4, tilt_sign_b=1,
        )
        joint, target = FaceFollower(cfg).plan(0.75, 0.5, 0, {1: 104}, {1})
        self.assertEqual((joint, target), (1, 106))

    def test_motion_feedback_does_not_ack_old_degree(self):
        ros = RosRuntime.__new__(RosRuntime)
        ros.latest_raw = {}
        ros._pending_motion = {1: {
            "before": 104, "target": 103, "min_complete": 1.2,
            "deadline": 2.0, "movement_logged": False,
        }}
        ros._check_motion_feedback({1: 104}, now=0.1)
        self.assertTrue(ros.arm_busy())
        ros._check_motion_feedback({1: 104}, now=1.3)
        self.assertTrue(ros.arm_busy())
        ros._check_motion_feedback({1: 103}, now=1.4)
        self.assertFalse(ros.arm_busy())

    def test_visual_intents(self):
        self.assertEqual(visual_intent("Where did I leave my glasses?"), ("object", "glasses"))
        self.assertEqual(visual_intent("Have you seen my keys?"), ("object", "keys"))
        self.assertEqual(visual_intent("Где я оставил очки?"), ("object", "очки"))
        self.assertEqual(visual_intent("What do you see?"), ("scene", ""))
        self.assertEqual(visual_intent("I feel sad"), ("chat", ""))

    def test_memory_only_records_confirmed_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects.json"
            memory = ObjectMemory(path)
            self.assertIsNone(memory.recall("glasses"))
            memory.remember("glasses", "on the desk next to the lamp")
            self.assertEqual(ObjectMemory(path).recall("glasses")["location"], "on the desk next to the lamp")

    def test_visual_locator_requires_visible_json(self):
        model = FakeModel('{"object":"glasses","visible":true,"location":"on the desk"}')
        result = model.locate_object("Where are my glasses?", "glasses", b"jpeg")
        self.assertEqual(result["location"], "on the desk")
        self.assertTrue(result["visible"])
        model.answer = '{"object":"glasses","visible":false,"location":""}'
        self.assertFalse(model.locate_object("Where are my glasses?", "glasses", b"jpeg")["visible"])
        model.answer = "Maybe near the desk"
        self.assertFalse(model.locate_object("Where are my glasses?", "glasses", b"jpeg")["reliable"])
        model.answer = '{"object":"headphones","visible":true,"location":"on the sofa"}'
        self.assertFalse(model.locate_object("Where are my glasses?", "glasses", b"jpeg")["visible"])

    def test_emotional_prompt(self):
        model = FakeModel("That sounds painful. Want to tell me what happened?")
        model.chat("I'm feeling sad")
        self.assertIn("vulnerable or painful feeling", model.payload["messages"][1]["content"])
        self.assertIn("How can I help you today", model.payload["messages"][0]["content"])

    def test_generic_emotional_reply_gets_llm_rewrite(self):
        model = FakeModel(["How can I help you today?", "That sounds really heavy. You do not have to solve it all at once; take one breath with me."])
        answer = model.chat("I'm sad")
        self.assertIn("really heavy", answer)
        self.assertNotIn("How can I help", answer)

    def test_proactive_prompt_forces_english(self):
        model = FakeModel("Hi, I'm MILO. What can I help you with?")
        answer = model.proactive_reply("Start the conversation.")
        self.assertIn("clear spoken English", model.payload["messages"][0]["content"])
        self.assertIn("Hi", answer)


if __name__ == "__main__":
    unittest.main()
