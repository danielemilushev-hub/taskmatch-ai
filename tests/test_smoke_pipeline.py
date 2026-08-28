"""Fast pipeline smoke test: fake model responses through real grading.

No live model needed. A FakeCtx stands in for engine.RunContext and returns
canned ChatResults, so these tests exercise the actual grading logic
(_run_one, _execute_candidate) end to end in under a couple of seconds.
"""

import unittest

from localbench.engine import ChatResult
from localbench.runner import ModelSwitchError, _switch_to_model
from localbench.suites.json_schema_suite import _run_one as json_run_one
from localbench.suites.coding_suite import _execute_candidate, extract_code


class FakeCtx:
    """Mimics engine.RunContext.call, returning a canned ChatResult."""

    def __init__(self, result: ChatResult):
        self._result = result

    def call(self, messages, **overrides):
        return self._result


SCHEMA_PROBLEM = {
    "id": "smoke_schema_1",
    "task": "Produce a person object.",
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    },
}

CODING_PROBLEM = {
    "id": "smoke_code_1",
    "entry_point": "add_two",
    "tests": [
        {"args": [1, 2], "expected": 3},
        {"args": [-5, 5], "expected": 0},
    ],
}


class TestJsonSchemaGrading(unittest.TestCase):
    def test_valid_json_passes(self):
        ctx = FakeCtx(ChatResult(success=True, finish_reason="stop",
                                 content='{"name": "Ada", "age": 36}'))
        res = json_run_one(SCHEMA_PROBLEM, ctx, {})
        self.assertTrue(res.passed)

    def test_schema_violation_fails_with_path(self):
        ctx = FakeCtx(ChatResult(success=True, finish_reason="stop",
                                 content='{"name": "Ada", "age": -1}'))
        res = json_run_one(SCHEMA_PROBLEM, ctx, {})
        self.assertFalse(res.passed)
        self.assertIn("schema validation failed", res.error)

    def test_garbage_output_fails_cleanly(self):
        ctx = FakeCtx(ChatResult(success=True, finish_reason="stop",
                                 content="Sure! Here is your JSON: oops"))
        res = json_run_one(SCHEMA_PROBLEM, ctx, {})
        self.assertFalse(res.passed)

    def test_truncation_is_flagged_not_just_wrong(self):
        ctx = FakeCtx(ChatResult(success=True, finish_reason="length",
                                 content='{"name": "Ad'))
        res = json_run_one(SCHEMA_PROBLEM, ctx, {})
        self.assertFalse(res.passed)
        self.assertTrue(res.truncated)

    def test_call_failure_fails_cleanly(self):
        ctx = FakeCtx(ChatResult(success=False, error="connection refused"))
        res = json_run_one(SCHEMA_PROBLEM, ctx, {})
        self.assertFalse(res.passed)
        self.assertIn("call failed", res.error)


class TestCodingSandbox(unittest.TestCase):
    def test_correct_code_passes_all_tests(self):
        code = "def add_two(a, b):\n    return a + b"
        results, error = _execute_candidate(code, CODING_PROBLEM, timeout_seconds=15)
        self.assertIsNone(error)
        self.assertTrue(all(r["ok"] for r in results))

    def test_wrong_code_fails_tests(self):
        code = "def add_two(a, b):\n    return a - b"
        results, error = _execute_candidate(code, CODING_PROBLEM, timeout_seconds=15)
        self.assertIsNone(error)
        self.assertFalse(all(r["ok"] for r in results))

    def test_raising_code_reports_error_per_test(self):
        code = "def add_two(a, b):\n    raise ValueError('boom')"
        results, error = _execute_candidate(code, CODING_PROBLEM, timeout_seconds=15)
        self.assertIsNone(error)
        self.assertTrue(all("error" in r for r in results))

    def test_infinite_loop_hits_timeout(self):
        code = "def add_two(a, b):\n    while True:\n        pass"
        results, error = _execute_candidate(code, CODING_PROBLEM, timeout_seconds=3)
        self.assertIsNone(results)
        self.assertIn("timed out", error)

    def test_memory_bomb_is_contained(self):
        # POSIX: RLIMIT_AS raises MemoryError inside the child (per-test error).
        # Windows: the parent-side psutil watchdog kills the process (top-level
        # error). Either way the host must survive and the result must be a
        # clean failure, never a pass.
        code = (
            "def add_two(a, b):\n"
            "    chunks = []\n"
            "    while True:\n"
            "        chunks.append(bytearray(50 * 1024 * 1024))\n"
        )
        results, error = _execute_candidate(code, CODING_PROBLEM, timeout_seconds=30)
        if error is not None:
            self.assertTrue("memory" in error.lower() or "timed out" in error)
        else:
            self.assertFalse(any(r["ok"] for r in results))
            self.assertTrue(any("MemoryError" in r.get("error", "") for r in results))

    def test_network_access_is_blocked(self):
        code = (
            "def add_two(a, b):\n"
            "    import socket\n"
            "    socket.socket()\n"
            "    return a + b\n"
        )
        results, error = _execute_candidate(code, CODING_PROBLEM, timeout_seconds=15)
        self.assertIsNone(error)
        self.assertFalse(any(r["ok"] for r in results))
        self.assertTrue(any("network access is disabled" in r.get("error", "")
                            for r in results))

    def test_code_extraction_from_fenced_response(self):
        text = "Here you go:\n```python\ndef add_two(a, b):\n    return a + b\n```"
        self.assertIn("def add_two", extract_code(text))


class TestToolCallingGrading(unittest.TestCase):
    def setUp(self):
        from localbench.data.tool_calling_problems import generate_tool_calling_problems
        self.problems = generate_tool_calling_problems(num_problems=7, seed=42)

    def test_native_tool_call_passes(self):
        from localbench.suites.tool_calling_suite import _run_one
        prob = self.problems[0]  # weather problem
        expected_args = prob["expected_args"]
        ctx = FakeCtx(ChatResult(
            success=True,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": prob["expected_tool"],
                    "arguments": f'{{"location": "{expected_args["location"]}", "unit": "{expected_args["unit"]}", "days": {expected_args["days"]}}}',
                },
            }]
        ))
        res = _run_one(prob, ctx, {})
        self.assertTrue(res.passed, f"Expected pass, got error: {res.error}")

    def test_text_fallback_tool_call_passes(self):
        from localbench.suites.tool_calling_suite import _run_one
        prob = self.problems[0]
        expected_args = prob["expected_args"]
        content = f'<tool_call>{{"name": "{prob["expected_tool"]}", "arguments": {{"location": "{expected_args["location"]}", "unit": "{expected_args["unit"]}", "days": {expected_args["days"]}}}}}</tool_call>'
        ctx = FakeCtx(ChatResult(success=True, content=content))
        res = _run_one(prob, ctx, {})
        self.assertTrue(res.passed)

    def test_wrong_tool_fails(self):
        from localbench.suites.tool_calling_suite import _run_one
        prob = self.problems[0]
        ctx = FakeCtx(ChatResult(
            success=True,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "convert_currency", "arguments": '{"amount": 10}'},
            }]
        ))
        res = _run_one(prob, ctx, {})
        self.assertFalse(res.passed)
        self.assertIn("wrong tool", res.error.lower())

    def test_schema_violation_fails(self):
        from localbench.suites.tool_calling_suite import _run_one
        prob = self.problems[0]
        # unit must be 'celsius' or 'fahrenheit', pass 'kelvin'
        ctx = FakeCtx(ChatResult(
            success=True,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": prob["expected_tool"],
                    "arguments": f'{{"location": "Tokyo", "unit": "kelvin", "days": 3}}',
                },
            }]
        ))
        res = _run_one(prob, ctx, {})
        self.assertFalse(res.passed)
        self.assertIn("schema validation failed", res.error.lower())

    def test_negative_control_passes_without_tool(self):
        from localbench.suites.tool_calling_suite import _run_one
        neg_prob = next(p for p in self.problems if p["is_negative"])
        ctx = FakeCtx(ChatResult(success=True, content="The chemical symbol for Gold is Au."))
        res = _run_one(neg_prob, ctx, {})
        self.assertTrue(res.passed)

    def test_negative_control_fails_on_hallucinated_tool(self):
        from localbench.suites.tool_calling_suite import _run_one
        neg_prob = next(p for p in self.problems if p["is_negative"])
        ctx = FakeCtx(ChatResult(
            success=True,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search_database", "arguments": '{"table": "users", "query": "gold"}'},
            }]
        ))
        res = _run_one(neg_prob, ctx, {})
        self.assertFalse(res.passed)
        self.assertIn("negative control failed", res.error.lower())


class TestMultiTurnGrading(unittest.TestCase):
    def test_secret_recall_passes(self):
        from localbench.suites.multi_turn_suite import _evaluate_dialogue
        responses = [
            "Understood, clearance code ALPHA-772 registered.",
            "The largest mammal is the blue whale.",
            "The clearance code is ALPHA-772.",
        ]
        passed, error = _evaluate_dialogue("exact_contains", "ALPHA-772", responses)
        self.assertTrue(passed)
        self.assertIsNone(error)

    def test_secret_recall_fails_when_forgotten(self):
        from localbench.suites.multi_turn_suite import _evaluate_dialogue
        responses = [
            "Understood.",
            "The blue whale.",
            "I'm sorry, I don't remember the clearance code.",
        ]
        passed, error = _evaluate_dialogue("exact_contains", "ALPHA-772", responses)
        self.assertFalse(passed)
        self.assertIn("expected to find", error.lower())

    def test_persistent_prefix_fails_if_turn_violates(self):
        from localbench.suites.multi_turn_suite import _evaluate_dialogue
        responses = [
            "[SECURE_LOG] Understood.",
            "210.",  # Missing prefix!
            "[SECURE_LOG] Canberra.",
        ]
        passed, error = _evaluate_dialogue("starts_with", "[SECURE_LOG]", responses)
        self.assertFalse(passed)
        self.assertIn("turn 2", error.lower())

    def test_forbidden_word_fails_if_used(self):
        from localbench.suites.multi_turn_suite import _evaluate_dialogue
        responses = [
            "I understand the rule.",
            "Use time-blocking.",
            "A yellow banana is curved.",  # Used forbidden word!
        ]
        passed, error = _evaluate_dialogue("forbidden_word", "banana", responses)
        self.assertFalse(passed)
        self.assertIn("forbidden word", error.lower())


class TestFrontierProgressAndData(unittest.TestCase):
    def test_frontier_suite_invokes_on_progress_and_retains_data(self):
        from localbench.suites import frontier_graded_suite
        from localbench.judge.base import JudgeChatResult, JudgeClient

        class FakeJudge(JudgeClient):
            def __init__(self):
                super().__init__("fake-model")

            def chat(self, messages, max_tokens=1024):
                return JudgeChatResult(text="ok", prompt_tokens=10, completion_tokens=10)

            def generate_task(self, category, task_spec):
                return {"prompt": "Write a 3-word poem."}, JudgeChatResult(text="task", prompt_tokens=10, completion_tokens=10)

            def grade(self, task, response, task_spec):
                return {"score": 9, "rationale": "Great poem"}, JudgeChatResult(text="grade", prompt_tokens=10, completion_tokens=10)

        progress_events = []
        ctx = FakeCtx(ChatResult(success=True, content="Sun shines bright.", prompt_tokens=5, completion_tokens=3))
        results = frontier_graded_suite.run(
            ctx,
            FakeJudge(),
            num_tasks=2,
            on_progress=lambda cur, tot, pid, ok, **kw: progress_events.append((cur, tot, pid, ok)),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(len(progress_events), 2)
        self.assertEqual(progress_events[0][0], 1)
        self.assertEqual(progress_events[1][0], 2)
        self.assertTrue(results[0].passed)
        self.assertEqual(results[0].prompt, "Write a 3-word poem.")
        self.assertEqual(results[0].response_content, "Sun shines bright.")


class TestModelSwitching(unittest.TestCase):
    def test_failed_load_command_raises_skippable_error(self):
        # A load command that exits non-zero (model not downloaded on this
        # machine) must raise ModelSwitchError -- which the run loop catches
        # to skip just that model -- not crash with a raw CalledProcessError.
        model_cfg = {
            "name": "not-a-real-model",
            "switch": {"load_cmd": "exit 1"},
        }
        logs = []
        with self.assertRaises(ModelSwitchError) as ctx:
            _switch_to_model(model_cfg, "http://localhost:9", None, logs.append, lambda m: None)
        self.assertIn("not-a-real-model", str(ctx.exception))
        self.assertIn("downloaded", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

