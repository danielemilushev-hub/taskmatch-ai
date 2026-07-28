"""Fast pipeline smoke test: fake model responses through real grading.

No live model needed. A FakeCtx stands in for engine.RunContext and returns
canned ChatResults, so these tests exercise the actual grading logic
(_run_one, _execute_candidate) end to end in under a couple of seconds.
"""

import unittest

from localbench.engine import ChatResult
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


if __name__ == "__main__":
    unittest.main()
