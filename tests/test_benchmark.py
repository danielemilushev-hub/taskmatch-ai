import unittest
from localbench.json_extract import extract_json
from localbench.hardware import get_hardware_snapshot
from localbench.results import ProblemResult, SuiteRunResult, ModelRunResult, RunRecord

class TestLocalbenchCore(unittest.TestCase):
    def test_json_extract_clean(self):
        val, err = extract_json('{"key": "value"}')
        self.assertIsNone(err)
        self.assertEqual(val, {"key": "value"})

    def test_json_extract_markdown_block(self):
        val, err = extract_json('```json\n{"score": 10}\n```')
        self.assertIsNone(err)
        self.assertEqual(val, {"score": 10})

    def test_json_extract_embedded(self):
        val, err = extract_json('Here is the json: {"a": [1, 2]} thanks!')
        self.assertIsNone(err)
        self.assertEqual(val, {"a": [1, 2]})

    def test_hardware_snapshot(self):
        snap = get_hardware_snapshot()
        self.assertIn("os", snap)
        self.assertIn("cpu", snap)
        self.assertIn("gpu", snap)

    def test_suite_result_pass_rate(self):
        p1 = ProblemResult(problem_id="1", passed=True, latency_seconds=1.0)
        p2 = ProblemResult(problem_id="2", passed=False, latency_seconds=2.0)
        suite = SuiteRunResult(suite="test_suite", problems=[p1, p2])
        self.assertEqual(suite.pass_count, 1)
        self.assertEqual(suite.total, 2)
        self.assertEqual(suite.pass_rate, 0.5)

if __name__ == "__main__":
    unittest.main()
