import unittest
from localbench.json_extract import extract_json
from localbench.hardware import get_hardware_snapshot
from localbench.results import ProblemResult, SuiteRunResult, ModelRunResult, RunRecord
from localbench.engine import _has_repetition

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

    def test_has_repetition_detects_repeated_phrase(self):
        # A live gemma-4-12b-qat transcript showed this exact pattern: the
        # same block of text (there, re-deriving an example) re-quoted
        # verbatim on every "wait, let me re-check" pass.
        phrase = "Wait, let me re-examine Example 2 again carefully. "
        text = "Some opening reasoning. " + phrase * 3
        self.assertTrue(_has_repetition(text, window_chars=4000, phrase_len=40, min_repeats=3))

    def test_has_repetition_ignores_normal_prose(self):
        text = (
            "The rule appears to be a horizontal flip of each row. "
            "Applying that to the new input grid gives the following result, "
            "which matches the pattern shown in both worked examples above."
        )
        self.assertFalse(_has_repetition(text, window_chars=4000, phrase_len=40, min_repeats=3))

    def test_has_repetition_below_threshold_is_not_a_loop(self):
        # Two repeats of a phrase is a normal restatement, not a loop -- only
        # three-plus is treated as the model failing to converge.
        phrase = "Let's look at the columns of the transformation again. "
        text = phrase * 2
        self.assertFalse(_has_repetition(text, window_chars=4000, phrase_len=40, min_repeats=3))

    def test_has_repetition_short_text_is_never_a_loop(self):
        self.assertFalse(_has_repetition("too short to loop", window_chars=4000, phrase_len=40, min_repeats=3))

if __name__ == "__main__":
    unittest.main()
