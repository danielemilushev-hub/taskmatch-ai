import unittest
from localbench.json_extract import extract_json
from localbench.hardware import get_hardware_snapshot
from localbench.results import ProblemResult, SuiteRunResult
from localbench.engine import _has_repetition
from localbench.gpu_probe import _rocm_smi_json_to_reading, _ioreg_entries_to_reading, _amd_smi_json_to_reading

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

    def test_has_repetition_short_dense_phrase_wrapped_in_varying_context(self):
        # A live bug_locator transcript repeated the exact same ~21-char
        # condition ("if a >= 0 and b >= 0:") 14+ times, but each occurrence
        # was wrapped in a line number that changed every time -- no 40+
        # char exact span ever recurred, so the long-phrase check alone
        # missed this. A shorter phrase length with a higher repeat count
        # catches it instead.
        lines = "\n".join(
            f"*   Line {666 + i * 4}: `if a >= 0 and b >= 0:`" for i in range(8)
        )
        self.assertTrue(_has_repetition(lines, window_chars=1200, phrase_len=18, min_repeats=6))

    def test_rocm_smi_json_to_reading_parses_documented_keys(self):
        # The rocm-smi label format most commonly cited in ROCm docs/guides --
        # NOT independently verified against a real ROCm install (no ROCm
        # hardware available). This locks in the parser's own behavior given
        # that shape, not that the shape itself is correct.
        data = {"card0": {"GPU use (%)": "42", "VRAM Total Used Memory (B)": str(4 * 1024**3)}}
        reading = _rocm_smi_json_to_reading(data)
        self.assertIsNotNone(reading)
        self.assertEqual(reading["util_percent"], 42.0)
        self.assertAlmostEqual(reading["used_mb"], 4 * 1024, places=0)

    def test_rocm_smi_json_to_reading_returns_none_on_unrecognized_shape(self):
        # A future ROCm version using different labels should degrade to
        # "unavailable", never a crash or a fabricated number.
        data = {"card0": {"Some Other Field": "1"}}
        self.assertIsNone(_rocm_smi_json_to_reading(data))

    def test_ioreg_entries_to_reading_parses_documented_shape(self):
        entries = [{"PerformanceStatistics": {"Device Utilization %": 55, "In use system memory": 4 * 1024**3}}]
        reading = _ioreg_entries_to_reading(entries)
        self.assertIsNotNone(reading)
        self.assertEqual(reading["util_percent"], 55.0)
        self.assertAlmostEqual(reading["used_mb"], 4 * 1024, places=0)

    def test_ioreg_entries_to_reading_survives_malformed_shapes(self):
        # A real bug this locks in: entries as a bare dict instead of a list
        # (or an entry that isn't a dict) previously raised AttributeError
        # uncaught, all the way up through query_gpu() into whichever
        # background thread called it -- resource_monitor.py's had zero
        # exception handling around this call at all.
        self.assertIsNone(_ioreg_entries_to_reading({"not": "a list"}))
        self.assertIsNone(_ioreg_entries_to_reading(["not a dict", 42]))
        self.assertIsNone(_ioreg_entries_to_reading([{"PerformanceStatistics": "not a dict"}]))
        self.assertIsNone(_ioreg_entries_to_reading([{"PerformanceStatistics": {"Device Utilization %": "not-a-number"}}]))
        self.assertIsNone(_ioreg_entries_to_reading(None))
        self.assertIsNone(_ioreg_entries_to_reading([]))

    def test_amd_smi_json_to_reading_parses_multi_gpu_list(self):
        parsed = [
            {"usage": {"gfx_activity": 30}, "vram_usage": {"vram_used": 2 * 1024**3}},
            {"usage": {"gfx_activity": 70}, "vram_usage": {"vram_used": 3 * 1024**3}},
        ]
        reading = _amd_smi_json_to_reading(parsed)
        self.assertIsNotNone(reading)
        self.assertEqual(reading["util_percent"], 70.0)  # busiest GPU
        self.assertAlmostEqual(reading["used_mb"], 5 * 1024, places=0)  # summed

    def test_amd_smi_json_to_reading_bad_entry_does_not_discard_good_ones(self):
        # A real bug this locks in: a non-numeric stat on entry 2 previously
        # raised inside the loop and discarded entry 1's already-valid data,
        # rather than just skipping the one bad entry.
        parsed = [
            {"usage": {"gfx_activity": 40}, "vram_usage": {"vram_used": 4 * 1024**3}},
            {"usage": {"gfx_activity": "garbage"}, "vram_usage": "not a dict"},
            "not even a dict",
        ]
        reading = _amd_smi_json_to_reading(parsed)
        self.assertIsNotNone(reading)
        self.assertEqual(reading["util_percent"], 40.0)
        self.assertAlmostEqual(reading["used_mb"], 4 * 1024, places=0)

    def test_amd_smi_json_to_reading_returns_none_when_nothing_usable(self):
        self.assertIsNone(_amd_smi_json_to_reading([{"usage": {}, "vram_usage": {}}]))
        self.assertIsNone(_amd_smi_json_to_reading(["not a dict"]))

    def test_has_repetition_short_threshold_ignores_normal_prose(self):
        # The higher repeat count (6, vs 3-4 for the long-phrase check) is
        # what keeps this from false-positiving on ordinary text: normal
        # prose reuses short generic fragments by chance sometimes, but not
        # six-plus times verbatim in a narrow window.
        text = (
            "The function should return the correct value. Let's check the "
            "output. The output looks correct here. Let's check the next "
            "case. The result is what we expect for this input."
        )
        self.assertFalse(_has_repetition(text, window_chars=1200, phrase_len=18, min_repeats=6))

    def test_live_hardware_monitor_null_disk_io_survives(self):
        from unittest.mock import patch
        from localbench.live_monitor import LiveHardwareMonitor

        mon = LiveHardwareMonitor(interval=0.1)
        with patch("psutil.disk_io_counters", return_value=None):
            sample = mon._collect()
            self.assertIn("cpu_percent", sample)
            self.assertEqual(sample["disk_read_mb_s"], 0.0)
            self.assertEqual(sample["disk_write_mb_s"], 0.0)

    def test_profiles_include_all_deterministic_suites(self):
        from localbench.profiles import PROFILE_SIZES, problems_for
        self.assertIn("tool_calling", PROFILE_SIZES)
        self.assertIn("multi_turn", PROFILE_SIZES)
        self.assertEqual(problems_for("tool_calling", "quick"), 8)
        self.assertEqual(problems_for("tool_calling", "full"), 16)
        self.assertEqual(problems_for("multi_turn", "quick"), 6)
        self.assertEqual(problems_for("multi_turn", "full"), 12)

    def test_tool_calling_problem_generator(self):
        from localbench.data.tool_calling_problems import generate_tool_calling_problems
        problems = generate_tool_calling_problems(num_problems=10, seed=42)
        self.assertEqual(len(problems), 10)
        self.assertTrue(any(p["is_negative"] for p in problems))
        self.assertTrue(any(not p["is_negative"] for p in problems))

    def test_multi_turn_problem_generator(self):
        from localbench.data.multi_turn_problems import generate_multi_turn_problems
        problems = generate_multi_turn_problems(num_problems=10, seed=42)
        self.assertEqual(len(problems), 10)
        for p in problems:
            self.assertGreaterEqual(len(p["turns"]), 3)

    def test_build_runtime_load_cmd_lmstudio(self):
        from localbench.runner import _build_runtime_load_cmd
        model_cfg = {
            "name": "ministral-3-14b",
            "runtime_flavor": "lmstudio",
            "context_length": 8192,
            "gpu_offload": "max",
            "speculative_draft_mtp": True,
        }
        cmd = _build_runtime_load_cmd(model_cfg)
        self.assertIn("lms load \"ministral-3-14b\"", cmd)
        self.assertIn("-c 8192", cmd)
        self.assertIn("--gpu max", cmd)
        self.assertIn("--speculative-draft-mtp", cmd)

    def test_build_runtime_load_cmd_llamacpp(self):
        from localbench.runner import _build_runtime_load_cmd
        model_cfg = {
            "name": "qwen2.5-7b.gguf",
            "runtime_flavor": "llamacpp",
            "context_length": 16384,
            "gpu_kv": "q4_0",
            "flash_attention": True,
            "split_mode": "row",
            "batch_size": 2048,
        }
        cmd = _build_runtime_load_cmd(model_cfg)
        self.assertIn("llama-server -m \"qwen2.5-7b.gguf\"", cmd)
        self.assertIn("-c 16384", cmd)
        self.assertIn("-ctk q4_0 -ctv q4_0", cmd)
        self.assertIn("-fa", cmd)
        self.assertIn("-sm row", cmd)
        self.assertIn("-b 2048", cmd)

    def test_build_runtime_load_cmd_vllm(self):
        from localbench.runner import _build_runtime_load_cmd
        model_cfg = {
            "name": "meta-llama/Llama-3-8B",
            "runtime_flavor": "vllm",
            "context_length": 32768,
            "gpu_kv": "fp8",
            "gpu_offload": 0.90,
        }
        cmd = _build_runtime_load_cmd(model_cfg)
        self.assertIn("vllm serve \"meta-llama/Llama-3-8B\"", cmd)
        self.assertIn("--max-model-len 32768", cmd)
        self.assertIn("--kv-cache-dtype fp8", cmd)
        self.assertIn("--gpu-memory-utilization 0.9", cmd)


if __name__ == "__main__":
    unittest.main()

