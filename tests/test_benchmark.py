import os
import unittest
from localbench.json_extract import extract_json
from localbench.hardware import get_hardware_snapshot
from localbench.results import ProblemResult, SuiteRunResult
from localbench.engine import _has_repetition
from localbench.gpu_probe import _rocm_smi_json_to_reading, _ioreg_entries_to_reading, _amd_smi_json_to_reading

class TestLocalbenchCore(unittest.TestCase):
    def setUp(self):
        # probe_backend_devices remembers the last successful device list per
        # binary (so a transient failure doesn't drop a working GPU backend).
        # That cache is process-global, so without clearing it a test that
        # mocks a successful probe leaks into a later test that mocks a
        # failure for the same fake path.
        from localbench import llamacpp_mgr
        llamacpp_mgr._LAST_GOOD_PROBE.clear()

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

    def test_switch_to_model_threads_backend_selection_through_to_launch(self):
        from unittest.mock import patch
        from localbench.runner import _switch_to_model

        model_cfg = {
            "name": "GLM-4.6V-Flash-Q4_K_M",
            "runtime_flavor": "llamacpp",
            "backend": "rocm",
            "devices": ["ROCm0"],
        }

        # is_model_already_serving is patched explicitly: unpatched it makes a
        # real HTTP call to localhost:8080, so whether this test passed
        # depended on whether a llama-server happened to be running on the
        # machine -- which made it fail spuriously mid-session.
        with patch("localbench.llamacpp_mgr.launch_llama_server", return_value=(True, "ok")) as mock_launch, \
             patch("localbench.llamacpp_mgr.wait_for_server_ready", return_value=True), \
             patch("localbench.llamacpp_mgr.find_model_gguf", return_value=r"C:\models\glm.gguf"), \
             patch("localbench.llamacpp_mgr.is_model_already_serving", return_value=False):
            _switch_to_model(model_cfg, base_url="http://localhost:8080/v1", unload_all_cmd=None, log=lambda *a: None, confirm=lambda *a: None)

        # Regression: this call used to omit backend= entirely, silently
        # falling back to launch_llama_server's legacy auto-pick (always
        # Vulkan on a machine where it's installed) regardless of which
        # backend the dashboard actually resolved -- which meant a ROCm
        # device id like "ROCm0" got handed to the Vulkan binary, which
        # doesn't recognize it and falls back to using every GPU.
        mock_launch.assert_called_once()
        self.assertEqual(mock_launch.call_args.kwargs.get("backend"), "rocm")

    def test_switch_to_model_reuses_already_loaded_model(self):
        from unittest.mock import patch
        from localbench.runner import _switch_to_model

        model_cfg = {"name": "GLM-4.6V-Flash-Q4_K_M", "runtime_flavor": "llamacpp"}
        logs = []

        # launch_llama_server always stops any running server first, so
        # relaunching a model that is already serving unloads and reloads the
        # identical multi-GB weights for nothing.
        with patch("localbench.llamacpp_mgr.launch_llama_server") as mock_launch, \
             patch("localbench.llamacpp_mgr.find_model_gguf", return_value=r"C:\models\glm.gguf"), \
             patch("localbench.llamacpp_mgr.is_model_already_serving", return_value=True):
            _switch_to_model(model_cfg, base_url="http://localhost:8080/v1", unload_all_cmd=None, log=logs.append, confirm=lambda *a: None)

        mock_launch.assert_not_called()
        self.assertTrue(any("reusing" in line for line in logs), logs)

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

    def test_llamacpp_mgr_build_args(self):
        from localbench.llamacpp_mgr import build_llama_server_args
        cfg = {
            "name": "test-model",
            "context_length": 8192,
            "parallel": 2,
            "gpu_kv": "q8_0",
            "flash_attention": True,
            "mmap": True,
            "batch_size": 2048,
        }
        # build_llama_server_args normalizes the model path to an absolute
        # one. "C:/models/test.gguf" is absolute on Windows but *relative* on
        # POSIX, where abspath() prefixes the cwd -- so a hardcoded Windows
        # path made this assertion fail on Linux while passing on Windows.
        # Use a path that is genuinely absolute on whichever platform runs it.
        model_path = os.path.abspath(os.path.join(os.sep, "models", "test.gguf"))
        args = build_llama_server_args(cfg, model_path, port=8080)
        self.assertIn("-m", args)
        self.assertEqual(os.path.normpath(args[1]), os.path.normpath(model_path))
        self.assertIn("-c", args)
        self.assertEqual(args[args.index("-c") + 1], "8192")
        self.assertIn("-ctk", args)
        self.assertEqual(args[args.index("-ctk") + 1], "q8_0")
        self.assertIn("-b", args)
        self.assertEqual(args[args.index("-b") + 1], "2048")
        self.assertIn("-fa", args)

    def test_llamacpp_mgr_find_model_gguf(self):
        import tempfile
        from localbench.llamacpp_mgr import find_model_gguf
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "Qwen3.5-9B-Q4_K_M.gguf")
            with open(f1, "w") as f:
                f.write("mock")
            matched = find_model_gguf("qwen/qwen3.5-9b", search_dirs=[tmpdir])
            self.assertIsNotNone(matched)
            self.assertEqual(os.path.abspath(f1), matched)

    def test_parse_gguf_metadata(self):
        from localbench.llamacpp_mgr import parse_gguf_metadata
        # Build the path with os.path.join rather than hardcoding a Windows
        # one: parse_gguf_metadata uses os.path.basename, which does not treat
        # "\" as a separator on POSIX, so a literal r"D:\models\...gguf" made
        # display_name come back as the entire path and failed this test on
        # Linux while passing on Windows.
        fake_path = os.path.join("models", "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf")
        meta = parse_gguf_metadata(fake_path)
        self.assertEqual(meta["params"], "24B")
        self.assertEqual(meta["quantization"], "Q4_K_M")
        self.assertEqual(meta["architecture"], "mistral")
        self.assertEqual(meta["display_name"], "Devstral-Small-2-24B-Instruct-2512-Q4_K_M")

    def test_settings_store_model_directories(self):
        from localbench import settings_store
        dirs = settings_store.get_model_directories()
        self.assertIsInstance(dirs, list)
        self.assertTrue(len(dirs) > 0)

    def test_hardware_perf_problems_fixed_shape(self):
        from localbench.data.hardware_perf_problems import generate_problems, NUM_PROBLEMS
        problems = generate_problems(seed=42)
        self.assertEqual(len(problems), NUM_PROBLEMS)
        self.assertEqual(NUM_PROBLEMS, 8)
        ids = [p["id"] for p in problems]
        self.assertEqual(len(ids), len(set(ids)), "problem ids must be unique")
        prefill_ids = [p["id"] for p in problems if p["task_type"] == "prefill"]
        decode_ids = [p["id"] for p in problems if p["task_type"] == "decode"]
        self.assertEqual(len(prefill_ids), 6)
        self.assertEqual(len(decode_ids), 2)

    def test_hardware_perf_tiers_respect_context_window(self):
        from localbench.data.hardware_perf_problems import (
            generate_problems, tiers_for_context, num_problems_for_context,
        )

        # A prefill probe that exceeds the context window is not a slow
        # measurement, it's a failed call -- so oversized tiers must be
        # dropped rather than attempted.
        self.assertEqual([t[0] for t in tiers_for_context(4096)], ["prefill_tiny", "prefill_small"])
        self.assertEqual(len(tiers_for_context(None)), 6)

        # Larger windows unlock strictly more tiers, never fewer.
        counts = [num_problems_for_context(c) for c in (4096, 8192, 16384, 32768, 131072)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 8)

        # Every generated prompt must plausibly fit its window.
        for ctx in (4096, 8192, 32768):
            for p in generate_problems(seed=42, max_context_tokens=ctx):
                if p["task_type"] != "prefill":
                    continue
                self.assertLessEqual(p["est_prompt_tokens"], ctx, f"{p['id']} at ctx={ctx}")

    def test_hardware_perf_problems_deterministic_per_seed(self):
        from localbench.data.hardware_perf_problems import generate_problems
        a = generate_problems(seed=42)
        b = generate_problems(seed=42)
        self.assertEqual([p["prompt"] for p in a], [p["prompt"] for p in b])

    def test_hardware_perf_prefill_prompts_scale_in_length(self):
        # The whole point of the tiering: each successive prefill prompt
        # should be meaningfully longer than the last, not accidentally
        # flat or out of order.
        from localbench.data.hardware_perf_problems import generate_problems
        problems = generate_problems(seed=42)
        prefill_lengths = [len(p["prompt"]) for p in problems if p["task_type"] == "prefill"]
        self.assertEqual(prefill_lengths, sorted(prefill_lengths))
        self.assertLess(prefill_lengths[0], prefill_lengths[-1] / 10)

    def test_hardware_perf_task_count_not_halved_by_quick_profile(self):
        # A real bug this locks in: setting num_problems for display purposes
        # would have made problems_for()'s quick-profile logic halve the
        # reported count (4 instead of 8), contradicting "always the fixed
        # set" -- see webapp/main.py's _suite_profile_count.
        from localbench.data.hardware_perf_problems import NUM_PROBLEMS
        from localbench.profiles import problems_for
        # hardware_perf is deliberately NOT in PROFILE_SIZES and never has
        # num_problems set in config, so the generic path returns None here --
        # webapp/main.py's _suite_profile_count special-cases it to
        # NUM_PROBLEMS instead of calling problems_for() at all.
        self.assertIsNone(problems_for("hardware_perf", "quick", None))
        self.assertEqual(NUM_PROBLEMS, 8)

    def test_hardware_perf_suite_grades_on_completion_not_correctness(self):
        from localbench.suites.hardware_perf_suite import _run_one
        from localbench.engine import ChatResult

        class FakeCtx:
            def call(self, messages, **kwargs):
                return ChatResult(
                    success=True,
                    content="anything at all, content is irrelevant here",
                    finish_reason="length",
                    latency_seconds=1.2,
                    ttft_seconds=0.3,
                    prompt_tokens=200,
                    completion_tokens=8,
                    requested_max_tokens=8,
                )

        problem = {"id": "prefill_tiny", "task_type": "prefill", "prompt": "x", "max_tokens": 8}
        result = _run_one(problem, FakeCtx(), timeout_seconds=30)
        # Truncated (finish_reason=length) elsewhere means "wrong" or
        # "loop_detected" -- here it just means the speed probe completed
        # and produced real timing data, which is all this suite grades on.
        self.assertTrue(result.passed)
        self.assertEqual(result.prompt_tokens, 200)
        self.assertAlmostEqual(result.prefill_tokens_per_sec, 200 / 0.3, places=2)

    def test_classify_backend_dir_recognizes_known_flavors(self):
        from localbench.llamacpp_mgr import _classify_backend_dir

        self.assertEqual(_classify_backend_dir("llama.cpp-win-x86_64-vulkan-avx2-2.31.2"), ("vulkan", "Vulkan"))
        self.assertEqual(_classify_backend_dir("llama.cpp-win-x86_64-amd-rocm-avx2-2.31.2"), ("rocm", "ROCm"))
        self.assertEqual(_classify_backend_dir("llama.cpp-win-x86_64-nvidia-cuda-avx2-2.28.2"), ("cuda", "CUDA"))
        self.assertEqual(_classify_backend_dir("llama.cpp-win-x86_64-avx2-2.31.2"), ("cpu", "CPU"))
        self.assertIsNone(_classify_backend_dir("some-unrelated-folder"))

    def test_extract_version_orders_correctly_for_latest_pick(self):
        from localbench.llamacpp_mgr import _extract_version

        self.assertLess(_extract_version("thing-2.24.0"), _extract_version("thing-2.31.2"))
        self.assertEqual(_extract_version("no-version-here"), (0,))

    def test_probe_backend_devices_parses_multi_gpu_vulkan_output(self):
        from unittest.mock import patch
        from localbench.llamacpp_mgr import probe_backend_devices

        fake_stdout = (
            "Available devices:\n"
            "  Vulkan0: AMD Radeon RX 7800 XT (16368 MiB, 15405 MiB free)\n"
            "  Vulkan1: AMD Radeon RX 6650 XT (8176 MiB, 7378 MiB free)\n"
        )

        class FakeProc:
            returncode = 0
            stdout = fake_stdout
            stderr = ""

        with patch("localbench.llamacpp_mgr.subprocess.run", return_value=FakeProc()):
            result = probe_backend_devices(r"C:\fake\llama-server.exe")

        self.assertTrue(result["available"])
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["devices"]), 2)
        self.assertEqual(result["devices"][0], {"id": "Vulkan0", "name": "AMD Radeon RX 7800 XT", "total_mb": 16368, "free_mb": 15405})

    def test_probe_backend_devices_empty_list_is_still_available_for_cpu_builds(self):
        from unittest.mock import patch
        from localbench.llamacpp_mgr import probe_backend_devices

        class FakeProc:
            returncode = 0
            stdout = "no gpu devices found\n"
            stderr = ""

        with patch("localbench.llamacpp_mgr.subprocess.run", return_value=FakeProc()):
            result = probe_backend_devices(r"C:\fake\llama-server.exe")

        self.assertTrue(result["available"])
        self.assertEqual(result["devices"], [])

    def test_probe_backend_devices_names_missing_dll_failure(self):
        from unittest.mock import patch
        from localbench.llamacpp_mgr import probe_backend_devices

        class FakeProc:
            returncode = 3221225781  # STATUS_DLL_NOT_FOUND, observed live on a ROCm build
            stdout = ""
            stderr = ""

        with patch("localbench.llamacpp_mgr.subprocess.run", return_value=FakeProc()):
            result = probe_backend_devices(r"C:\fake\llama-server.exe")

        self.assertFalse(result["available"])
        self.assertEqual(result["devices"], [])
        self.assertIn("runtime DLL", result["error"])

    def test_probe_backend_devices_survives_timeout_and_missing_binary(self):
        import subprocess
        from unittest.mock import patch
        from localbench.llamacpp_mgr import probe_backend_devices

        with patch("localbench.llamacpp_mgr.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=15)):
            result = probe_backend_devices(r"C:\fake\llama-server.exe", timeout=15)
        self.assertFalse(result["available"])
        self.assertIn("timed out", result["error"])

        with patch("localbench.llamacpp_mgr.subprocess.run", side_effect=OSError("no such file")):
            result = probe_backend_devices(r"C:\does\not\exist.exe")
        self.assertFalse(result["available"])
        self.assertIn("no such file", result["error"])

    def test_resolve_vendor_dirs_reads_manifest_and_resolves_existing_dirs(self):
        import json
        import tempfile
        from localbench.llamacpp_mgr import _resolve_vendor_dirs

        with tempfile.TemporaryDirectory() as tmp:
            backends_root = os.path.join(tmp, "backends")
            backend_dir = os.path.join(backends_root, "llama.cpp-fake-rocm")
            vendor_pkg = os.path.join(backends_root, "vendor", "fake-vendor")
            vendor_bin = os.path.join(vendor_pkg, "bin")
            os.makedirs(backend_dir)
            os.makedirs(vendor_bin)
            with open(os.path.join(backend_dir, "backend-manifest.json"), "w") as f:
                json.dump({"vendor_lib_package_names": ["fake-vendor"]}, f)

            dirs = _resolve_vendor_dirs(backend_dir)
            self.assertIn(vendor_pkg, dirs)
            self.assertIn(vendor_bin, dirs)

    def test_resolve_vendor_dirs_empty_without_manifest_or_vendor_field(self):
        import json
        import tempfile
        from localbench.llamacpp_mgr import _resolve_vendor_dirs

        with tempfile.TemporaryDirectory() as tmp:
            # No manifest file at all.
            self.assertEqual(_resolve_vendor_dirs(tmp), [])

            # Manifest present but declares no vendor package (e.g. CPU build).
            backend_dir = os.path.join(tmp, "cpu-backend")
            os.makedirs(backend_dir)
            with open(os.path.join(backend_dir, "backend-manifest.json"), "w") as f:
                json.dump({"vendor_lib_package_names": []}, f)
            self.assertEqual(_resolve_vendor_dirs(backend_dir), [])

    def test_classify_vendors_detects_amd_and_nvidia_from_device_names(self):
        from localbench.llamacpp_mgr import _classify_vendors

        self.assertEqual(_classify_vendors({"AMD Radeon RX 7800 XT", "AMD Radeon RX 6650 XT"}), {"amd"})
        self.assertEqual(_classify_vendors({"NVIDIA GeForce RTX 4090"}), {"nvidia"})
        self.assertEqual(_classify_vendors({"AMD Radeon RX 7800 XT", "NVIDIA GeForce RTX 4090"}), {"amd", "nvidia"})
        self.assertEqual(_classify_vendors(set()), set())

    def test_list_llama_backends_filters_cuda_on_amd_only_machine(self):
        from unittest.mock import patch
        from localbench.llamacpp_mgr import list_llama_backends_with_status

        fake_backends = [
            {"id": "rocm", "label": "ROCm", "path": "rocm.exe"},
            {"id": "cuda", "label": "CUDA", "path": "cuda.exe"},
            {"id": "vulkan", "label": "Vulkan", "path": "vulkan.exe"},
            {"id": "cpu", "label": "CPU", "path": "cpu.exe"},
        ]

        def fake_probe(path):
            if path == "vulkan.exe":
                return {"available": True, "devices": [{"name": "AMD Radeon RX 7800 XT", "id": "Vulkan0", "total_mb": 1, "free_mb": 1}], "error": None}
            if path == "rocm.exe":
                return {"available": True, "devices": [{"name": "AMD Radeon RX 7800 XT", "id": "ROCm0", "total_mb": 1, "free_mb": 1}], "error": None}
            if path == "cpu.exe":
                return {"available": True, "devices": [], "error": None}
            # cuda: fails to launch at all on this AMD machine, same as reality.
            return {"available": False, "devices": [], "error": "missing a required runtime DLL"}

        with patch("localbench.llamacpp_mgr.discover_llama_backends", return_value=fake_backends), \
             patch("localbench.llamacpp_mgr.probe_backend_devices", side_effect=fake_probe):
            result = list_llama_backends_with_status()

        ids = {b["id"] for b in result}
        self.assertEqual(ids, {"rocm", "vulkan", "cpu"})
        self.assertNotIn("cuda", ids)

    def test_logic_math_forwards_detect_loops(self):
        import inspect
        from localbench.suites import logic_math_suite

        # The config template enables detect_loops for logic_math on the
        # evidence of a real 4096-token repetition loop. A config flag the
        # suite cannot accept would be silently inert -- exactly the kind of
        # setting that looks configured but does nothing.
        self.assertIn("detect_loops", inspect.signature(logic_math_suite.run).parameters)

        src = inspect.getsource(logic_math_suite.run)
        self.assertIn('call_kwargs["detect_loops"]', src)

    def test_hardware_perf_prefill_not_marked_truncated(self):
        from localbench.suites.hardware_perf_suite import _run_one
        from localbench.engine import ChatResult

        class FakeCtx:
            def call(self, messages, **kwargs):
                # A prefill probe always stops at its tiny max_tokens.
                # ChatResult.truncated is derived from finish_reason, so
                # "length" is what makes it truncated.
                return ChatResult(
                    success=True, content="OK", finish_reason="length",
                    latency_seconds=6.4, ttft_seconds=6.3,
                    prompt_tokens=7080, completion_tokens=8,
                    requested_max_tokens=8,
                )

        prefill = _run_one({"id": "prefill_large", "task_type": "prefill", "prompt": "x", "max_tokens": 8}, FakeCtx(), 30)
        # Truncation is the intended outcome here and carries no information,
        # so it must not be recorded (it made every passing probe show a
        # TRUNCATED flag).
        self.assertFalse(prefill.truncated)
        self.assertTrue(prefill.passed)

        decode = _run_one({"id": "decode_long", "task_type": "decode", "prompt": "x", "max_tokens": 4096}, FakeCtx(), 30)
        # Decode probes keep the real value, where hitting the cap is meaningful.
        self.assertTrue(decode.truncated)

    def test_version_is_single_source_of_truth(self):
        import re
        from pathlib import Path
        from localbench import __version__

        # Three places used to carry a version and two of them drifted:
        # pyproject said 0.2.0, localbench/__init__ still said 0.1.0, and the
        # dashboard header showed a hardcoded "PRO v2.5" unrelated to either.
        # That made "which version am I running?" unanswerable, which is
        # exactly the confusion this guards against.
        root = Path(__file__).resolve().parent.parent
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
        self.assertIsNotNone(m, "pyproject.toml must declare a version")
        self.assertEqual(m.group(1), __version__, "pyproject.toml and localbench.__version__ disagree")

        # The header/asset URLs must be templated, not hardcoded, so the page
        # always reports the version actually serving it.
        index_html = (root / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("__APP_VERSION__", index_html)
        self.assertNotIn("PRO v2.5", index_html)

    def test_launch_requires_a_model_name(self):
        from unittest.mock import patch
        from localbench import llamacpp_mgr

        # A request without a model name used to default to "model", which
        # fuzzy-matches a real model.gguf on disk -- so a malformed request
        # silently started loading an arbitrary multi-GB model and reported
        # success. It must refuse instead, and must not reach the launcher.
        with patch.object(llamacpp_mgr, "find_model_gguf") as mock_find:
            for cfg in ({}, {"name": ""}, {"name": "   "}):
                ok, msg = llamacpp_mgr.launch_llama_server(cfg, in_terminal=False)
                self.assertFalse(ok, cfg)
                self.assertIn("no model name", msg.lower())
            mock_find.assert_not_called()

    def test_background_launch_reports_immediate_model_load_failure(self):
        from unittest.mock import patch, MagicMock
        from localbench import llamacpp_mgr

        # Starting the process is not the same as loading the model: a GGUF
        # the backend cannot read makes llama-server exit within a second.
        # Reporting "started successfully" for that showed a success message
        # for a load that had already failed.
        dead = MagicMock()
        dead.poll.return_value = 1  # already exited

        with patch.object(llamacpp_mgr, "find_llama_server_binary", return_value=r"C:\b\llama-server.exe"), \
             patch.object(llamacpp_mgr, "find_model_gguf", return_value=r"C:\models\bad.gguf"), \
             patch.object(llamacpp_mgr, "stop_llama_server", return_value=True), \
             patch.object(llamacpp_mgr, "_env_with_vendor_dirs", return_value={}), \
             patch.object(llamacpp_mgr.subprocess, "Popen", return_value=dead), \
             patch.object(llamacpp_mgr.time, "sleep", return_value=None):
            ok, msg = llamacpp_mgr.launch_llama_server({"name": "bad"}, in_terminal=False)

        self.assertFalse(ok)
        self.assertIn("exited immediately", msg)

        # A process still running after the grace period is a real success.
        alive = MagicMock()
        alive.poll.return_value = None
        with patch.object(llamacpp_mgr, "find_llama_server_binary", return_value=r"C:\b\llama-server.exe"), \
             patch.object(llamacpp_mgr, "find_model_gguf", return_value=r"C:\models\good.gguf"), \
             patch.object(llamacpp_mgr, "stop_llama_server", return_value=True), \
             patch.object(llamacpp_mgr, "_env_with_vendor_dirs", return_value={}), \
             patch.object(llamacpp_mgr.subprocess, "Popen", return_value=alive), \
             patch.object(llamacpp_mgr.time, "sleep", return_value=None):
            ok, msg = llamacpp_mgr.launch_llama_server({"name": "good"}, in_terminal=False)

        self.assertTrue(ok)

    def test_probe_falls_back_to_last_good_devices_on_transient_failure(self):
        import subprocess as sp
        from unittest.mock import patch
        from localbench import llamacpp_mgr as m

        m._LAST_GOOD_PROBE.clear()

        class Ok:
            returncode = 0
            stdout = "  CUDA0: NVIDIA GeForce RTX 5060 Laptop GPU (8188 MiB, 8000 MiB free)\n"
            stderr = ""

        with patch.object(m.subprocess, "run", return_value=Ok()):
            first = m.probe_backend_devices("cuda.exe")
        self.assertTrue(first["available"])
        self.assertEqual(len(first["devices"]), 1)

        # A backend that demonstrably worked must not become unavailable
        # because a later probe timed out (GPU busy after a run) -- that is
        # the "my NVIDIA card got lost after one test" report.
        with patch.object(m.subprocess, "run", side_effect=sp.TimeoutExpired(cmd="x", timeout=40)):
            again = m.probe_backend_devices("cuda.exe")
        self.assertTrue(again["available"])
        self.assertTrue(again.get("stale"))
        self.assertEqual([d["name"] for d in again["devices"]], [d["name"] for d in first["devices"]])

        # But a binary that never worked must stay unavailable -- the cache
        # must not invent a working backend.
        with patch.object(m.subprocess, "run", side_effect=sp.TimeoutExpired(cmd="x", timeout=40)):
            never = m.probe_backend_devices("never-worked.exe")
        self.assertFalse(never["available"])
        m._LAST_GOOD_PROBE.clear()

    def test_list_llama_backends_keeps_all_when_no_devices_enumerated(self):
        from unittest.mock import patch
        from localbench.llamacpp_mgr import list_llama_backends_with_status

        fake_backends = [
            {"id": "cuda", "label": "CUDA", "path": "cuda.exe"},
            {"id": "rocm", "label": "ROCm", "path": "rocm.exe"},
            {"id": "vulkan", "label": "Vulkan", "path": "vulkan.exe"},
            {"id": "cpu", "label": "CPU", "path": "cpu.exe"},
        ]

        # Every probe fails to enumerate -- the ordinary case when the GPU is
        # busy serving a model that was just benchmarked, VRAM is full, or a
        # probe times out. Regression: this was read as "no NVIDIA present"
        # and deleted the CUDA backend outright, so an NVIDIA laptop lost its
        # card from the dashboard after one successful run.
        def probe_empty(path):
            return {"available": False, "devices": [], "error": "timed out after 15.0s"}

        with patch("localbench.llamacpp_mgr.discover_llama_backends", return_value=[dict(b) for b in fake_backends]), \
             patch("localbench.llamacpp_mgr.probe_backend_devices", side_effect=probe_empty):
            ids = {b["id"] for b in list_llama_backends_with_status()}

        self.assertEqual(ids, {"cuda", "rocm", "vulkan", "cpu"})

    def test_list_llama_backends_keeps_backend_that_found_its_own_devices(self):
        from unittest.mock import patch
        from localbench.llamacpp_mgr import list_llama_backends_with_status

        fake_backends = [
            {"id": "cuda", "label": "CUDA", "path": "cuda.exe"},
            {"id": "vulkan", "label": "Vulkan", "path": "vulkan.exe"},
        ]

        # A GPU whose name matches no vendor hint (e.g. Intel Arc) must not
        # cause the backend that successfully enumerated it to be discarded.
        def probe(path):
            if path == "cuda.exe":
                return {"available": True, "devices": [{"name": "Some Unlabelled Accelerator", "id": "d0", "total_mb": 1, "free_mb": 1}], "error": None}
            return {"available": True, "devices": [{"name": "Some Unlabelled Accelerator", "id": "d0", "total_mb": 1, "free_mb": 1}], "error": None}

        with patch("localbench.llamacpp_mgr.discover_llama_backends", return_value=[dict(b) for b in fake_backends]), \
             patch("localbench.llamacpp_mgr.probe_backend_devices", side_effect=probe):
            ids = {b["id"] for b in list_llama_backends_with_status()}

        self.assertIn("cuda", ids)

    def test_list_llama_backends_filters_rocm_on_nvidia_only_machine(self):
        from unittest.mock import patch
        from localbench.llamacpp_mgr import list_llama_backends_with_status

        fake_backends = [
            {"id": "rocm", "label": "ROCm", "path": "rocm.exe"},
            {"id": "cuda", "label": "CUDA", "path": "cuda.exe"},
            {"id": "vulkan", "label": "Vulkan", "path": "vulkan.exe"},
            {"id": "cpu", "label": "CPU", "path": "cpu.exe"},
        ]

        def fake_probe(path):
            if path == "vulkan.exe":
                return {"available": True, "devices": [{"name": "NVIDIA GeForce RTX 5060", "id": "Vulkan0", "total_mb": 1, "free_mb": 1}], "error": None}
            if path == "cuda.exe":
                return {"available": True, "devices": [{"name": "NVIDIA GeForce RTX 5060", "id": "CUDA0", "total_mb": 1, "free_mb": 1}], "error": None}
            if path == "cpu.exe":
                return {"available": True, "devices": [], "error": None}
            # rocm: no AMD GPU present, fails to launch (or simply irrelevant).
            return {"available": False, "devices": [], "error": "missing a required runtime DLL"}

        with patch("localbench.llamacpp_mgr.discover_llama_backends", return_value=fake_backends), \
             patch("localbench.llamacpp_mgr.probe_backend_devices", side_effect=fake_probe):
            result = list_llama_backends_with_status()

        ids = {b["id"] for b in result}
        self.assertEqual(ids, {"cuda", "vulkan", "cpu"})
        self.assertNotIn("rocm", ids)

    def test_env_with_vendor_dirs_prepends_to_path(self):
        from unittest.mock import patch
        from localbench.llamacpp_mgr import _env_with_vendor_dirs

        with patch("localbench.llamacpp_mgr._resolve_vendor_dirs", return_value=[r"C:\vendor\rocm\bin"]):
            env = _env_with_vendor_dirs(r"C:\backends\rocm\llama-server.exe")
        self.assertTrue(env["PATH"].startswith(r"C:\vendor\rocm\bin" + os.pathsep))

        with patch("localbench.llamacpp_mgr._resolve_vendor_dirs", return_value=[]):
            env = _env_with_vendor_dirs(r"C:\backends\vulkan\llama-server.exe")
        self.assertEqual(env["PATH"], os.environ.get("PATH", ""))

    def test_run_gpu_description_never_overclaims_gpus(self):
        from localbench.report import _run_gpu_description

        two_gpu_host = {"gpu": [{"name": "GPU A"}, {"name": "GPU B"}]}

        # 1. Recorded selection wins outright.
        m = {"runtime_load_info": {"gpuSelection": ["GPU A"], "split_mode": "none"}}
        self.assertEqual(_run_gpu_description(m, two_gpu_host), "GPU A")
        m = {"runtime_load_info": {"devices": ["ROCm0"]}}
        self.assertEqual(_run_gpu_description(m, two_gpu_host), "ROCm0")

        # 2. No recorded devices but split_mode "none" means a single GPU was
        # used -- must NOT list both host cards (the bug this guards).
        m = {"runtime_load_info": {"split_mode": "none"}}
        desc = _run_gpu_description(m, two_gpu_host)
        self.assertIn("1 of 2", desc)
        self.assertNotIn("GPU B", desc)

        # 3. Genuinely unknown: host GPUs may be shown, but must be labelled
        # as the host's rather than presented as the run's.
        m = {"runtime_load_info": {"split_mode": "layer"}}
        desc = _run_gpu_description(m, two_gpu_host)
        self.assertIn("GPU A", desc)
        self.assertIn("host GPUs", desc)

        # 4. Single-GPU host needs no hedging.
        self.assertEqual(_run_gpu_description({}, {"gpu": [{"name": "Only GPU"}]}), "Only GPU")

        # 5. No GPU data at all must not raise.
        self.assertEqual(_run_gpu_description({}, {}), "unknown")

    def test_comparison_markdown_excludes_hardware_perf_from_pass_rate(self):
        from localbench.report import render_comparison_markdown

        run = {
            "run_id": "r1",
            "started_at": "2026-08-30T01:00:00",
            "hardware": {"gpu": [{"name": "GPU A"}]},
            "models": {
                "m": {
                    "runtime_load_info": {"runtime_flavor": "llamacpp"},
                    "suites": {
                        "hardware_perf": {
                            "pass_rate": 1.0, "pass_count": 6, "total": 6,
                            "avg_latency_seconds": 1.0, "avg_tokens_per_sec": 50.0, "problems": [],
                        },
                        "coding": {
                            "pass_rate": 0.5, "pass_count": 3, "total": 6,
                            "avg_latency_seconds": 2.0, "avg_tokens_per_sec": 40.0, "problems": [],
                        },
                    },
                }
            },
        }
        md = render_comparison_markdown([run])
        hw_line = next(l for l in md.splitlines() if "hardware_perf" in l)
        coding_line = next(l for l in md.splitlines() if "| coding |" in l)
        # A speed suite must never present a pass rate next to a real one.
        self.assertIn("n/a (speed-only suite)", hw_line)
        self.assertNotIn("100%", hw_line)
        self.assertIn("50%", coding_line)

    def test_verify_api_key_reports_validity_without_leaking_key(self):
        from unittest.mock import patch
        from localbench import settings_store

        class Resp:
            def __init__(self, code, payload=None):
                self.status_code = code
                self._payload = payload or {}
            def json(self):
                return self._payload

        # Valid key.
        with patch.object(settings_store, "_get_key", return_value="sk-secret-value"), \
             patch("requests.get", return_value=Resp(200)):
            r = settings_store.verify_api_key("openrouter")
        self.assertTrue(r["ok"])
        self.assertNotIn("sk-secret-value", str(r))

        # Rejected key surfaces the provider's own wording, not just a code.
        with patch.object(settings_store, "_get_key", return_value="sk-bad"), \
             patch("requests.get", return_value=Resp(401, {"error": {"message": "User not found."}})):
            r = settings_store.verify_api_key("openrouter")
        self.assertFalse(r["ok"])
        self.assertIn("User not found", r["detail"])
        self.assertNotIn("sk-bad", str(r))

        # No key saved is an answer, not an exception.
        with patch.object(settings_store, "_get_key", return_value=None):
            r = settings_store.verify_api_key("anthropic")
        self.assertFalse(r["ok"])

        # Network failure must not raise either.
        import requests as _rq
        with patch.object(settings_store, "_get_key", return_value="sk-x"), \
             patch("requests.get", side_effect=_rq.exceptions.ConnectionError("boom")):
            r = settings_store.verify_api_key("openai")
        self.assertFalse(r["ok"])
        self.assertIn("Could not reach", r["detail"])

        # Unknown provider is a caller error.
        with self.assertRaises(ValueError):
            settings_store.verify_api_key("bogus")

    def test_auxiliary_gguf_filtering(self):
        from localbench.llamacpp_mgr import is_auxiliary_gguf

        # Real files that llama-server refuses as a main model. The suffix
        # cases are the regression: an older startswith("mmproj") check let
        # every one of them into the model list.
        for name in ("Qwen3.5-2B.BF16-mmproj.gguf", "qwen36_mtp.gguf",
                     "mmproj-model-f16.gguf", "llama-3-draft.gguf"):
            self.assertTrue(is_auxiliary_gguf(name), name)

        # Real models must never be hidden -- including ones whose names merely
        # contain those letters mid-word.
        for name in ("model.gguf", "qwen.gguf", "Qwen3.8-27B-Q4_K_M.gguf",
                     "some-mtp-model-7B.gguf", "Draft-Llama-8B.gguf"):
            self.assertFalse(is_auxiliary_gguf(name), name)

        self.assertFalse(is_auxiliary_gguf("notes.txt"))

    def test_is_model_already_serving_matches_only_exact_model(self):
        from unittest.mock import patch
        from localbench import llamacpp_mgr

        class Resp:
            status_code = 200
            def __init__(self, path): self._p = path
            def json(self): return {"data": [{"id": self._p}]}

        target = r"C:\models\foo.gguf"
        with patch("localbench.llamacpp_mgr.requests.get", return_value=Resp(target)):
            self.assertTrue(llamacpp_mgr.is_model_already_serving(target))
        # A different model loaded must force a reload, not be reused.
        with patch("localbench.llamacpp_mgr.requests.get", return_value=Resp(r"C:\models\other.gguf")):
            self.assertFalse(llamacpp_mgr.is_model_already_serving(target))
        # No server at all -> not serving (and must not raise).
        with patch("localbench.llamacpp_mgr.requests.get", side_effect=OSError("refused")):
            self.assertFalse(llamacpp_mgr.is_model_already_serving(target))

    def test_build_llama_server_args_threads_device_selection(self):
        from localbench.llamacpp_mgr import build_llama_server_args

        args = build_llama_server_args({"settings": {"devices": ["Vulkan0"]}}, r"C:\models\x.gguf")
        self.assertIn("-dev", args)
        self.assertEqual(args[args.index("-dev") + 1], "Vulkan0")

        args_multi = build_llama_server_args({"settings": {"devices": ["Vulkan0", "Vulkan1"]}}, r"C:\models\x.gguf")
        self.assertEqual(args_multi[args_multi.index("-dev") + 1], "Vulkan0,Vulkan1")

        args_none = build_llama_server_args({}, r"C:\models\x.gguf")
        self.assertNotIn("-dev", args_none)


if __name__ == "__main__":
    unittest.main()


