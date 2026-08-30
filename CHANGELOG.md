# Changelog

All notable changes to TaskMatch AI are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [semantic versioning](https://semver.org/).

## [0.2.1] — 2026-08-30

Fixes two problems reported from a fresh install on a second machine.

### Fixed

- **A GPU could disappear from the dashboard after a successful run.** When
  no backend probe managed to enumerate any device — which happens for
  ordinary, temporary reasons such as the GPU still serving the model just
  benchmarked, VRAM being full, or a probe timing out — that silence was read
  as "this vendor's hardware is not present" and the vendor-specific backend
  (CUDA, ROCm) was removed from the list entirely, with no way to get it back
  short of a restart. Filtering now requires positive evidence of the GPU
  inventory: if nothing was enumerated, nothing is filtered. A backend that
  enumerated its own devices is also always kept, even if the device name
  matches no known vendor pattern (e.g. Intel Arc).
- **No way to tell which version was running.** The header showed a hardcoded
  "PRO v2.5" unrelated to the real version, and `localbench/__init__.py` had
  drifted to 0.1.0 while `pyproject.toml` said 0.2.0. There is now a single
  `__version__`, shown in the dashboard header, returned by `/api/version`,
  and printed by `python cli.py --version`; a test fails if the two files
  disagree.

- **The dashboard always claimed a model was loaded in LM Studio.** Load
  state was inferred from `/v1/models`, which lists every *downloaded* model
  rather than the loaded one, so the first entry was reported as active
  whenever LM Studio was merely running. That produced a permanent "another
  model is active — will auto-switch" warning and had users unloading by hand
  before every run. Load state now comes from LM Studio's own
  `/api/v0/models`, which carries a real per-model `state` field.
- **Prefill probes were flagged TRUNCATED.** `hardware_perf` prefill probes
  set `max_tokens` to 8 on purpose so that timing is dominated by prompt
  processing, which means they always stop at the limit. Recording and
  displaying that as truncation made every passing probe look broken. Decode
  probes, where hitting the cap is meaningful, are unchanged.
- **The model library looked like several models were already selected.**
  Models listed in `config.yaml` were pre-selected on load and rendered
  highlighted, implying a multi-select that no longer exists — a run targets
  one model, chosen via Configure & Run.

### Changed

- `logic_math` now runs with loop detection enabled, and the runner actually
  forwards that setting (it previously could not be passed to that suite at
  all). Evidence: a reasoning model asked "What is 47 + 11 - -14?" spent its
  full 4096-token budget emitting zero answer characters, its reasoning trace
  repeating one line verbatim — the same non-convergent pattern the other
  suites already abort on.

### Added

- Static assets are served as `app.js?v=<version>` / `styles.css?v=<version>`,
  so an upgraded install cannot keep executing a previous release's cached
  JavaScript.
- README documents updating an existing checkout (`git pull`), how to check
  the running version, and that a wrong version means a second copy of the
  repo rather than a caching problem.

## [0.2.0] — 2026-08-30

A large release focused on **measuring the right thing and saying so
honestly**: real compute-backend and GPU selection, a hardware speed suite
that finds your hardware's ceiling, and a series of fixes to places where the
dashboard reported numbers it had not actually measured.

### Added

- **Compute backend selection (Vulkan / ROCm / CUDA / CPU).** Backends are
  discovered from the installed llama.cpp builds and then *probed for real* —
  each candidate binary is launched with `--list-devices` and only offered if
  it genuinely runs on this machine. A backend whose vendor isn't present is
  hidden entirely (CUDA on an all-AMD box, ROCm on an NVIDIA one); Vulkan is
  always offered since it is vendor-neutral.
- **Per-GPU device selection.** Detected GPUs are listed with capacity,
  backend and device id, each with its own on/off switch, plus a multi-GPU
  strategy picker (split evenly / row split / main GPU only). Selection is
  applied via llama.cpp's `-dev` flag. Hardware is chosen *before* the
  backend, and the backend list reacts to it — ROCm is offered only for a
  single GPU, because it cannot combine mixed GPU generations.
- **`hardware_perf` suite** — raw throughput rather than correctness: prefill
  (prompt-processing) speed across escalating context tiers, plus sustained
  decode speed at two generation lengths. Runs first, so every other suite's
  numbers can be read against a known speed baseline.
- **Hardware Bench tab** — a dedicated speed comparison across runs, with a
  sortable table and a bar chart that can rank models by any prefill tier,
  either decode length, or time-to-first-token. Deliberately separate from
  the accuracy-focused Model Arena.
- **Per-run suite selection** in the model configuration modal, with All /
  None shortcuts, so a run can target just the suites you care about.
- **API key testing.** Each frontier provider key can be validated against
  the provider directly (a free, read-only call). Previously a revoked key
  still displayed as "Set", and the only way to discover it was a paid run
  failing on every task.
- **`start.sh`** for macOS/Linux, matching the existing Windows `start.bat`.
- **Deep links** — `?view=hardware` (etc.) opens a specific tab directly.
- Confidence intervals, failure-state breakdowns (`truncated` /
  `loop_detected` / `early_exit`), prefill throughput, and the
  backend/GPU actually used are now included in Markdown/HTML/PDF reports.

### Fixed

- **Compute backend was ignored by actual benchmark runs.** The runner
  launched llama-server without passing the selected backend, silently
  falling back to an auto-pick. Because `-dev` device ids are
  backend-specific, a ROCm device id handed to a Vulkan binary matched
  nothing and the run quietly used *every* GPU — so single-GPU selections
  were not honoured.
- **Frontier judge could never run.** The state the run flow checks
  (`judgeSettings`) was populated inside a function that returned early
  because the UI element it rendered had been removed, so the judge was
  unreachable regardless of settings.
- **Model-fit VRAM was hardcoded** to the development machine's GPUs (15.98 /
  23.96 GB) in four places, including the filter pill labels. It is now
  derived from detected hardware, and hidden entirely when VRAM can't be
  read.
- **Runs claimed GPUs they never used.** Views fell back to listing every GPU
  *installed on the host* when a run's actual device selection wasn't
  recorded. Attribution now distinguishes recorded, inferred, and unknown,
  and never presents host inventory as the run's hardware.
- **Comparisons mixed incompatible suite sets.** "vs baseline" averaged the
  baseline over all suites (including `hardware_perf`, which passes ~100% by
  construction) while averaging the compared model over accuracy suites only.
  Comparisons now use the accuracy suites present in *both* runs, and show
  nothing when there is no overlap.
- **Hardware Bench kept only the latest run per model**, destroying exactly
  the comparison the view exists for (same model, different backend or GPU
  count). All runs are kept, newest first.
- **Companion GGUF files were listed as loadable models.** `mmproj`
  projectors and multi-token-prediction draft weights were offered and failed
  to load; the previous filter only matched them as a filename *prefix*.
- **Loading a model twice.** Pre-loading a model and then starting a
  benchmark unloaded and reloaded the identical weights, because the launcher
  always stops any running server first. An already-serving model is now
  reused.
- **Live speed readout was mislabeled.** During a prefill probe the HUD showed
  decode rate — a structurally tiny number (8 tokens over a latency dominated
  by a huge prompt) that badly understated the hardware. Prefill and decode
  are now shown side by side, each keeping its own last real reading.
- Cross-platform portability: Windows-only drive paths no longer resolve to
  junk directories on Linux/macOS, drive scanning works on POSIX mount
  points, a developer's home directory is no longer a hardcoded search path,
  and `uvicorn webapp.main:app` no longer crashes on a fresh clone.
- The two `config.example.yaml` copies had drifted, breaking CI, and the
  template referenced the developer's own model files.

### Changed

- Runtime engine (llama.cpp / LM Studio / Ollama / vLLM) is now a single
  global setting rather than a per-model dropdown, matching the reality that
  only one local server runs at a time.
- `hardware_perf` prefill tiers extended to ~16k and ~32k tokens. Tiers that
  would exceed a model's configured context window are skipped rather than
  attempted, so the suite adapts to small-context setups.
- A finished or cancelled run now offers "Back to Model Library" and
  "View Results" instead of leaving a dead "Stop Run" button.
- Percentages are shown to one decimal place, so a one-problem difference on
  a small suite is visible rather than rounded away.

### Known limitations

- The ROCm and macOS GPU *telemetry* probes are still unverified against real
  hardware of that kind; they degrade to "unavailable" rather than guessing.
  (ROCm *inference* is verified working — see above.)
- `hardware_perf` grades completion, not correctness, and is deliberately
  excluded from every accuracy aggregate.

## [0.1.0]

Initial release: deterministic local-LLM benchmarking across eight accuracy
suites, a live hardware monitor, run history, Compare Studio, and an opt-in
frontier-model judge.
