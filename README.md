# TaskMatch AI

**Task-driven evaluation for local LLMs.**

A benchmarking tool for locally-running LLMs (LM Studio, Ollama, llama.cpp,
or anything else exposing an OpenAI-compatible `/v1/chat/completions`
endpoint). It exists to answer one question with real, reproducible data:
**which local model should I use for which job** — so you can route
data-sensitive or cost-sensitive work to a free local model instead of a
paid frontier one, backed by numbers instead of a guess.

## What it measures

Six fully deterministic job suites (no LLM-as-judge — every result is
exact-match, schema-validated, or actually executed):

- **json_schema** — prompts the model for structured JSON against a given
  schema, validates with `jsonschema`.
- **coding** — HumanEval-style problems; the model's code is *actually
  executed* in a sandboxed subprocess against real test cases, not compared
  as text.
- **logic_math** — synthetically generated arithmetic/logic problems (we
  generate them, so the answer is always known), graded by exact match.
- **instruction_following** — IFEval-style formatting constraints (exact
  paragraph count, forbidden letters, exact ending phrase, word limits,
  etc.) graded by mechanical rule-checking, never an LLM judge.
- **pattern_reasoning** — ARC-AGI-style abstract grid-transformation
  puzzles, synthetically generated, graded by exact grid match.
- **long_context** — a large (1000+ line) source excerpt with either a
  planted "needle" value to retrieve or a mechanically-injected bug to
  locate by line number; can use a real file on your machine or a bundled
  synthetic fallback.

Every run also captures: total latency, time-to-first-token (TTFT, measured
via streaming — separates prefill/"thinking" time from raw decode speed),
tokens/sec, a hardware snapshot (CPU/RAM/GPU — best-effort, vendor-generic),
and peak CPU/RAM/VRAM usage sampled during generation relative to a
pre-run baseline (a useful signal for whether a model spilled out of VRAM
into CPU/RAM — see Known Limitations for the VRAM caveat). A failed problem
is also tagged as `truncated` when it hit the token budget mid-thought,
distinct from actually answering wrong.

A seventh, opt-in **frontier_graded** suite exists for open-ended tasks that
can't be graded deterministically (see below) — it's never mixed into the
six deterministic suites' scoring.

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml     # Windows: copy config.example.yaml config.yaml
```

`config.yaml` is **gitignored on purpose** — it holds your own runtime URL,
model list, judge choice and (optionally) a real path from your disk, none of
which belongs in a shared repo. Only `config.example.yaml` is tracked, so
your working config can never be pushed by accident.

Edit `config.yaml`:

```yaml
runtime:
  base_url: "http://localhost:1234/v1"   # your LM Studio / Ollama / llama.cpp endpoint

models:
  - name: "your-model-name"
    switch:
      load_cmd: 'lms load "{model}" -y --gpu max'   # LM Studio's CLI; omit for manual switching
      unload_cmd: "lms unload --all"
```

Or skip hand-editing the file entirely — the dashboard's **Settings** tab
can change the runtime URL, and its **New Run** tab can detect models
directly from your runtime's `/v1/models` endpoint.

Models are benchmarked **strictly sequentially** — one fully loaded and
tested before the next is loaded — because these are large models on
limited VRAM. If a model has no `switch.load_cmd`, the tool pauses and
waits for you to load it manually (works for any runtime, not just LM
Studio).

## Running

```bash
python cli.py run --job all                 # every suite, every model in config.yaml
python cli.py run --job coding               # just one suite
python cli.py run --job all --model NAME     # just one model from config.yaml
```

Each run writes:
- `results/runs/<timestamp>.json` — the full, self-contained result (share
  this file directly to compare with someone else).
- `results/<timestamp>.md` — a markdown leaderboard rendered from it.

### Web dashboard

```bash
python cli.py serve   # http://localhost:8000 by default; --port to change
```

FastAPI backend + a plain HTML/CSS/JS frontend, no build step. Five tabs:

- **New Run** — pick models (including live-detected ones straight from the
  runtime, complete with real size/quantization/context-length/vision/
  tool-use badges pulled from LM Studio's own catalog when available) and
  suites, start a run, and watch:
  - a live-scrolling log with per-problem `[i/N] problem_id: PASS/FAIL`
    lines as each one finishes, not just a static "running suite..." line
  - a **live hardware monitor** — CPU%, RAM (used/total), GPU utilization
    %, GPU memory, and disk read/write throughput, sampled roughly every
    second for the duration of the run (see Known Limitations for exactly
    how GPU data is obtained and its refresh cadence)
  - the manual-switch confirm banner, if a model has no `load_cmd`
  - an optional **Frontier Judge** panel (see below) with a live
    provider/model picker, task/call-count breakdown, and a duration/cost
    estimate drawn from your own prior runs with that judge
- **History** — every saved run, filterable by suite/model, with delete,
  and one-click Markdown/PDF/raw-JSON downloads. Suite pass-rate pills show
  a truncation-count badge when some failures hit the token budget rather
  than answering incorrectly.
- **Compare** — pick any combination of past runs/models and get: a
  Pareto-frontier scatter (accuracy vs. speed), a per-suite radar chart, bar
  charts (pass rate, tokens/sec, TTFT), a full sortable/filterable table
  with a click-through problem inspector (prompt/response/reasoning/error,
  and judge score+rationale for the frontier suite), and Markdown/CSV
  export. A separate **Frontier Judge** section appears when any compared
  run used it, kept apart from the deterministic charts above.
- **Playground** — run an arbitrary one-off prompt (optionally with a JSON
  schema to validate against) against any detected model, outside of any
  saved benchmark run.
- **Settings** — runtime base URL (with LM Studio/Ollama/llama.cpp presets
  and a live connection test), frontier judge provider/model/task-count/
  pass-threshold, and a per-provider API key manager (stored in a local,
  gitignored `.env`; keys are never echoed back once saved).

## Frontier judge (opt-in, paid)

For tasks that genuinely can't be graded deterministically (instruction-
following quality, summarization faithfulness, creative writing, etc.), a
frontier model (Anthropic/OpenAI/Gemini/OpenRouter) can generate tasks and
grade the local model's responses per the rubric in
[TASK_SPEC.md](TASK_SPEC.md) — including a dedicated anti-hallucination
rule: the judge is never asked to recall real-world facts from memory as
ground truth, only to evaluate against self-contained or
harness-verifiable content. This never runs by accident, from either
interface:

**CLI:**
1. Copy `.env.example` to `.env` and fill in the key for your chosen
   provider only.
2. Set `judge.enabled: true` in `config.yaml` and pick a `provider`/`model`.
3. Run `python cli.py run --job frontier_graded` — this prints exactly how
   many paid API calls it will make and requires a `y` confirmation before
   spending anything.

**Dashboard:** enable the judge and add a key in Settings, then check
"Include frontier-graded suite" on the New Run tab — a provider/model
picker (with live model lists fetched from each provider's own API, or
OpenRouter's public catalog) and an explicit paid-cost confirmation dialog
appear before it runs. Duration and cost estimates are shown when available
from a previous run with that exact judge:
- **Duration** is estimated from real wall-clock time observed last time
  (local call + both paid judge calls), for any provider.
- **Cost** is a real dollar estimate only for **OpenRouter** — it publishes
  live, public, per-token pricing, which is combined with real token usage
  from your last run. Anthropic/OpenAI/Gemini don't expose pricing via a
  stable public API, so their estimate is deliberately left as "check your
  provider's pricing page" rather than a hardcoded number that would go
  stale.

Reports always show this suite in a visually distinct "Frontier-Graded
(non-deterministic, paid)" section, never mixed into the six suites'
pass-rate numbers.

Only install the SDK for the provider you're actually using (see
`requirements.txt`) — each is imported lazily, so the rest of the app
works with none of them installed. `openrouter` reuses the `openai`
package pointed at OpenRouter's endpoint, so it needs no separate SDK.

## Config notes

- **Per-suite `max_tokens`/`call_timeout_seconds` overrides**: reasoning
  models can spend a large, variable number of hidden "thinking" tokens
  before writing an answer. If the budget/timeout is too low, a response
  gets cut off mid-thought — the harness reports this distinctly as
  `truncated`, never silently as "wrong answer." The shipped defaults were
  tuned against a live 12B reasoning model that needed roughly 2-3x the
  naive default budget on the harder suites.
- **`long_context.source_file`**: optional path to a real file on your
  machine for a more realistic excerpt than the bundled synthetic
  generator. Left `null` by default — if you set a real path, that path
  will be visible to anyone you share `config.yaml` with (including a git
  remote), so consider keeping that particular edit local/uncommitted.

## Known limitations

- **Sandbox isolation**: the `coding` suite runs generated code in its own
  subprocess with a hard timeout, in a throwaway temp directory — not full
  OS-level sandboxing (no seccomp/cgroups/chroot). That's the practical
  ceiling on Windows without extra tooling; don't run this against
  untrusted models you don't want executing arbitrary code on your machine.
- **VRAM is tracked two different ways, with two different levels of
  vendor support:**
  - The **per-run saved delta** (shown in History/Compare) samples via
    `nvidia-smi` — NVIDIA only. On other GPUs it's honestly reported as
    unavailable, never guessed.
  - The **live hardware monitor** (during a run, New Run tab) instead
    queries Windows' own GPU performance counters (the same ones backing
    Task Manager's GPU tab), which work for *any* vendor including
    AMD/Intel — but each query takes a few seconds, so GPU samples there
    refresh on a slower cadence than CPU/RAM/disk, and neither this nor
    the live monitor's readings are persisted into the saved run.
  - Static hardware-snapshot VRAM *size* (not usage) falls back to a
    Windows registry read when `nvidia-smi` isn't present, which does work
    cross-vendor.
  - None of the supported runtimes expose an exact GPU-vs-CPU offload
    ratio via their API/CLI regardless of source — if tokens/sec looks
    much lower than expected for a model's size, check the runtime's own
    UI (e.g. LM Studio's Developer tab) to confirm.
- **`lms load` can stack duplicate model instances** instead of reusing an
  already-loaded one — `runtime.unload_all_cmd` exists specifically to
  guarantee a clean slate before every switch.
- **Windows-first**: the live hardware monitor and the registry-based VRAM
  size fallback are Windows-specific (`platform.system() == "Windows"`
  gated); CPU/RAM figures use `psutil` and work anywhere, but GPU data
  won't be available on Linux/macOS yet.

## Project layout

```
config.yaml               # your runtime/model/suite/judge configuration
.env.example               # documents the 4 judge API key names (.env is gitignored)
TASK_SPEC.md                # the frontier judge's task-generation/grading rubric
cli.py                       # entry point (run / serve)
localbench/                # Python package name kept as-is (see note below)
  engine.py                   # runtime-agnostic streaming chat_completion() caller
  runner.py                   # sequential model-switch orchestration + per-problem progress
  report.py                   # markdown/HTML/PDF report rendering
  storage.py                  # save/list/load/delete run records
  settings_store.py           # dashboard Settings: config.yaml edits + .env key management
  live_monitor.py             # continuous CPU/RAM/GPU/disk sampling during a run (real-time only)
  hardware.py                 # one-time CPU/RAM/GPU snapshot per run
  resource_monitor.py         # per-suite baseline->peak CPU/RAM/VRAM delta (saved to the report)
  results.py                  # ProblemResult / SuiteRunResult / RunRecord
  suites/                      # one module per job suite
  data/                        # problem generators/data for each suite
  judge/                        # frontier judge provider clients + factory
webapp/
  main.py                       # FastAPI backend for the dashboard
  run_manager.py                 # background run threads + live status/progress
  static/                         # HTML/CSS/JS frontend, no build step
results/runs/                     # one JSON file per benchmark run (gitignored)
```

## Documentation

| Document | Contents |
|---|---|
| [SECURITY.md](SECURITY.md) | What data leaves your machine (and what doesn't), how API keys are stored, and the code-execution risk in the `coding` suite. **Read before benchmarking a model you don't trust.** |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, tests, how to add a job suite, and the project's core rule on never displaying an unmeasured number. |
| [TASK_SPEC.md](TASK_SPEC.md) | The rubric the frontier judge is given, including the anti-hallucination contract. |

## License

[Apache License 2.0](LICENSE).
