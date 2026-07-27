# localbench

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
and peak CPU/RAM usage sampled during generation (a useful signal for
whether a model spilled out of VRAM into CPU/RAM — see Known Limitations).

A seventh, opt-in **frontier_graded** suite exists for open-ended tasks that
can't be graded deterministically (see below) — it's never mixed into the
six deterministic suites' scoring.

## Setup

```bash
pip install -r requirements.txt
```

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

`python cli.py serve` launches a local web dashboard at `http://localhost:8000`
(FastAPI backend + a plain HTML/CSS/JS frontend, no build step) with three
views:
- **New Run** — pick models (including live-detected ones straight from the
  runtime, not just what's hand-typed into `config.yaml`) and suites, start
  a run, watch live progress, and confirm manual model switches from the
  browser instead of a terminal prompt.
- **History** — every saved run, filterable by suite/model, with one-click
  Markdown/PDF/raw-JSON downloads.
- **Playground** — run an arbitrary one-off prompt (optionally with a JSON
  schema to validate against) against any detected model, outside of any
  saved benchmark run.

## Frontier judge (opt-in, paid)

For tasks that genuinely can't be graded deterministically (instruction-
following quality, summarization faithfulness, creative writing, etc.), a
frontier model (Anthropic/OpenAI/Gemini/OpenRouter) can generate tasks and
grade the local model's responses per the rubric in
[TASK_SPEC.md](TASK_SPEC.md) — including a dedicated anti-hallucination
rule: the judge is never asked to recall real-world facts from memory as
ground truth, only to evaluate against self-contained or
harness-verifiable content. This never runs by accident:

1. Copy `.env.example` to `.env` and fill in the key for your chosen
   provider only.
2. Set `judge.enabled: true` in `config.yaml` and pick a `provider`/`model`.
3. Run `python cli.py run --job frontier_graded` — this prints exactly how
   many paid API calls it will make and requires a `y` confirmation before
   spending anything. It always shows in reports as a separate
   "Frontier-Graded (non-deterministic, paid)" section, never mixed into
   the six suites' pass-rate numbers.

Only install the SDK for the provider you're actually using (see
`requirements.txt`) — each is imported lazily, so the rest of localbench
works with none of them installed.

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
- **GPU/VRAM detection**: uses `nvidia-smi` when present, otherwise a
  Windows-specific WMI query (works for AMD/Intel/NVIDIA on Windows, but
  WMI's reported VRAM size is a known-unreliable 32-bit field on many
  drivers). No Linux/macOS GPU probes yet. None of the supported runtimes
  expose an exact GPU-vs-CPU offload ratio via their API/CLI — if
  tokens/sec looks much lower than expected for a model's size, check the
  runtime's own UI (e.g. LM Studio's Developer tab) to see whether it
  spilled into CPU/RAM.
- **`lms load` can stack duplicate model instances** instead of reusing an
  already-loaded one — `runtime.unload_all_cmd` exists specifically to
  guarantee a clean slate before every switch.

## Project layout

```
config.yaml            # your runtime/model/suite/judge configuration
.env.example            # documents the 4 judge API key names (.env is gitignored)
TASK_SPEC.md             # the frontier judge's task-generation/grading rubric
cli.py                   # entry point (run / serve)
localbench/
  engine.py               # runtime-agnostic streaming chat_completion() caller
  runner.py               # sequential model-switch orchestration
  report.py               # markdown/HTML/PDF report rendering
  storage.py              # save/list/load run records
  hardware.py             # CPU/RAM/GPU snapshot
  resource_monitor.py     # peak CPU/RAM sampling during a suite run
  results.py              # ProblemResult / SuiteRunResult / RunRecord
  suites/                 # one module per job suite
  data/                   # problem generators/data for each suite
  judge/                  # frontier judge provider clients + factory
webapp/
  main.py                 # FastAPI backend for the dashboard
  static/                 # HTML/CSS/JS frontend, no build step
results/runs/             # one JSON file per benchmark run (gitignored)
```
