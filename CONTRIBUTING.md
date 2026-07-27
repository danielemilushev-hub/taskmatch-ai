# Contributing

## Setup

```bash
git clone <your-fork>
cd taskmatch-ai
pip install -r requirements.txt
cp config.example.yaml config.yaml     # Windows: copy config.example.yaml config.yaml
```

Point `runtime.base_url` in `config.yaml` at your local runtime, then:

```bash
python cli.py serve
```

## Running tests

```bash
python -m unittest discover -s tests -v
```

CI runs these on Linux and Windows against Python 3.11 and 3.12, plus an
import check over the modules the unit tests don't reach, plus a scan that
fails the build if `config.yaml`, `.env`, or anything resembling a credential
ever becomes tracked.

## The one rule that matters most

**Never display a number the tool did not actually measure.**

This project exists to answer "which local model should I use for which job"
with real data. A plausible-looking fabricated figure is worse than no figure,
because it silently destroys the reason to use the tool at all. In practice:

- If a probe is unavailable, render `not captured` / `n/a` — never a default,
  an estimate, or a zero.
- If a number is illustrative rather than measured (e.g. a placeholder API
  rate), label it as illustrative *at the point it is displayed*, not only in
  a comment.
- Distinguish "the measurement failed" from "the model scored badly." A judge
  that returned HTTP 429 is not a model that scored 0% — the codebase has
  `SuiteRunResult.judge_infrastructure_failed` specifically for this.
- Prefer hedged wording for heuristics. The "possible spillover" badge is
  worded that way deliberately: the thresholds are uncalibrated.
- Percentages that are shares of time (CPU, GPU utilization) are clamped to
  0–100, because a value above 100 is physically impossible and indicates
  counter overshoot.

Several existing comments explain *why* a metric is computed a particular
way. Please keep that habit — the reasoning is usually less obvious than the
code.

## Adding a job suite

1. Add a problem generator or fixed problem list under `localbench/data/`.
2. Add `localbench/suites/<name>_suite.py` exposing:

   ```python
   def run(ctx, ..., on_progress=None) -> list[ProblemResult]
   ```

   Follow the existing shape: a `_run_one(problem, ctx, call_kwargs)` helper
   returning one `ProblemResult`, with `run()` looping and calling
   `on_progress(idx + 1, total, problem_id, passed)` after each one. That
   callback is what drives the live per-problem progress log.

3. Register it in `localbench/runner.py`, in `ALL_SUITES` in both `cli.py`
   and `webapp/main.py`, and in `SUITE_METADATA` in `webapp/static/app.js`.
4. Grading must be deterministic — exact match, schema validation, or actual
   execution. If a task can only be graded by judgement, it belongs in the
   frontier-graded suite, not here.

Handle these three outcomes distinctly, never collapsing them into "failed":

- the call failed (network/timeout) → `error`, no `truncated`
- the response hit the token budget → `truncated=True`
- the model answered, and the answer was wrong → `error` describing the diff

## Frontend

Plain HTML/CSS/JS, no build step. Two conventions:

- **Build DOM with `createElement` + `textContent`**, not `innerHTML` string
  interpolation, anywhere model output or a run id is involved.
- **Use `confirmDialog()`, never `window.confirm()`.** Embedded and preview
  browsers suppress native dialogs and return `false` silently, which turns
  any confirmed action into a no-op that looks like a broken button.

Colours come from the CSS custom properties at the top of `styles.css`
(`--series-*` for categorical, `--good`/`--warning`/`--critical` for status).
Don't introduce raw hex values, and don't reuse status colours for series.

## Commit messages

Explain *why*, especially for a fix: what the wrong behaviour was, what caused
it, and how you confirmed it's fixed. The existing history follows this and is
worth skimming for the tone.
