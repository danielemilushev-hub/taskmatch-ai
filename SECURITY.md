# Security

## Reporting a vulnerability

Please report security issues privately via GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
rather than opening a public issue.

## What this tool does with your data

TaskMatch AI is designed to run entirely on your own machine. It is worth
being precise about what leaves it, because the whole point of benchmarking
local models is usually to *avoid* sending data anywhere.

**Nothing leaves your machine unless you explicitly enable the frontier
judge.** The six deterministic suites talk only to the local runtime URL you
configure (default `http://localhost:1234/v1`). The dashboard binds to
`127.0.0.1` and is not exposed to your network.

**The frontier judge is the one exception, and it is opt-in twice.** When
enabled, it sends generated task prompts and your local model's responses to
the third-party provider you select (Anthropic, OpenAI, Gemini, or
OpenRouter). It requires both an explicit config flag *and* a per-run
confirmation, and it never runs as part of a normal benchmark.

## Files that must never be committed

Both are in `.gitignore`, and CI fails the build if either becomes tracked:

| File | Why |
|---|---|
| `.env` | Your frontier provider API keys. |
| `config.yaml` | Your runtime URL, model list, judge choice, and — if you set `long_context.source_file` — a real path from your disk. |

Only `.env.example` and `config.example.yaml` are tracked. Copy them to
create your local versions.

`results/` is also gitignored: saved runs contain full prompts, full model
responses, and a hardware snapshot of the machine that produced them. Review
a run file before sharing it.

## How API keys are handled

- Keys live only in a local, gitignored `.env`.
- They are **never** written to `config.yaml`, which is the file most likely
  to be shared or committed.
- The `/api/settings` endpoint returns only a boolean per provider
  (`is set` / `not set`) — it never returns key values, so a saved key
  cannot be read back out through the dashboard.
- A key is sent only to the provider it belongs to, and only while the
  frontier judge is running.

## Code execution — read this before benchmarking untrusted models

The `coding` suite **executes model-generated Python on your machine.** That
is inherent to the measurement: the only honest way to know whether generated
code works is to run it against real tests.

"Sandboxed" here means a throwaway subprocess, in a temporary directory, with
a hard wall-clock timeout and no shared state with the harness. It is **not**
OS-level isolation — there is no seccomp, no cgroups, no chroot, no container,
and no filesystem or network restriction. Generated code runs with your user's
full permissions.

This is an accepted, documented limitation rather than an oversight; portable
sandboxing on Windows would require significantly more machinery. The practical
guidance:

- Benchmarking mainstream models you downloaded yourself is the intended use
  and is low risk.
- **Do not** point the coding suite at a model you do not trust, or at one
  serving output you did not generate, on a machine you care about. Use a VM
  or container if you need that.
- To avoid code execution entirely, leave the `coding` suite unticked.

## Network exposure

`python cli.py serve` binds uvicorn to `127.0.0.1`, so the dashboard is
reachable only from your own machine. If you change that binding, note that
the API has **no authentication** — it can start runs, delete saved runs, and
write API keys to `.env`. Do not expose it to an untrusted network.

## Dependencies

The frontier judge SDKs (`anthropic`, `openai`, `google-genai`) are
deliberately *not* in `requirements.txt` and are imported lazily. If you never
use the judge, you never install them.
