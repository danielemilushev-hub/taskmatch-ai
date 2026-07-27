"""localbench entry point.

    python cli.py run --job coding
    python cli.py run --job all
    python cli.py serve
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from localbench import report
from localbench.config import load_config
from localbench.runner import run_benchmark

ALL_SUITES = [
    "json_schema",
    "coding",
    "logic_math",
    "instruction_following",
    "pattern_reasoning",
    "long_context",
]


def _make_stdout_unicode_safe() -> None:
    """Reports contain non-ASCII (the Δ in "RAM Δ", em dashes), and a Windows
    console defaults to a legacy codepage -- cp1251 here, cp437/cp1252
    elsewhere. Printing the report then raises UnicodeEncodeError and takes
    down the CLI *after* the benchmark already succeeded and saved, which
    looks like the run itself failed. Prefer UTF-8, and degrade to replacement
    characters rather than crashing on a console that can't represent them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def cmd_run(args: argparse.Namespace) -> None:
    _make_stdout_unicode_safe()
    config = load_config(args.config)

    if args.model:
        matching = next((m for m in config["models"] if m["name"] == args.model), None)
        if matching is None:
            print(
                f"note: '{args.model}' isn't in config.yaml -- benchmarking it in "
                "manual-switch mode (no load/unload command known for it)."
            )
            matching = {"name": args.model}
        config["models"] = [matching]

    run_frontier_graded = args.job == "frontier_graded"
    if run_frontier_graded:
        judge_cfg = config.get("judge", {})
        if not judge_cfg.get("enabled"):
            raise SystemExit(
                "judge.enabled is false in config.yaml -- set it to true and configure "
                "a provider/model before running --job frontier_graded"
            )
        print(
            f"This will make paid API calls to {judge_cfg.get('provider')}/{judge_cfg.get('model')} "
            f"({judge_cfg.get('num_tasks', 6)} tasks x 2 calls each: generate + grade, per model). "
            "Check your provider's pricing before proceeding."
        )
        confirm = input("Continue? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        suites_cfg = config.setdefault("suites", {})
        for suite_name in ALL_SUITES:
            suites_cfg.setdefault(suite_name, {})["enabled"] = False
    else:
        suites_cfg = config.setdefault("suites", {})
        for suite_name in ALL_SUITES:
            suite_entry = suites_cfg.setdefault(suite_name, {})
            if args.job == "all":
                continue
            suite_entry["enabled"] = suite_name == args.job

    run = run_benchmark(config, run_frontier_graded=run_frontier_graded)

    results_dir = Path(config.get("output", {}).get("results_dir", "results"))
    run_dict = run.to_dict()
    md = report.render_markdown(run_dict)

    md_path = results_dir / f"{run.run_id}.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\nmarkdown report written to {md_path}\n")
    print(md)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")
    uvicorn.run("webapp.main:app", host="127.0.0.1", port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="localbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run benchmark suites from the terminal")
    run_parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    run_parser.add_argument(
        "--job", default="all", choices=["all", *ALL_SUITES, "frontier_graded"]
    )
    run_parser.add_argument(
        "--model", default=None, help="benchmark only this model name from config.yaml"
    )
    run_parser.set_defaults(func=cmd_run)

    serve_parser = subparsers.add_parser("serve", help="launch the web dashboard")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--no-browser", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
