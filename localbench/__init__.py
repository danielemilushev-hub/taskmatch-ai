"""TaskMatch AI — task-driven benchmarking for locally-running LLMs.

`__version__` is the single source of truth for the running version. It is
surfaced in the dashboard header, by `cli.py --version`, and as the
cache-busting token on static assets, so "which version am I actually
running?" always has one answer you can read off the screen. A test asserts
it matches pyproject.toml so the two cannot drift apart.
"""

__version__ = "0.2.1"
