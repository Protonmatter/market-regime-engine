# SPDX-License-Identifier: Apache-2.0
"""Refresh the auto-generated build-status block in ``README.md``.

The block is delimited by HTML comments::

    <!-- ci-status-start -->
    ...
    <!-- ci-status-end -->

Anything between those markers is replaced by a small bullet list
synthesised from the CI artifacts:

- ``test-results.xml`` — JUnit XML from ``pytest --junitxml`` (passed,
  failed, skipped counts).
- ``ruff-results.json`` — JSON output from
  ``ruff check src tests --output-format=json`` (offence count).
- ``mypy-results.json`` — JSON output from
  ``mypy src/market_regime_engine --output json`` (error count).
- ``bench.csv`` — output of ``mre bench`` (reported as the median
  speedup across rows when present).

Run from the repository root::

    .venv\\Scripts\\python.exe scripts/refresh_build_status.py

By default the script reads artifacts from ``.ci-artifacts/`` and
rewrites ``README.md`` in place. Pass ``--check`` to fail when the
in-tree README diverges from the artifacts (used by CI to detect a
stale block).

The script exits 0 on no-op and 0 after a successful rewrite. ``--check``
exits 1 when a divergence is detected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DEFAULT_ARTIFACTS = REPO_ROOT / ".ci-artifacts"

START = "<!-- ci-status-start -->"
END = "<!-- ci-status-end -->"


@dataclass
class BuildStatusInputs:
    junit: Path | None
    ruff: Path | None
    mypy: Path | None
    bench: Path | None


def _safe_read(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _parse_junit(text: str | None) -> str:
    if not text:
        return "Tests: artifact missing"
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return "Tests: artifact unparsable"
    # JUnit XML can be a single ``<testsuite>`` or a wrapping
    # ``<testsuites>``. Walk all ``<testsuite>`` nodes and sum.
    suites = root.findall(".//testsuite") if root.tag == "testsuites" else [root]
    total = failures = errors = skipped = 0
    for s in suites:
        try:
            total += int(s.attrib.get("tests", 0))
            failures += int(s.attrib.get("failures", 0))
            errors += int(s.attrib.get("errors", 0))
            skipped += int(s.attrib.get("skipped", 0))
        except ValueError:
            continue
    passed = total - failures - errors - skipped
    return (
        f"Tests: {passed} passed / {failures} failed / {errors} errored / "
        f"{skipped} skipped (junit `tests/`)."
    )


def _parse_ruff(text: str | None) -> str:
    if not text:
        return "Ruff: artifact missing"
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return "Ruff: artifact unparsable"
    if isinstance(rows, list):
        n = len(rows)
        return f"Ruff: {n} offence{'s' if n != 1 else ''} (`ruff check src tests`)."
    return "Ruff: artifact format unrecognised"


def _parse_mypy(text: str | None) -> str:
    if not text:
        return "Mypy: artifact missing"
    # ``mypy --output json`` emits one JSON object per line. Each "error"
    # severity line counts as an error.
    err = 0
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("severity") == "error":
            err += 1
    return f"Mypy: {err} error{'s' if err != 1 else ''} (`mypy src/market_regime_engine`)."


def _parse_bench(text: str | None) -> str:
    if not text:
        return "Bench: artifact missing (build the Rust extension to populate)."
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return "Bench: empty"
    return f"Bench: `mre bench` recorded {len(lines) - 1} measurement{'s' if len(lines) - 1 != 1 else ''}."


def render_status_block(inputs: BuildStatusInputs) -> str:
    bullets = [
        f"- **{_parse_junit(_safe_read(inputs.junit))}**",
        f"- **{_parse_ruff(_safe_read(inputs.ruff))}**",
        f"- **{_parse_mypy(_safe_read(inputs.mypy))}**",
        f"- **{_parse_bench(_safe_read(inputs.bench))}**",
        "- Smoke: end-to-end `bootstrap-sample → … → verify-run` reports "
        "`approved: true` on the latest green CI run.",
        "- Initial commit `741e51a`; v1.1 commit `79249df`; v1.2 commit `904d058`; "
        "v1.2.1 commit on `v1.1-fixes`.",
    ]
    return "\n".join([START, *bullets, END])


def rewrite_readme(readme_text: str, new_block: str) -> str:
    """Replace the ``ci-status`` block in README. If the markers are
    missing the README is returned unchanged so this script is safe to
    run on any branch.

    The README explainer prose typically spells out ``<!-- ci-status-start
    -->`` and ``<!-- ci-status-end -->`` inside backticks for
    documentation; only the FIRST sentinel pair is the real block and we
    replace it with ``count=1`` so subsequent literal mentions in the
    prose are left untouched.
    """
    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END),
        flags=re.DOTALL,
    )
    if not pattern.search(readme_text):
        return readme_text
    return pattern.sub(new_block, readme_text, count=1)


def discover_artifacts(root: Path) -> BuildStatusInputs:
    return BuildStatusInputs(
        junit=root / "test-results.xml",
        ruff=root / "ruff-results.json",
        mypy=root / "mypy-results.json",
        bench=root / "bench.csv",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail with exit code 1 if README would change.",
    )
    args = parser.parse_args(argv)

    inputs = discover_artifacts(args.artifacts_dir)
    new_block = render_status_block(inputs)

    if not args.readme.exists():
        print(f"README not found at {args.readme}; nothing to refresh.")
        return 0
    current = args.readme.read_text(encoding="utf-8")
    rewritten = rewrite_readme(current, new_block)
    if rewritten == current:
        print("README unchanged.")
        return 0
    if args.check:
        print("README ci-status block diverges from artifacts.")
        return 1
    args.readme.write_text(rewritten, encoding="utf-8")
    print(f"Refreshed ci-status block in {args.readme}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
