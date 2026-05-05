# SPDX-License-Identifier: Apache-2.0
"""Insert ``# SPDX-License-Identifier: Apache-2.0`` at the top of every
``.py`` file under ``src/market_regime_engine/``.

Idempotent: running twice does not duplicate the header. The script
preserves the existing leading shebang line if present, and slots the
SPDX header *before* the module docstring (so static analysers and the
Apache 2.0 SPDX scanner pick it up immediately).

Usage::

    python scripts/add_spdx_headers.py

By default the script targets ``src/market_regime_engine/**/*.py``. Pass
explicit file paths to override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HEADER = "# SPDX-License-Identifier: Apache-2.0\n"
SENTINEL = "SPDX-License-Identifier"


def _insertion_point(lines: list[str]) -> int:
    """Return the index at which to insert the SPDX header.

    Skip a leading shebang line if present so we never break ``#!`` files.
    """
    if lines and lines[0].startswith("#!"):
        return 1
    return 0


def add_header(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if SENTINEL in text.splitlines()[:5]:
        return False
    lines = text.splitlines(keepends=True)
    idx = _insertion_point(lines)
    new_lines = lines[:idx] + [HEADER] + lines[idx:]
    path.write_text("".join(new_lines), encoding="utf-8")
    return True


def discover_targets(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            out.append(root)
        elif root.is_dir():
            out.extend(sorted(root.rglob("*.py")))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to walk. Defaults to src/market_regime_engine.",
    )
    args = parser.parse_args(argv)
    if not args.paths:
        repo_root = Path(__file__).resolve().parents[1]
        args.paths = [repo_root / "src" / "market_regime_engine"]

    targets = discover_targets(args.paths)
    changed = 0
    for path in targets:
        if add_header(path):
            changed += 1
            print(f"  + {path}")
    print(f"SPDX header added to {changed} / {len(targets)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
