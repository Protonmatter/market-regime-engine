# SPDX-License-Identifier: Apache-2.0
"""Build the v1.4.1 combined release zip.

The combined zip mirrors the v1.4 bundle's structure and ships the
wheel + sdist + audit zip + V1_4_1_FIXES.md + the two demo JSON
captures in a single archive. The output path is
``C:/Users/mkang/market-regime-engine-v0.8/market-regime-engine-v1.4.1-release.zip``
(parent of the source tree, mirroring how the v1.4 bundle was built).

Run after ``scripts/build_release.py`` has produced the wheel + sdist
+ audit zip in ``dist/``::

    .venv\\Scripts\\python.exe scripts/build_v141_release_bundle.py
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DOCS = ROOT / "docs"
PARENT = ROOT.parent

VERSION = "1.4.1"
OUTPUT = PARENT / f"market-regime-engine-v{VERSION}-release.zip"

INPUTS = [
    (DIST / f"market_regime_engine-{VERSION}-py3-none-any.whl", f"market_regime_engine-{VERSION}-py3-none-any.whl"),
    (DIST / f"market_regime_engine-{VERSION}.tar.gz", f"market_regime_engine-{VERSION}.tar.gz"),
    (DIST / f"market-regime-engine-{VERSION}-source.zip", f"market-regime-engine-{VERSION}-source.zip"),
    (DOCS / "V1_4_1_FIXES.md", "V1_4_1_FIXES.md"),
    (DOCS / "v141_demo_verify_run_extra_drift.json", "v141_demo_verify_run_extra_drift.json"),
    (DOCS / "v141_demo_release_gate_default_production.json", "v141_demo_release_gate_default_production.json"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    missing = [src for src, _ in INPUTS if not src.exists()]
    if missing:
        print("FAIL: missing input artefacts:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "Run `python scripts/build_release.py --clean` first, then "
            "`python scripts/v141_capture_verify_demos.py` to produce "
            "the demo JSON captures.",
            file=sys.stderr,
        )
        return 1
    OUTPUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src, arcname in INPUTS:
            zf.write(src, arcname)
            print(f"  + {arcname}  ({src.stat().st_size:,} bytes)")
    bundle_sha = _sha256(OUTPUT)
    bundle_size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print()
    print(f"wrote {OUTPUT}")
    print(f"  size:   {bundle_size_mb:.2f} MB")
    print(f"  sha256: {bundle_sha}")
    print()
    print("Per-input sha256s:")
    for src, arcname in INPUTS:
        print(f"  {arcname:60s}  {_sha256(src)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
