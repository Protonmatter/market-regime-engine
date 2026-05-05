# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for the v1.2 ``build_zip.py``.

The real implementation moved to ``scripts/build_audit_zip.py`` in v1.2.1
so the audit zip lives next to the new ``scripts/build_release.py``
release driver. Existing tooling that called ``python build_zip.py``
continues to work.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "build_audit_zip.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
