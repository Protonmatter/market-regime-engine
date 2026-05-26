# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_regime_engine"
ALLOWED_STABLE_FRONTIER_IMPORTERS = {
    "release_gates.py",
    "package_boundary.py",
    "cli_handlers.py",  # command boundary; commands are explicit operator entrypoints
    "orchestration.py",  # research/stable workflow boundary
}
ALLOWED_FRONTIER_UTILITY_MODULES = {
    "market_regime_engine.frontier",
    "market_regime_engine.frontier.data_cleaning",
    "market_regime_engine.frontier.release_calendars",
    "market_regime_engine.frontier.experimental",
}


def _absolute_import_name(node: ast.ImportFrom) -> str | None:
    if node.module is None:
        return None
    if node.level == 0:
        return node.module
    # Convert relative imports inside market_regime_engine back to absolute
    # names so ``from .frontier.dfm_mq import ...`` cannot bypass the boundary
    # audit. The test files are scanned relative to SRC/package root.
    prefix = "market_regime_engine"
    if node.level == 1:
        return f"{prefix}.{node.module}"
    return node.module


def test_stable_core_does_not_import_frontier_implementations_directly() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if rel.parts[0] == "frontier":
            continue
        if str(rel) in ALLOWED_STABLE_FRONTIER_IMPORTERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = _absolute_import_name(node)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name.startswith("market_regime_engine.frontier")
                        and alias.name not in ALLOWED_FRONTIER_UTILITY_MODULES
                    ):
                        offenders.append(f"{rel}: import {alias.name}")
            if mod and mod.startswith("market_regime_engine.frontier") and mod not in ALLOWED_FRONTIER_UTILITY_MODULES:
                offenders.append(f"{rel}: from {mod} import ...")
    assert offenders == []
