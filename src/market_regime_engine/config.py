# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_catalog(path: str | Path = "config/series_catalog.yaml") -> list[dict]:
    return list(load_yaml(path).get("series", []))


def env_db_path() -> str:
    return os.getenv("MRE_DB_PATH", "data/mre.db")
