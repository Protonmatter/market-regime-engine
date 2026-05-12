"""Quick PR-9 bench dump for the PR body."""

from __future__ import annotations

import os
import tempfile
import time
import uuid

import pandas as pd

from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score,
)
from market_regime_engine.storage import Warehouse


def main() -> None:
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "perf.duckdb")
    wh = Warehouse(path=db_path)

    n_rows = 100_000
    ts = pd.date_range("2020-01-01", periods=n_rows, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "model_run_id": [f"run-{i}" for i in range(n_rows)],
            "timestamp": ts,
            "regime_score": [50.0 + (i % 20) for i in range(n_rows)],
            "regime_label": ["watch_transition"] * n_rows,
            "confidence": [0.5] * n_rows,
            "drivers_json": ["[]"] * n_rows,
            "component_scores_json": ["{}"] * n_rows,
            "release_gate": [1] * n_rows,
            "artifact_hash": [uuid.uuid4().hex] * n_rows,
            "metadata_json": ["{}"] * n_rows,
        }
    )
    wh.write_credit_regime_score(df)

    timings_ms: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter()
        latest_credit_regime_score(wh)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    p50 = timings_ms[len(timings_ms) // 2]
    p99 = timings_ms[-2]  # 99th percentile of 200 samples = idx 198
    print(f"PR-9 indexed-SQL latest_credit_regime_score on 100k rows:")
    print(f"  p50 = {p50:.3f} ms")
    print(f"  p99 = {p99:.3f} ms")
    print(f"  n_samples = {len(timings_ms)}")


if __name__ == "__main__":
    main()
