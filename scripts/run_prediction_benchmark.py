#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the Market Regime Engine prediction-evidence harness.

This is intentionally a file-based CLI so it can run in:
- GitHub Actions
- change-management jobs
- analyst workstations
- stripped production shells

Example
-------
python scripts/run_prediction_benchmark.py \
  --binary data/validation/binary_oos.csv \
  --quantile data/validation/quantile_oos.csv \
  --out-json data/validation/prediction_evidence.json \
  --out-md data/validation/PREDICTION_EVIDENCE.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from market_regime_engine.prediction_evidence import EvidenceThresholds, build_prediction_evidence_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an audit-grade prediction evidence report.")
    parser.add_argument("--binary", help="CSV/Parquet binary OOS predictions: date,target,horizon,model_name,y,p")
    parser.add_argument("--quantile", help="CSV/Parquet quantile OOS predictions with y and q_lo/q_hi or q05/q95")
    parser.add_argument("--out-json", default="data/validation/prediction_evidence.json")
    parser.add_argument("--out-md", default="data/validation/PREDICTION_EVIDENCE.md")
    parser.add_argument("--min-observations", type=int, default=60)
    parser.add_argument("--max-brier", type=float, default=0.25)
    parser.add_argument("--max-log-loss", type=float, default=0.75)
    parser.add_argument("--max-ece", type=float, default=0.08)
    parser.add_argument("--max-regime-ece", type=float, default=0.12)
    parser.add_argument("--min-interval-coverage", type=float, default=0.85)
    parser.add_argument("--max-tail-miss-rate", type=float, default=0.20)
    parser.add_argument("--fail-on-hold", action="store_true", help="Exit 2 when the evidence report is not approved.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = EvidenceThresholds(
        min_observations=args.min_observations,
        max_brier=args.max_brier,
        max_log_loss=args.max_log_loss,
        max_ece=args.max_ece,
        max_regime_ece=args.max_regime_ece,
        min_interval_coverage=args.min_interval_coverage,
        max_tail_miss_rate=args.max_tail_miss_rate,
    )
    report = build_prediction_evidence_report(
        binary_predictions=args.binary,
        quantile_predictions=args.quantile,
        thresholds=thresholds,
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(report.to_json() + "\n", encoding="utf-8")
    out_md.write_text(report.to_markdown(), encoding="utf-8")

    print(report.to_markdown())
    if args.fail_on_hold and not report.approved:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
