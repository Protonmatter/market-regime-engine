# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "docs" / "method_cards"
REQUIRED = {
    "hmm.md",
    "msvar.md",
    "bayesian_msvar.md",
    "dfm_mq.md",
    "gw.md",
    "mcs.md",
    "conformal.md",
    "execution_confidence.md",
    "protocol_recommendation.md",
    "xpro_decision_artifact.md",
    "liquidity_stress.md",
    "credit_spread_regime.md",
}
REQUIRED_SECTIONS = [
    "## Production status",
    "## Module path",
    "## Mathematical equation",
    "## Inputs",
    "## Outputs",
    "## Assumptions",
    "## Failure modes",
    "## Diagnostics",
    "## Release-gate requirements",
    "## Tests that validate it",
    "## Known limitations",
]


def test_required_method_cards_exist() -> None:
    existing = {p.name for p in CARDS.glob("*.md")}
    assert REQUIRED <= existing


def test_method_cards_have_required_sections_and_tests() -> None:
    for name in REQUIRED:
        text = (CARDS / name).read_text(encoding="utf-8")
        missing = [section for section in REQUIRED_SECTIONS if section not in text]
        assert missing == [], f"{name} missing sections: {missing}"
        assert "tests/" in text, f"{name} must reference concrete tests"
