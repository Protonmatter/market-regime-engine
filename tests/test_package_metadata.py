# SPDX-License-Identifier: Apache-2.0
"""Package-sanity tests for the v1.2.1 wheel install round-trip.

The CI ``package-sanity`` job builds the wheel, installs it into a fresh
venv outside the source tree, and runs this module. The assertions here
guarantee that:

- The package metadata version matches the source ``__version__``.
- The package can be imported without warnings (no leftover
  ``DeprecationWarning`` from the v1.1 PIT-router refactor).
- The Apache-2.0 license metadata is present.
- A representative public symbol is importable.

Run from the repository root, the test still passes against the editable
install — the metadata version is read via ``importlib.metadata`` which
handles both editable and wheel-installed packages.
"""

from __future__ import annotations

import warnings


def test_metadata_version_matches_source() -> None:
    from importlib import metadata

    import market_regime_engine

    installed = metadata.version("market-regime-engine")
    assert installed == market_regime_engine.__version__, (
        f"metadata version {installed!r} != source __version__ {market_regime_engine.__version__!r}. Reinstall."
    )
    # v1.4 baseline: every release must keep the version monotone-increasing
    # (1.10.0 > 1.3.0 etc) and stay >= the v1.3 floor.
    parts = tuple(int(p) for p in market_regime_engine.__version__.split(".")[:3])
    assert parts >= (1, 3, 0), f"version regressed below v1.3 floor: {market_regime_engine.__version__}"


def test_metadata_declares_apache_license() -> None:
    from importlib import metadata

    md = metadata.metadata("market-regime-engine")
    raw_classifiers = md.get_all("Classifier") or []
    license_field = md.get("License") or md.get("License-Expression") or ""
    license_files = md.get_all("License-File") or []
    license_blob = " ".join(
        [
            *raw_classifiers,
            license_field,
            *license_files,
        ]
    ).lower()
    assert "apache" in license_blob, (
        f"package metadata does not declare Apache-2.0; saw {license_blob!r}. "
        "Check pyproject [project] license + classifiers."
    )


def test_package_imports_clean() -> None:
    """The top-level import must not surface a DeprecationWarning. v1.1
    introduced an explicit ``filterwarnings = ["error::DeprecationWarning:
    market_regime_engine"]`` rule so any new import-time deprecation
    immediately blows up; this test pins the contract."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Force a fresh import path. We deliberately re-import the top
        # package (already imported by the test runner) — this just
        # walks the module's __init__ statements again and would trip
        # the filterwarnings rule if any of them emit a warning.
        import importlib

        import market_regime_engine

        importlib.reload(market_regime_engine)


def test_public_symbols_are_importable() -> None:
    """Smoke: a representative subset of the v1.2.1 public surface must
    import without exceptions. We deliberately exclude the legacy ``api``
    mount (which now raises at import time without the env-var ack)."""
    from market_regime_engine import (  # noqa: F401
        __version__,
        configure_logging,
        get_logger,
    )
    from market_regime_engine.api_v1 import app  # noqa: F401
    from market_regime_engine.asof import (  # noqa: F401
        latest_vintage_observations_per_asof_grid,
        materialize_feature_asof_values,
    )
    from market_regime_engine.cli import main  # noqa: F401
    from market_regime_engine.model_runs import (  # noqa: F401
        build_repro_envelope,
        create_model_run,
        verify_run,
    )
    from market_regime_engine.training_data import (  # noqa: F401
        TrainingMode,
        load_training_panel,
    )


def test_console_script_entrypoint_resolves() -> None:
    """``mre`` must resolve to ``market_regime_engine.cli_dispatch:main``
    per pyproject ``[project.scripts]`` so the wheel exposes the CLI.

    v1.5 (PR-1 dispatch landing): the entry point flipped from
    ``cli:main`` to ``cli_dispatch:main`` so the FI ``fi-*`` fast-path
    can route before delegating to the legacy ``cli.main`` for the
    macro commands. The test now pins the new value.
    """
    from importlib import metadata

    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        scripts = list(eps.select(group="console_scripts"))
    else:  # pragma: no cover - we require 3.11+ but be defensive
        scripts = list(eps.get("console_scripts", []))
    names = [ep.name for ep in scripts]
    assert "mre" in names, f"console_scripts missing 'mre' entry; saw {names!r}"
    mre_ep = next(ep for ep in scripts if ep.name == "mre")
    assert mre_ep.value == "market_regime_engine.cli_dispatch:main"
