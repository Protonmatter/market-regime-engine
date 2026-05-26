# XPro Track B Validation

Use these commands for a repeatable Track B release check.

```powershell
python -m compileall -q src tests
python -m pytest --collect-only -q
python -m pytest -q tests/test_numeric_contracts.py tests/test_protocol_recommendation.py tests/test_xpro_decision_artifact.py
python -m pytest -q tests/test_xpro_decision_api_endpoint.py tests/test_xpro_decision_cli.py
python -m pytest -q tests/test_execution_validation_certification_cli.py tests/test_storage_xpro_decision_artifacts.py
python -m pytest -q tests/test_execution_confidence.py tests/test_execution_confidence_api_endpoint.py tests/test_execution_confidence_cli.py
python -m pytest -q tests/test_canonical_json_rfc8785.py tests/test_fixed_income_evidence_pack_hmac.py tests/test_certification_release_and_execution_validation.py
python -m pytest -q tests/test_method_cards_docs_audit.py
```

If full-suite runtime needs investigation, run:

```powershell
python -m pytest -q --durations=25
```

The XPro verifier accepts unsigned hash-valid artifacts in dev mode. Use strict verification when required:

```powershell
mre fi-verify-xpro-decision --db data/mre.duckdb --decision-id <decision_id> --require-hmac
```
