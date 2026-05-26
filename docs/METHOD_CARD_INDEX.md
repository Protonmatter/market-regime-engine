# Method Card Index

Every production or frontier quantitative method must have a method card under
`docs/method_cards/`. Each card states production status, equations/contracts,
assumptions, failure modes, diagnostics, release-gate requirements, and tests.

The docs audit in `tests/test_method_cards_docs_audit.py` fails when a required
card is missing or omits mapped tests.
