# Testing

VeraGrid uses automated tests, linting, and fuzzing to reduce regressions in file parsing, simulations, and user-facing workflows.

## Main automated test suite

The primary test runner is `pytest`.

- Test root: `src/tests`
- Discovery rules: [pytest.ini](pytest.ini)
- File pattern: `test_*.py`

Run the full suite:

```bash
python -m pytest
```

Run a focused subset:

```bash
python -m pytest src/tests/FileFormats
python -m pytest src/tests/GUI
```

For RMS cancellation coverage:

```bash
python -m pytest src/tests/RMS/test_rms_cancelled_results.py src/tests/GUI/test_rms_cancelled_workflow.py
```

For a manual check, start RMS from a converged power flow, cancel during
integration, and inspect a variable curve. The accepted prefix should remain
available, the unused tail should be NaN, and the GUI should report cancellation
with partial results. Save with results enabled, reopen the file, and load the
saved RMS study through its context menu. The same samples and cancellation
status should be restored. A new RMS run should still complete normally.

## Test policy

As major new functionality is added, automated tests should be added with it.

The same applies to bug fixes whenever the bug can be reproduced in a stable test.

This policy is enforced through pull request review and documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## Linting and static analysis

The repository uses:

- `pylint` in GitHub Actions for Python linting
- Code scanning via Scorecard SARIF upload and GitHub security tooling

Run pylint locally:

```bash
python -m pylint $(git ls-files '*.py')
```

## Fuzzing and dynamic analysis

The repository includes ClusterFuzzLite integration and Python fuzz targets under `fuzz/`.

- Workflow: [.github/workflows/cflite_pr.yml](.github/workflows/cflite_pr.yml)
- Fuzz target example: [fuzz/raw_functions_fuzzer.py](fuzz/raw_functions_fuzzer.py)

These targets are intended to exercise parser and deserialization code paths that are prone to malformed-input bugs.

## Pre-merge expectations

Before opening or updating a pull request:

- Run the relevant `pytest` tests locally.
- Run `pylint` on changed Python code when practical.
- Add or update tests for any new behavior or bug fix.
- Call out any untested risk explicitly in the pull request.
