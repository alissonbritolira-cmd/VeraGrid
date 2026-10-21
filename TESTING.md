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

### Optional indexed RMS archives

In **Settings > File**, enable **Save results in .veragrid files** and then
**Save RMS channels in chunks (experimental)**. Save a completed or cancelled
RMS study, reopen it, and load its saved RMS result through the context menu.
Open one variable curve and a multi-variable plot group. The values, accepted
prefix, group availability, cancellation flags and parameter-derived curves
must match the dense archive. Save again to the same file and consult another
curve without reopening. Disable chunked saving to export the historical dense
layout and confirm that this archive still opens in an older version.
While a current result still reads from a source archive, overwriting that
archive without results is refused by the GUI. Enable result saving or use
Save As. Engine callers are responsible for retaining source archives used by
their lazy result objects.

The checkbox is off by default. The optional layout is versioned and stores
float64 NPY row chunks independently for each variable and event group, with a
SHA-256 index. Opening a saved session restores the small metadata immediately;
plotting reads the requested channels. It does not reduce the integration
driver's memory, recover an interrupted unsaved simulation, or resume a solver.
Older builds cannot read this optional RMS result layout; their ordinary dense
archives remain readable and the default writer is unchanged.

Engine callers can opt in with `results.archive_chunk_rows = 131072` before
saving. Supported counts are 1..131072; the current implementation caps an index
at 50000 chunks and 16 MiB. Indexed tensors support integer/slice indexing;
`np.asarray(results.values)` explicitly loads the entire tensor. No files are
extracted and no pickle payloads are used. A missing/changed source archive or
invalid chunk raises an error instead of substituting a safe-looking curve.

Focused automated coverage (Qt tests are separate from desktop acceptance):

```bash
python -m pytest src/tests/RMS/test_rms_archive.py src/tests/GUI/test_rms_archive_options.py src/tests/Base/test_dynamic_results_save_load.py
```

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
