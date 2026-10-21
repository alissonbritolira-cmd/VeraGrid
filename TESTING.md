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

## Experimental RMS disk trajectories and recovery

The RMS settings page has an optional **Write accepted samples to disk in blocks**
setting and a recovery-folder selector. Memory storage remains the default. The
three native Backward Euler implementations can write regular accepted rows to
SQLite transactions instead of retaining the full trajectory tensor. This does
not add an IDA integrator or change the numerical equations or tolerances.

This adapts the existing development SQLite store, including its metadata,
`rms_groups`, `blobs`, `tiles`, `clocks` and `large_metadata` tables. Transactions
commit accepted Float64 rows and their clocks together. The lossless codec uses
XOR differences and reversible byte planes, with 16-channel tiles and SHA-256
verification. Its staging buffer is bounded at 32 MiB and its decoded SQLite
tile cache at 32 MiB; block height is at most 512 rows. Saved `.veragrid` tensors
use the existing development `result_index.json` format, 16-channel NPY tiles
and a 16 MiB read cache, so normal session restore/resave need not materialize
the full tensor.

The clock arrays, circuit, integrator state and irregular metadata still occupy
memory; clock memory grows with the number of steps. Requesting the whole array
with `np.asarray()` explicitly materializes it. This is not a constant-memory
solver. Native timestamp and available-memory limits still apply. At least
512 MiB plus the 32 MiB staging allowance must be free before input preparation;
the 512 MiB reserve is checked again before each commit. This is a reserve, not
an estimate of the complete run size. Prefer a local disk. Network shares,
partially synchronized cloud copies and filesystem power-loss behavior have not
been qualified by this contribution. No IDA or integrator restart is included.

**Recover results...** verifies the preserved input archive, metadata, block
coverage and every selected tile's digest. It exports a new `.veragrid` history
while preserving the input/recovery files and any existing study. Recovery
restores confirmed regular samples for inspection; it does not resume the
integrator or reconstruct an uncommitted buffer. A complete committed result
snapshot retains its histories, maps and irregular records. Without that snapshot,
the missing final/irregular state is disclosed and initialization/convergence are
false. A snapshot whose accepted count exceeds its committed frontier is refused
rather than silently discarding persisted state. The development JSON-stream
reader currently limits an individual decoded metadata stream to 128 MiB, so very
large histories can require a later metadata-stream extension. A recovered partial
group remains unconverged. The `recovered` flag distinguishes recovered history
from the original live run.

Run focused checks:

```bash
python -m pytest src/tests/RMS/test_rms_disk_trajectory.py src/tests/RMS/test_rms_cancelled_results.py src/tests/GUI/test_rms_storage_settings.py src/tests/GUI/test_rms_cancelled_workflow.py src/tests/Base/test_dynamic_results_save_load.py
```

For real desktop validation, run a public RMS case in memory and disk modes with
identical options and compare accepted times, variables and event-group identities.
Cancel a disk run through the normal GUI and inspect a variable curve. Save with
results enabled, reopen and load the saved RMS study through its context menu.
Then use **Recover results...** to choose that execution's UUID subfolder and a new
output name; reopen the exported history and compare its confirmed prefix. Cancel
recovery during a long export, verify that originals remain unchanged and no
partial output is presented as complete, and repeat recovery. Close the application
while a recovery worker is active to verify that shutdown waits for it. The native
Qt settings/worker tests run internally; they are not evidence of Windows mouse and
keyboard navigation.
