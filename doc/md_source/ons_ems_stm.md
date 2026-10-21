# ONS EMS state-estimator STM/PRM import

Open an STM through **File > Open**. Optionally select its PRM in the same file
picker or use **Choose PRM** in the import dialog. An unselected neighbouring
file is never read. STM does not declare system frequency, so confirm the
frequency in the dialog (initially 60 Hz).

The importer adds a bounded ONS EMS STM v5 AC-network conversion alongside the
other native formats. It does not execute an external study or import DYN,
events or DSA.

## Electrical conversion supported by this first slice

| Data | Treatment |
| --- | --- |
| STM version/base | Version 5; positive finite system MVA base |
| Buses | Bus/section code, name, kV, PQ/PV/reference/disconnected state, voltage seed, angle and limits |
| Groups | Areas, zones, owners and substations; bus area/zone/substation references |
| Generators | Local V or Q control, MW/Mvar, limits, nominal MVA, enabled state and dispatch permission |
| Loads | Constant-power, current and impedance portions; current/impedance normalized from the initial voltage |
| Lines | Per-unit R/X, total charging Mvar converted using the system base, ratings, fixed state and terminal shunts |
| Transformers | Fixed tap ratio and phase in radians, R/X, ratings and both-closed/both-open state |
| Fixed shunts | BUS G/B injection; a SHUNTS mode-0 declaration is retained without counting it twice |
| Source data | Complete decoded STM/selected PRM text, SHA256, filenames, assignment order and diagnostics in native circuit metadata |

The normal rating maps to the native rating; the source third rating supplies
the contingency/protection ratio when the normal rating is nonzero. The second
rating and fields without an equivalent in this PF slice remain in the original
source text. This does not qualify short-circuit, dynamic or contingency-study
equivalence with an external application.

## Larger cases: read-only inspection

DC converter records, VSCs, series capacitors, switched shunts, SVCs, remote
generator voltage regulation, automatic transformer controls and individual
open-terminal equivalents need additional qualified adapters. The GUI opens a
**source inspection** instead of replacing the current project when it finds
these features. It contains all 14 sections with source line numbers, full
original text, PRM assignments and the unsupported-feature counts. It does not
create a partial circuit or offer simulation of it.

The engine `FileOpen` path rejects such a conversion with an explanation. It
never silently removes devices to obtain a solvable network. Malformed records,
duplicate identifiers, missing buses, invalid states and non-finite electrical
quantities are also refused.

## PRM precedence and consultation

The PRM remains descriptive unless the user explicitly selects **Apply supported
PRM flow settings**. This first contribution can apply only:

* `PFMAXITER`: a positive integer that fits the GUI iteration field.
* `PFFLTSTRT`: `T` disables the existing **use voltage guess** setting; `F` enables it.

The current GUI settings otherwise remain authoritative. Tolerance units,
delayed Q switching, transformer/shunt permissions and commands such as
`CTGBLOCKQG` are not approximated. They are shown as **preserved, not applied**.
Selecting the checkbox does not execute those other commands. A PRM is never
evaluated as Python, a shell command or a script.

After a supported import, use **File > ONS EMS import data** to inspect original
text, hashes, assignment history and which options were explicitly applied.
Use the existing PF settings to review or change the effective calculation
options. Saving as `.veragrid` stores the original sources and GUI settings.
Reopening does not re-execute source commands. Older native files without
import metadata remain readable.

## Examples and checks

`src/tests/FileFormats/data/ons_ems/ac_network_v5.stm` is a small synthetic,
redistributable example covering the supported AC conversion. Its PF is
compared with an independently constructed native circuit and its native
save/reopen path retains the electrical values and provenance.

The two larger cases in `examples/ons_ems` are explicitly **inspection and
coverage examples**. They contain unsupported features and are not represented
as qualified PF imports. Their README records exact names, source hashes and
current limitations.

Run the focused automated checks:

```text
python -m pytest src/tests/FileFormats/test_ons_ems_stm.py src/tests/GUI/test_ons_ems_import.py src/tests/GUI/test_io_open_file_single_path.py
```

Native desktop acceptance should separately check opening the synthetic case,
choosing/removing a PRM, applying explicit PF options, F5, inspecting sources,
saving/reopening and cancelling import. For each larger case, verify that all
source records remain available in the inspection and the existing project is
unchanged after closing it.
