# ONS EMS state-estimator STM/PRM import

Open an STM through **File > Open**. Select its PRM in the same file picker or
use **Choose PRM** in the import dialog. The PRM is optional, and an unselected
neighbouring file is never read. Confirm the system frequency in the dialog
(initially 60 Hz); STM does not declare it.

This format imports an ONS EMS STM v5 network into the existing VeraGrid project.
The electrical adapters reuse the developed STM power-flow path. They do not
import DYN models, events or DSA studies, or execute an external application.

## Network conversion

| Data | Treatment |
| --- | --- |
| Buses and groups | Original identifiers, sections, names, kV, states, voltage seeds, limits, areas, zones, owners and substations |
| Generators | MW/Mvar, limits, nominal MVA, enabled and dispatch states, local and supported shared remote voltage regulation |
| Loads | Constant-power, current and impedance portions; selected PRM load laws are applied on the private calculation state |
| Lines | R/X, charging, ratings, terminal shunts and individual terminal states; an open-ended line retains its source branch and the developed closed-end input-admittance representation |
| Transformers | Two-winding R/X, tap ratio, phase, ratings, terminal states and supported automatic tap/phase controls |
| Series capacitors | Negative series reactance and the developed bypass-state treatment; original values and conversion adjustments remain recorded |
| Shunts | BUS fixed injection without double counting mode-0 declarations; switched-bank positions and steps; continuous SVC limits and voltage targets |
| DC poles | Existing LCC/CCC bridge, losses, terminal reactive-power and AC-coupling adapters; source orders and converter records remain available |
| Provenance | Complete decoded STM and selected PRM, filenames, SHA256, all section records with line numbers, assignment history and effective calculation metadata |

Equipment is not deleted to make a case converge. A reference bus that was
already isolated and has no source is deenergized only in a private PF state;
the original bus and registered loads remain in the project. The solver logs
this treatment. The native **ignore single-node islands** setting is not enabled
silently. Inactive equipment with an absent terminal is retained with an explicit
association restriction and cannot be energized without the missing data.

The converter and control adapters validate their supported electrical domain.
An unsupported requested control, incomplete active converter, invalid
association or numerical failure produces a diagnostic rather than an omitted
device or a fabricated successful calculation. Preserved terminal auxiliaries
that have no associated branch retain the developed explicit non-applied
contract; they are not converted into an invented branch.

## PRM settings and precedence

After importing, review **Settings > Power flow** before pressing **F5**.
The selected PRM seeds the developed numeric PF defaults, including tolerance,
maximum iterations and voltage-guess use. The developed path assigns
`PFMVATOL` directly to native PF tolerance: `0.001` appears as `1e-3 pu`
in the GUI. Its voltage-control adapter can refine internal precision separately;
the requested and effective tolerances are retained in execution metadata.
No newly inferred MVA/base conversion is introduced by this port.

Explicit GUI choices take precedence over the selected PRM, followed by native
defaults. Three automatic control permissions start blocked on a fresh import:

* Transformer tap magnitude.
* Transformer phase.
* Discrete shunt banks, with their own checkbox.

Use **Apply PRM options** in the PF settings to review requested changes before
applying supported permissions. The import dialog can open that same preview
after loading. Cancelling the preview leaves the options unchanged. Continuous
SVCs and remote generator regulation follow the developed voltage-control path
and their limits; they are distinct from discrete bank switching.

The parameter inventory separates supported, conditional, inactive, superseded
and unimplemented commands. The PF adapters handle supported load-law, reactive
limit, tap/shunt, thermal-metric and low-voltage policies. An intermediate
`PFGENVARIT` delay is not qualified: an explicit GUI reactive-limit choice is
recorded as an override, not as execution of that delay. Distributed slack,
unqualified area controls and a CP/I-to-Z transition below `PFVTHRS` are not
silently approximated. Contingency-prefixed commands can be inspected but this
importer does not execute a contingency study.

Use **File > ONS EMS import data** to inspect all original records, PRM command
status, effective options, source hashes and raw text. Saving as `.veragrid`
preserves the sources, associations, source/effective control metadata and GUI
choices. Reopening uses the saved project without requiring the source files.
Older native projects without import metadata remain readable.

## Qualification limits

Successful import and a converged PF do not establish numerical equivalence
with another application. The port preserves the developed LCC/CCC qualification
metadata, including its experimental external-equivalence status. It does not
turn that status into a claim of complete reference-engine compatibility.
Unsupported VSC/control variants remain subject to the existing adapters'
explicit checks. Short-circuit, RMS, event, contingency and security-region
equivalence are outside this contribution.

DC nominal ratings are retained as equipment data. They do not automatically
become operative transfer limits. Without a registered operative limit, the
developed policy leaves that pole's loading indicator unavailable (`NaN`) and
records `missing_limit` with `effective_dc_limits.complete=False`. Terminal
power and losses can still be solved. Such an execution does not establish
compliance with missing DC limits.

## Examples and checks

`examples/ons_ems` contains the two contributor-supplied full STM/PRM pairs. Both
are imported as complete networks; they are not substituted with a read-only
preview. The MS case contains 1,958 buses and 3,056 AC branches. The SIN case
contains 10,614 buses and 14,134 AC branches, including active DC converters.
Their README records the exact names, hashes and scope.

`src/tests/FileFormats/data/ons_ems/ac_network_v5.stm` is a small synthetic
network used for an independent native-circuit PF comparison. The developed
control fixture and ported regression tests cover private-state preservation,
LCC/CCC energy and AC coupling, continuous/discrete controls, rollback,
cancellation, repeated execution, native persistence and the import widgets.

```text
python -m pytest src/tests/FileFormats/test_ons_ems_stm.py src/tests/FileFormats/test_ons_ems_developed_pf.py src/tests/FileFormats/test_ons_ems_developed_ccc.py src/tests/FileFormats/test_ons_ems_developed_discrete.py src/tests/GUI/test_ons_ems_import.py src/tests/GUI/test_io_open_file_single_path.py
```

Native desktop acceptance is separate from these automated checks. Open each
supplied pair with the actual file picker, review the import dialog and PF
controls, run F5, inspect convergence and diagnostics, consult source/effective
PRM data, then save and reopen. Also exercise PRM removal, a cancelled preview
and repeat execution. Record this as GUI validation only when the real desktop
path has actually been performed.

The contributor's Windows desktop campaign imported both supplied pairs,
reviewed the settings, ran F5 and inspected the native convergence log. Each
case was saved as `.veragrid`, reopened through the native file picker, and
its saved power-flow result was loaded from the Results context menu. The
MS case retained all 3,056 AC branches (including 32 series capacitors); the
SIN case retained all 14,134 AC branches (including 126 series capacitors)
and 14 DC poles. Its 11 active poles have solved terminal powers and losses
but no registered operative transfer limits, as described above. This is
functional desktop evidence, not a performance benchmark or proof of
reference-engine equivalence.
