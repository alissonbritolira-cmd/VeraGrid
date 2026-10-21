# ONS EMS state-estimator STM/PRM examples

These are the exact STM and PRM contents supplied by the contributor, renamed
without changing their bytes:

| Files | Source coverage | STM SHA256 |
| --- | --- | --- |
| `SIN Brazil Example.stm` + `SIN Brazil Example.prm` | SP/SIN case; 10,614 buses, 14,134 AC branches | `0a73caeee066a5dad5e02280966eebe1b193bf6319276930185bcd9146fef6dc` |
| `Soulth SIN Brazil Example.stm` + `Soulth SIN Brazil Example.prm` | MS case; 1,958 buses, 3,056 AC branches | `16d42595770aeba6b084cb069da4f09f1f0b6e5f7a15a10647c8370156e29a81` |

The spelling `Soulth` is intentional, as supplied by the contributor. These
are ONS EMS state-estimator files. The `.gitattributes` in this directory keeps
Git from changing their original line endings or whitespace.

Open one pair through **File > Open**, confirm 60 Hz and review the selected
PRM. The complete network imports into the normal project. Review **Settings >
Power flow**, then press **F5**. Taps, phase shifters and discrete bank switching
start blocked on a fresh import. **Apply PRM options** provides a preview for
explicit changes. **File > ONS EMS import data** shows the complete source and
the effective settings. Save as `.veragrid` to retain that information.

The MS source includes 32 series capacitors, 177 switched shunt banks, 9 SVCs,
578 automatically controlled transformers and 10 remotely controlled
generators. Its 12 DC records describe 4 inactive poles. The SIN source includes
126 series capacitors, 796 banks, 45 SVCs, 2,014 automatically controlled
transformers and 43 remotely controlled generators. Its 42 DC records describe
14 poles, of which 11 are active. Source records and physical pole counts are
different quantities.

The import and PF reuse the developed electrical adapters, including converter
AC coupling and isolated-reference handling; no equipment is removed to produce
a simplified demonstration. The source-free isolated-reference warnings concern
the private PF state and retain the original equipment and load values.

Convergence is not a certificate of equivalence with another application. In
particular, the preserved LCC/CCC qualification metadata still identifies the
experimental external-equivalence status. Requested commands without qualified
execution are preserved and diagnosed. For example, an explicit GUI reactive
limit choice overrides an unqualified intermediate `PFGENVARIT` delay and is
identified as an override. See [import details](../../doc/md_source/ons_ems_stm.md).

PRM SHA256 values:

* MS: `8e3131b55f95a55573247a0a60c076a59ed31537a60e042d79e4b6396adb9e8f`.
* SIN: `a2e69b3862b5776ba05f441e9cceed9ecf5facfd93c757bf8d3aa408c3272d72`.

No DYN, DSA, executable or external installation is included. These data files
retain their existing ownership and any applicable data rights. Their inclusion
does not apply a new software license to them.

Both pairs were also opened, solved with F5, saved and reopened through the
Windows GUI. The SIN example does not supply operative DC transfer limits:
its active-pole powers are solved, while operative loading remains unavailable
and is diagnosed as `missing_limit`. The equipment nominal rating is not
silently substituted for that missing limit.
