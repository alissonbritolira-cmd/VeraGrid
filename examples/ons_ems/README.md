# ONS EMS state-estimator STM/PRM examples

These are the exact STM and PRM contents supplied by the contributor, renamed
without changing their bytes:

| Files | Source coverage | STM SHA256 |
| --- | --- | --- |
| `SIN Brazil Example.stm` + `SIN Brazil Example.prm` | SP/SIN case; 10,614 buses | `0a73caeee066a5dad5e02280966eebe1b193bf6319276930185bcd9146fef6dc` |
| `Soulth SIN Brazil Example.stm` + `Soulth SIN Brazil Example.prm` | MS case; 1,958 buses | `16d42595770aeba6b084cb069da4f09f1f0b6e5f7a15a10647c8370156e29a81` |

The filename spelling `Soulth` is intentional, as supplied by the contributor.
These are ONS EMS state-estimator source files. Renaming them is not an assertion
that they have been converted or numerically validated in VeraGrid.

**Current support:** these real cases are read-only source inspection and parser
coverage examples. This first contribution does not qualify their complete
electrical conversion. Opening either pair through File > Open shows all source
sections, PRM records and the reasons requiring further adapters; it keeps the
current circuit unchanged. No equipment is removed to create a misleading
simulatable subset.

The MS file contains 32 series-capacitor records, 12 DC records, 177 switched
shunt banks, 9 SVCs, 578 automatically controlled transformers and 10 remotely
controlled generators. The SIN file contains 126 series-capacitor records,
42 DC records, 796 switched shunt banks, 45 SVCs, 2,014 automatically controlled
transformers and 43 remotely controlled generators. DC figures are source record
counts, not necessarily complete physical-link counts.

The PRMs are retained for inspection; their commands are not executed. No DYN,
DSA, executable or installation is included. The small synthetic AC example at
`src/tests/FileFormats/data/ons_ems` demonstrates the supported executable
conversion instead.

These data files retain their existing ownership and any applicable data rights.
Their inclusion does not apply a new software license to them.
