# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
import numpy as np
def ac_branches(grid):
    return grid.get_branches(add_hvdc=False, add_vsc=False, add_switch=False)


def normalized_flows(grid, results, metadata):
    """Inclui shunts terminais e o equivalente de linha aberta na comparação."""
    branches = ac_branches(grid)
    if len(results.Sf) != len(branches) or len(results.St) != len(branches):
        raise ValueError("Ordem ou quantidade de ramos incompatível com o resultado CA.")
    sf, st = np.array(results.Sf, complex).copy(), np.array(results.St, complex).copy()
    record_data = metadata.get("branch_records", []) if metadata else []
    if isinstance(record_data, dict):
        records = record_data
    else:
        records = {str(record["key"]): record for record in record_data}
    voltage = {str(bus.code): complex(v) for bus, v in zip(grid.buses, results.voltage)}
    for index, branch in enumerate(branches):
        record = records.get(str(branch.code))
        if not record:
            continue
        vf = abs(voltage[str(branch.bus_from.code)])
        vt = abs(voltage[str(branch.bus_to.code)])
        for side, vm in (("f", vf), ("t", vt)):
            if record.get("sht"+side) and record.get("s"+side):
                power = complex(record.get("g"+side, 0.), -record.get("b"+side, 0.))*grid.Sbase*vm**2
                if side == "f":
                    sf[index] += power
                else:
                    st[index] += power
        if record.get("kind") == "line" and bool(record.get("sf")) != bool(record.get("st")):
            ys = 1/complex(record["r"], record["x"])
            yc = 1j*float(record["b"])/2
            yin = yc+ys*yc/(ys+yc)
            if record["sf"]:
                sf[index] += np.conj(yin)*grid.Sbase*vf**2
            else:
                st[index] += np.conj(yin)*grid.Sbase*vt**2
    return sf, st, records
