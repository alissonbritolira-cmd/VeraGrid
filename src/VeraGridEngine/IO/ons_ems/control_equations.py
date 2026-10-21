# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Physical allocation equations accompanying remote voltage control.

Units: V in pu, B in MVAr at 1 pu, Q in MVAr, allocation residual in Q/Sbase.
Production8.6.2 GUI probes R57 distinguish remote SVC equal-Q sharing from
local SVC precedence, and remote-generator participation against a local PV.
No controller, limit or electrical contribution is removed by this solver.
"""
import numpy as np
from VeraGridEngine.enumerations import GeneratorControlMode


def equal_q(total, lower, upper):
    """Equal outputs, redistributing only over devices with available range."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    assigned = np.zeros(len(lower))
    free = np.ones(len(lower), dtype=bool)
    for _ in range(len(lower)+1):
        if not free.any():
            break
        level = (total-assigned[~free].sum())/free.sum()
        low, high = free & (level < lower), free & (level > upper)
        if not (low | high).any():
            assigned[free] = level
            break
        assigned[low], assigned[high] = lower[low], upper[high]
        free[low | high] = False
    return assigned, free


class ControlEquations:
    def __init__(self, grid, controls):
        self.grid, self.controls = grid, controls
        self.buses = {b.code: i for i, b in enumerate(grid.buses)}
        self.gen_indices = {g.code: i for i, g in enumerate(grid.generators)}
        self.gen_by_code = {g.code: g for g in grid.generators}
        grouped = {}
        self.anchors = {}
        for k, c in enumerate(controls):
            if c.get('svc_sharing') == 'equal_reactive_power':
                grouped.setdefault(c['bus'], []).append(k)
            if c.get('local_anchor_codes'):
                self.anchors[k] = c['local_anchor_codes']
        self.groups = [members for members in grouped.values() if len(members) > 1]

    def anchor_active(self, k):
        return any(self.gen_by_code[code].control_mode == GeneratorControlMode.V
                   for code in self.anchors[k])

    def group_state(self, result, members):
        cs = [self.controls[k] for k in members]
        physical = [self.buses[c['device'].bus.code] for c in cs]
        vm = abs(result.voltage[physical])
        b = np.array([c['value'] for c in cs])
        lo, hi = np.array([c['min'] for c in cs]), np.array([c['max'] for c in cs])
        q = b*vm**2
        allocated, available = equal_q(q.sum(), lo*vm**2, hi*vm**2)
        return physical, vm, b, lo, hi, q, allocated, available

    def residual(self, result):
        error = np.array([abs(result.voltage[self.buses[c['bus']]])-c['target']
                          for c in self.controls])
        for k, codes in self.anchors.items():
            if self.anchor_active(k):
                c = self.controls[k]
                local_q = sum(result.gen_q[self.gen_indices[code]] for code in codes)
                error[k] = (c['value']-c['local_anchor_factor']*local_q)/self.grid.Sbase
        for members in self.groups:
            *_, q, allocated, available = self.group_state(result, members)
            for j, k in enumerate(members[1:], 1):
                error[k] = (q[j]-allocated[j])/self.grid.Sbase
        return error

    def limited(self, result, error):
        limited = np.array([(c['value'] <= c['min']+1e-7 and e > 0) or
                            (c['value'] >= c['max']-1e-7 and e < 0)
                            for c, e in zip(self.controls, error)])
        for members in self.groups:
            c = self.controls[members[0]]
            verr = abs(result.voltage[self.buses[c['bus']]])-c['target']
            all_limited = all((self.controls[k]['value'] <= self.controls[k]['min']+1e-7 and verr > 0) or
                              (self.controls[k]['value'] >= self.controls[k]['max']-1e-7 and verr < 0)
                              for k in members)
            limited[members] = all_limited
        return limited

    def jacobian(self, matrix, voltage_response, generation_response, result, free):
        rows = {int(k): i for i, k in enumerate(free)}
        for k, codes in self.anchors.items():
            if k in rows and self.anchor_active(k):
                c = self.controls[k]
                buses = {self.buses[self.gen_by_code[code].bus.code] for code in codes}
                # Q at a local PV is the balancing generation of that bus.
                derivative = -c['local_anchor_factor']*generation_response[list(buses)].sum(axis=0)
                derivative[rows[k]] += 1.
                matrix[rows[k]] = derivative/self.grid.Sbase
        for members in self.groups:
            physical, vm, b, lo, hi, q, allocated, available = self.group_state(result, members)
            dq = 2*b[:, None]*vm[:, None]*voltage_response[physical]
            for j, k in enumerate(members):
                if k in rows:
                    dq[j, rows[k]] += vm[j]**2
            da = np.zeros_like(dq)
            for j in np.where(~available)[0]:
                bound = lo[j] if allocated[j] <= lo[j]*vm[j]**2 else hi[j]
                da[j] = 2*bound*vm[j]*voltage_response[physical[j]]
            if available.any():
                da[available] = (dq.sum(axis=0)-da[~available].sum(axis=0))/available.sum()
            for j, k in enumerate(members[1:], 1):
                if k in rows:
                    matrix[rows[k]] = (dq[j]-da[j])/self.grid.Sbase
        return matrix

    def records(self, result, error, limited):
        allocated_rows = {k for m in self.groups for k in m[1:]}
        records = []
        for k, (c, e, bound) in enumerate(zip(self.controls, error, limited)):
            allocation = k in allocated_rows or (k in self.anchors and self.anchor_active(k))
            records.append(dict(name=c['name'], value=float(c['value']), error_pu=float(e),
                limited=bool(bound), equation='reactive_allocation' if allocation else 'voltage',
                voltage_error_pu=float(abs(result.voltage[self.buses[c['bus']]])-c['target']),
                allocation_error_mvar=float(e*self.grid.Sbase) if allocation else None))
        return records
