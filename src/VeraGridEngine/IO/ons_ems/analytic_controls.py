# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Sensibilidades dos controles remotos por diferenciação das equações AC."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.Simulations.Derivatives.ac_jacobian import create_J_vc_csc
from VeraGridEngine.Utils.Sparse.csc2 import mat_to_scipy


def distribute_q(control, value):
    """Participação de Q com saturação individual, usada no ajuste e na derivada."""
    devices = control['devices']
    lo = np.array([g.Qmin for g in devices]); hi = np.array([g.Qmax for g in devices])
    weights = np.asarray(control['weights'])
    q = np.clip(value*weights, lo, hi)
    for _ in range(len(q)+1):
        remainder = value-q.sum()
        if abs(remainder) < 1e-9:
            break
        free = (q < hi-1e-9) if remainder > 0 else (q > lo+1e-9)
        if not free.any():
            break
        share = weights*free
        if share.sum() <= 0:
            share = free.astype(float)
        q = np.clip(q+remainder*share/share.sum(), lo, hi)
    return q


def control_jacobian(solver, result, controls, free):
    """Fatora uma Jacobiana por ilha e resolve todos os controles juntos.

    Inclui a derivada ZIP e a injeção V² dos SVC. Não reutiliza fatorações
    após mudanças de tensão, topologia ou conjunto PV/PQ. Casos fora do
    domínio validado voltam às diferenças finitas no chamador.
    """
    nc = solver.last_nc
    opts = solver.options
    if nc is None or opts.distributed_slack:
        raise ValueError('Formulação não suportada pelas sensibilidades analíticas.')
    if np.any(nc.passive_branch_data.reducible):
        raise ValueError('Redução topológica requer diferenças finitas.')
    grid = solver.grid
    buses = {bus: i for i, bus in enumerate(grid.buses)}
    lookup = {str(bus.code): i for i, bus in enumerate(grid.buses)}
    injection = np.zeros((nc.nbus, len(free)))
    observed = [lookup[str(controls[k]['bus'])] for k in free]
    for col, k in enumerate(free):
        c = controls[k]
        if c['kind'] == 'svc':
            bus = buses[c['device'].bus]
            injection[bus, col] += abs(result.voltage[bus])**2 / nc.Sbase
        else:
            # Diferencia somente a participação algébrica, com o mesmo lado
            # da perturbação do método de referência. Não executa um fluxo.
            value = c['value']
            step = max(.05, abs(value)*1e-4)
            if value+step > c['max']:
                step = -step
            changed = float(np.clip(value+step, c['min'], c['max']))
            if changed == value:
                raise ValueError('Controle sem faixa de perturbação.')
            slope = (distribute_q(c, changed)-np.array([g.Q for g in c['devices']]))/(changed-value)
            for g, weight in zip(c['devices'], slope):
                injection[buses[g.bus], col] += weight / nc.Sbase
    response = np.zeros_like(injection)
    generation_response = np.zeros_like(injection)
    needs_generation_response = any(c.get('local_anchor_codes') for c in controls)
    for island in nc.split_into_islands(ignore_single_node_islands=opts.ignore_single_node_islands,
                                       consider_hvdc_as_island_links=False, logger=Logger()):
        original = island.bus_data.original_idx
        indices = island.get_simulation_indices(Sbus=island.get_power_injections_pu())
        if len(indices.pqv) or len(indices.p):
            raise ValueError('Controle remoto nativo requer diferenças finitas.')
        va = np.r_[indices.pv, indices.pq, indices.pqv, indices.p].astype(np.int32)
        vm = np.r_[indices.pq, indices.p].astype(np.int32)
        q = np.r_[indices.pq, indices.pqv].astype(np.int32)
        if not len(indices.vd):
            raise ValueError('Ilha sem referência angular.')
        if not len(vm) or not injection[original].any():
            continue
        voltage = result.voltage[original]
        adm = island.get_admittance_matrices().Ybus.tocsc()
        J = mat_to_scipy(create_J_vc_csc(len(original), adm.data, adm.indptr, adm.indices,
                                        voltage, va, vm, va, q))
        # F = S(V) - [S0 + conj(I0 + Y0 |V|) |V|].
        zip_derivative = np.conj(island.get_current_injections_pu() +
                                 2*island.get_admittance_injections_pu()*abs(voltage))
        cols = {int(bus): i+len(va) for i, bus in enumerate(vm)}
        rows, columns, data = [], [], []
        for offset, selected, component in ((0, va, zip_derivative.real),
                                             (len(va), q, zip_derivative.imag)):
            for row, bus in enumerate(selected):
                if int(bus) in cols and component[bus] != 0:
                    rows.append(offset+row); columns.append(cols[int(bus)]); data.append(component[bus])
        if data:
            J = J-csc_matrix((data, (rows, columns)), shape=J.shape)
        rhs = np.vstack((np.zeros((len(va), len(free))), injection[original][q]))
        solved = splu(J.tocsc()).solve(rhs)
        response[original[vm]] = solved[len(va):]
        if needs_generation_response:
            d_voltage = np.zeros((len(original), len(free)), dtype=complex)
            d_voltage[va] += 1j*voltage[va,None]*solved[:len(va)]
            d_voltage[vm] += voltage[vm,None]/abs(voltage[vm,None])*solved[len(va):]
            d_power = (d_voltage*np.conj(adm@voltage)[:,None] +
                       voltage[:,None]*np.conj(adm@d_voltage))
            generation_response[original] = (d_power.imag-
                zip_derivative.imag[:,None]*response[original]-injection[original])*nc.Sbase
    matrix = response[observed]
    from VeraGridEngine.IO.ons_ems.control_equations import ControlEquations
    matrix = ControlEquations(grid, controls).jacobian(matrix, response, generation_response, result, free)
    if not np.isfinite(matrix).all():
        raise ValueError('Sensibilidade não finita.')
    return matrix
