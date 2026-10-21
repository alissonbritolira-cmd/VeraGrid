# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Reutiliza ramos fixos dentro de um ajuste de controles, sem guardar soluções AC."""
from __future__ import annotations

import copy
import time
from enum import Enum
import numpy as np
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.Compilers import circuit_to_data as compiler
from VeraGridEngine.DataStructures.bus_data import BusData
from VeraGridEngine.DataStructures.generator_data import GeneratorData
from VeraGridEngine.DataStructures.shunt_data import ShuntData
from VeraGridEngine.DataStructures.load_data import LoadData
from VeraGridEngine.Simulations.PowerFlow.power_flow_worker import multi_island_pf_nc
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver


def _freeze(value):
    if isinstance(value, np.ndarray):
        return value.shape, value.dtype.str, value.tobytes()
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, complex, bool, Enum, np.generic)):
        return value
    return id(value)


class ControlSolver:
    """Cache restrito ao cenário privado. Geradores e shunts são recompilados.

    Não compartilha matrizes entre processos ou estudos. Taps automáticos,
    baterias, conversores, fluidos e redução topológica usam o driver completo.
    Cargas fixas são reutilizadas somente dentro do ajuste de controles.
    A Ybus e a Jacobiana continuam sendo montadas pelo solver nativo.
    """
    def __init__(self, grid, options, telemetry=None, fixed_network=False, warm_start=False,
                 compilation_cache=None):
        self.grid, self.options, self.telemetry = grid, options, telemetry
        self.template = None
        self.key = None
        self.info = None
        self.fixed_network = fixed_network
        self.raw_bus = None
        self.last_nc = None
        self.last_voltage = None
        self.warm_start = warm_start
        self.load_bus = None
        # Only numerical structure is retained by the caller, never this
        # solver, its model, voltage history or PowerFlowResults.
        self.compilation_cache = compilation_cache

    def signature(self):
        branches = self.grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=True)
        # Todas as propriedades escalares e referências dos ramos, inclusive
        # ratings, controles e impedâncias. Objetos de modelo que não são usados
        # pelo fluxo positivo não são transformados em texto a cada consulta.
        fields = ('name', 'idtag', 'active', 'reducible', 'R', 'X', 'G', 'B',
                  'R0', 'X0', 'G0', 'B0', 'R2', 'X2', 'G2', 'B2',
                  'rate', 'contingency_factor', 'protection_rating_factor', 'monitor_loading',
                  'contingency_enabled', 'mttf', 'mttr', 'Cost',
                  'tap_module', 'tap_phase', 'tap_module_control_mode', 'tap_phase_control_mode',
                  'tap_module_min', 'tap_module_max', 'tap_phase_min', 'tap_phase_max',
                  'Pset', 'Qset', 'vset', 'temp_base', 'temp_oper', 'alpha', 'tolerance',
                  'virtual_tap_f', 'virtual_tap_t', 'winding_type', 'conn', 'conn_f', 'conn_t', 'HV', 'LV', 'is_dc')
        bus_fields = ('name', 'idtag', 'Vnom', 'active', 'is_slack', 'is_grounded', 'is_dc',
                      'Vmin', 'Vmax', 'Vm_cost', 'angle_min', 'angle_max')
        hvdc_fields = ('name', 'idtag', 'active', 'dispatchable', 'rate', 'contingency_factor',
                       'protection_rating_factor', 'angle_droop', 'r', 'control_mode', 'Pset',
                       'Vset_f', 'Vset_t', 'firing_angle_min_f', 'firing_angle_max_f',
                       'firing_angle_min_t', 'firing_angle_max_t')
        # Vm0/Va0 and every injection are rebuilt for a new scenario. They
        # do not determine whether the passive branch structure is reusable.
        return (float(self.grid.Sbase), float(self.grid.fBase),
                tuple((id(b), type(b), id(b.area), id(b.substation),
                       *(_freeze(getattr(b, name, None)) for name in bus_fields)) for b in self.grid.buses),
                tuple((id(a), a.name) for a in self.grid.areas), tuple(id(s) for s in self.grid.substations),
                tuple(tuple((id(d), type(d), id(d.bus), id(getattr(d, 'control_bus', None)))
                            for d in devices) for devices in (self.grid.generators,
                            self.grid.get_load_like_devices(), self.grid.get_shunt_like_devices())),
                tuple((id(b), type(b), id(b.bus_from), id(b.bus_to), id(getattr(b, 'regulation_bus', None)),
                       id(getattr(b, 'regulation_branch', None)),
                       id(getattr(b, 'template', None)),
                       tuple(_freeze(getattr(b.tap_changer, name)) for name in b.tap_changer.__slots__)
                           if hasattr(b, 'tap_changer') else (),
                       *(_freeze(getattr(b, name, None)) for name in fields)) for b in branches),
                tuple((id(h), id(h.bus_from), id(h.bus_to),
                       *(_freeze(getattr(h, name, None)) for name in hvdc_fields)) for h in self.grid.hvdc_lines),
                tuple((name, _freeze(getattr(self.options, name))) for name in self.options.__slots__))

    def _supports_scenario_cache(self):
        grid, options = self.grid, self.options
        if (not self.fixed_network or options.initialize_angles or grid.external_grids or
                any(b.is_dc for b in grid.buses) or
                any(record.get('active') for record in grid.import_metadata.get('lcc_records', []))):
            return False
        allowed = {'Bus', 'Line', 'Transformer2W', 'SeriesReactance', 'Generator', 'Load',
                   'StaticGenerator', 'CurrentInjection', 'Shunt', 'ControllableShunt'}
        branches = grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=True)
        devices = [*grid.buses, *branches, *grid.generators,
                   *grid.get_load_like_devices(), *grid.get_shunt_like_devices()]
        return (all(type(device).__name__ in allowed for device in devices) and
                not any(branch.reducible or (branch.R == 0. and branch.X == 0.) for branch in branches))

    def _refresh_scenario_inputs(self, nc, logger):
        """Rebuild all injections and bus state before a different dispatch."""
        grid, options = self.grid, self.options
        nc.bus_data = BusData(nbus=nc.nbus)
        nc.generator_data = GeneratorData(nelm=nc.ngen, nbus=nc.nbus)
        nc.shunt_data = ShuntData(nelm=nc.nshunt, nbus=nc.nbus)
        nc.load_data = LoadData(nelm=nc.nload, nbus=nc.nbus)
        buses = {bus: i for i, bus in enumerate(grid.buses)}
        used = np.zeros(nc.nbus, dtype=bool)
        compiler.get_bus_data(nc.bus_data, grid, {area: i for i, area in enumerate(grid.areas)}, None,
                              use_stored_guess=options.use_stored_guess, logger=logger)
        compiler.get_generator_data(nc.generator_data, grid, buses, used, logger, nc.bus_data,
                                    t_idx=None, use_stored_guess=options.use_stored_guess,
                                    control_remote_voltage=options.control_remote_voltage)
        compiler.get_shunt_data(nc.shunt_data, grid, buses, used, nc.bus_data, None, logger,
                                use_stored_guess=options.use_stored_guess,
                                control_remote_voltage=options.control_remote_voltage)
        compiler.get_load_data(nc.load_data, grid, buses, used, nc.bus_data, None, logger,
                               use_stored_guess=options.use_stored_guess)
        nc.init_idtags_dict()

    def run(self):
        grid, options = self.grid, self.options
        excluded = (options.control_taps_modules or options.control_taps_phase or grid.batteries or
                    any(h.active for h in grid.hvdc_lines) or grid.vsc_devices or grid.upfc_devices or grid.fluid_nodes or
                    grid.transformers3w or grid.switch_devices or not options.use_stored_guess)
        if excluded:
            if self.compilation_cache is not None:
                self.compilation_cache.clear()
            self.template = self.key = None
            self.last_voltage = None
            self.last_nc = None
            driver = PowerFlowDriver(grid, options, import_adapter_internal=True)
            driver.run()
            return driver.results
        started = time.perf_counter()
        # Dentro do ajuste privado, somente Q, modo dos geradores e B dos
        # controles variam. A fotografia dos ramos e das barras é imutável.
        # Fora desse escopo, conferir a assinatura a cada chamada.
        if self.fixed_network and self.template is not None:
            key = self.key
        elif self.fixed_network and self.compilation_cache is None:
            # Este ajuste nasce numa rede privada e recompila integralmente a
            # primeira AC. Não compara estruturas de cenários distintos: uma
            # chave local basta para o mesmo contrato imutável já usado abaixo.
            key = object()
        else:
            key = self.signature()
        logger = Logger()
        if self.template is None or self.key != key:
            cache = self.compilation_cache
            eligible = cache is not None and self._supports_scenario_cache()
            if cache is not None and not eligible:
                cache.clear()
            reusable = eligible and cache.get('key') == key and cache.get('template') is not None
            if reusable:
                nc = cache['template'].copy()
                self._refresh_scenario_inputs(nc, logger)
                if self.telemetry is not None:
                    self.telemetry['scenario_compilation_reuses'] = self.telemetry.get('scenario_compilation_reuses', 0)+1
            else:
                if cache and self.telemetry is not None:
                    self.telemetry['scenario_compilation_invalidations'] = self.telemetry.get('scenario_compilation_invalidations', 0)+1
                nc = compiler.compile_numerical_circuit_at(circuit=grid,
                    apply_temperature=options.apply_temperature_correction,
                    branch_tolerance_mode=options.branch_impedance_tolerance_mode,
                    use_stored_guess=options.use_stored_guess,
                    control_taps_modules=options.control_taps_modules, control_taps_phase=options.control_taps_phase,
                    control_remote_voltage=options.control_remote_voltage, logger=logger)
                if self.telemetry is not None:
                    self.telemetry['full_compilations'] = self.telemetry.get('full_compilations', 0)+1
            # This template belongs only to the current private adjustment.
            # A persistent template is a separate copy, never passed to Newton
            # or shared by the shallow copies used for inner control steps.
            self.template, self.key = nc, key
            if eligible and not reusable:
                cache.clear()
                if not np.any(nc.passive_branch_data.reducible) and not nc.active_branch_data.any_pf_control:
                    persisted = nc.copy()
                    persisted.init_idtags_dict()
                    cache.update(key=key, template=persisted)
            self.info = grid.get_branch_areas_info()
            self.raw_bus = BusData(nbus=nc.nbus)
            compiler.get_bus_data(self.raw_bus, grid, {area: i for i, area in enumerate(grid.areas)}, None,
                                  use_stored_guess=options.use_stored_guess, logger=logger)
            self.load_bus = None
            if self.fixed_network and not grid.external_grids and not np.any(nc.passive_branch_data.reducible):
                # As cargas ZIP e suas contribuições fixas não mudam enquanto
                # este ajuste altera somente Q/modo dos geradores e B dos SVC.
                self.load_bus = self.raw_bus.copy()
                compiler.get_load_data(data=LoadData(nelm=nc.nload, nbus=nc.nbus), circuit=grid,
                    bus_dict={bus: i for i, bus in enumerate(grid.buses)},
                    bus_voltage_used=np.zeros(nc.nbus, dtype=bool), bus_data=self.load_bus,
                    t_idx=None, logger=logger, use_stored_guess=options.use_stored_guess)
        else:
            # No topology reduction or automatic tap can reach this path.
            # Native Newton works on private island slices. Only fixed branch
            # and load data are shared inside this adjustment, as before.
            nc = copy.copy(self.template)
            nc.bus_data = (self.load_bus or self.raw_bus).copy() if self.fixed_network else BusData(nbus=nc.nbus)
            nc.generator_data = GeneratorData(nelm=nc.ngen, nbus=nc.nbus)
            nc.shunt_data = ShuntData(nelm=nc.nshunt, nbus=nc.nbus)
            nc.load_data = self.template.load_data if self.load_bus is not None else LoadData(nelm=nc.nload, nbus=nc.nbus)
            buses = {bus: i for i, bus in enumerate(grid.buses)}
            used = np.zeros(nc.nbus, dtype=bool)
            if not self.fixed_network:
                compiler.get_bus_data(nc.bus_data, grid, {area: i for i, area in enumerate(grid.areas)}, None,
                                      use_stored_guess=options.use_stored_guess, logger=logger)
            compiler.get_generator_data(nc.generator_data, grid, buses, used, logger, nc.bus_data,
                                        t_idx=None, use_stored_guess=options.use_stored_guess,
                                        control_remote_voltage=options.control_remote_voltage)
            compiler.get_shunt_data(nc.shunt_data, grid, buses, used, nc.bus_data, None, logger,
                                    use_stored_guess=options.use_stored_guess,
                                    control_remote_voltage=options.control_remote_voltage)
            if self.load_bus is None:
                compiler.get_load_data(data=nc.load_data, circuit=grid, bus_dict=buses, bus_voltage_used=used,
                                       bus_data=nc.bus_data, t_idx=None, logger=logger,
                                       use_stored_guess=options.use_stored_guess)
            elif self.telemetry is not None:
                self.telemetry['load_compilation_reuses'] = self.telemetry.get('load_compilation_reuses', 0)+1
            if self.telemetry is not None:
                self.telemetry['branch_compilation_reuses'] = self.telemetry.get('branch_compilation_reuses', 0)+1
            nc.init_idtags_dict()
        if np.any(nc.passive_branch_data.reducible):
            # A redução modifica conectividade e índices; não reutilizá-los.
            self.template = None
        if self.telemetry is not None:
            self.telemetry['compilation_s'] = self.telemetry.get('compilation_s', 0)+time.perf_counter()-started
        if self.warm_start and self.last_voltage is not None:
            # Mantém os módulos especificados nas barras PV e de referência.
            # A tensão anterior pertence apenas a este ajuste privado.
            fixed = nc.bus_data.bus_types != 1
            guess = self.last_voltage.copy()
            guess[fixed] = abs(nc.bus_data.Vbus[fixed])*np.exp(1j*np.angle(guess[fixed]))
            nc.bus_data.Vbus[:] = guess
            if self.telemetry is not None:
                self.telemetry['inner_voltage_reuses'] = self.telemetry.get('inner_voltage_reuses', 0)+1
        result = multi_island_pf_nc(nc=nc, options=options, logger=logger)
        self.last_nc = nc
        self.last_voltage = result.voltage.copy() if result.converged and np.isfinite(result.voltage).all() else None
        result.area_names, result.bus_area_indices, result.F, result.T, result.hvdc_F, result.hvdc_T = self.info
        return result
