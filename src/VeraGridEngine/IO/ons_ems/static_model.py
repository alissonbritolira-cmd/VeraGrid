# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Cópia privada para estudos AC instantâneos; não serve para simulação dinâmica."""
from __future__ import annotations

import copy
from VeraGridEngine.Devices.multi_circuit import MultiCircuit


def copy_static_model(grid):
    """Preserva equipamentos, perfis, referências e tabelas de taps da rede.

    Diagramas e blocos simbólicos RMS/EMT não participam do fluxo estático.
    São omitidos, em vez de compartilhados com a rede aberta pelo usuário.
    Um único memo mantém os apontadores entre os equipamentos copiados.
    A cópia geral do VeraGrid permanece disponível para estudos dinâmicos.
    """
    result = MultiCircuit(name=grid.name, Sbase=grid.Sbase, fbase=grid.fBase, idtag=grid.idtag)
    fields = ('branch_groups', 'lines', 'dc_lines', 'transformers2w', 'hvdc_lines',
              'vsc_devices', 'upfc_devices', 'switch_devices', 'transformers3w',
              'windings', 'series_reactances', 'buses', 'loads', 'generators',
              'external_grids', 'shunts', 'batteries', 'static_generators',
              'current_injections', 'controllable_shunts', 'bus_bars',
              'overhead_line_types', 'wire_types', 'underground_cable_constructions',
              'underground_cable_types', 'dc_cable_types', 'sequence_line_types',
              'transformer_types', 'substations', 'voltage_levels', 'areas', 'zones',
              'countries', 'communities', 'regions', 'municipalities', 'time_profile',
              'contingencies', 'contingency_groups', 'technologies', 'fuels',
              'emission_gases', 'fluid_nodes', 'fluid_paths', 'modelling_authorities')
    memo = {id(grid): result, id(grid.var_factory): result.var_factory}
    # Impede também que um apontador de equipamento traga modelos excluídos.
    excluded = ('_rms_model', '_emt_model', '_rms_template', '_emt_template',
                '_rms_fmu_template', '_emt_fmu_template', '_grid_synchronized_emt_model')
    profiles = []
    devices = list(grid.get_all_elements_iter())
    for device in devices:
        for name in excluded:
            value = getattr(device, name, None)
            if value is not None:
                memo[id(value)] = None
    for device in devices:
        for name in device.properties_with_profile.values():
            profile = getattr(device, name)
            if id(profile) in memo:
                continue
            private = copy.copy(profile)
            memo[id(profile)] = private
            profiles.append((profile, private))
            # Os arrays numéricos são copiados diretamente pelo NumPy. Perfis
            # de apontadores aguardam o memo completo dos equipamentos.
            array = getattr(profile, '_dense_array', None)
            if array is not None and array.dtype != object:
                private._dense_array = array.copy()
            private._sparse_array = copy.deepcopy(profile._sparse_array, memo)
    for name in fields:
        setattr(result, name, copy.deepcopy(getattr(grid, name), memo))
    for source, private in profiles:
        private._default_value = copy.deepcopy(source._default_value, memo)
        array = getattr(source, '_dense_array', None)
        if array is not None and array.dtype == object:
            private._dense_array = copy.deepcopy(array, memo)
    result.import_metadata = copy.deepcopy(grid.import_metadata)
    return result
