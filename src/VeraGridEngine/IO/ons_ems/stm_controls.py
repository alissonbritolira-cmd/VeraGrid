# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
import copy,math
import numpy as np
from VeraGridEngine.enumerations import GeneratorControlMode,ShuntControlMode
from VeraGridEngine.IO.ons_ems.stm_parser import bus_id
def prepare_baseline_references(grid):
    """Estado de execução das referências isoladas sem dispositivo de fonte.

    Aplicado apenas na preparação N-0, às barras marcadas SW no STM original.
    Não retira equipamentos nem zera cargas, e não se aplica a ilhas de N-1.
    """
    metadata = grid.import_metadata
    if metadata.get('format') != 'ONS EMS STM v5':
        return
    original = {bus_id(row[0]): int(row[3]) for row in metadata.get('source_records', {}).get('buses', [])}
    mapping = {str(row['original']): str(row['converted']) for row in metadata.get('association', {}).get('buses', [])}
    original = {mapping.get(code, code): kind for code, kind in original.items()}
    connected = {bus for branch in grid.get_branches() if branch.active and branch.bus_from.active and branch.bus_to.active
                 for bus in (branch.bus_from, branch.bus_to)}
    sources = [device for collection in (grid.generators, grid.static_generators, grid.batteries,
                                        grid.external_grids, grid.current_injections)
               for device in collection if device.active and device.bus is not None]
    source_buses = {device.bus for device in sources}
    records = metadata.setdefault('baseline_reference_states', [])
    for bus in grid.buses:
        if (bus.active and bus.is_slack and original.get(str(bus.code)) == 3
                and bus not in connected and bus not in source_buses):
            loads = [l for l in grid.loads if l.bus is bus and l.active]
            records.append({'bus_code': str(bus.code), 'original_active': True, 'original_reference': True,
                            'executed_energized': False, 'executed_reference': False,
                            'reason': 'Referência original isolada, sem conexão fechada nem dispositivo de fonte.',
                            'load_codes': [str(l.code) for l in loads],
                            'retained_p_mw': sum(l.P for l in loads), 'retained_q_mvar': sum(l.Q for l in loads)})
            bus.active = False
            bus.is_slack = False


def reactive_block_selection(grid):
    """Seleção PV→PQ, separada do bloqueio completo de uma unidade.

    O STM SAGE sem tradução não contém o marcador de seleção que a conversão
    nativa acrescenta. Nesse domínio a seleção inicial é vazia. Uma seleção
    explícita incorporada ao estudo prevalece; redes traduzidas exigem seu mapa.
    """
    selected = grid.import_metadata.get('reactive_block_selection')
    if selected is not None:
        if not isinstance(selected, dict) or selected.get('complete') is not True:
            raise ValueError('Seleção PV→PQ de reativo incompleta; informe as barras selecionadas.')
        codes = list(dict.fromkeys(map(str, selected.get('bus_codes', []))))
        buses = {str(bus.code) for bus in grid.buses}
        missing = set(codes) - buses
        if missing:
            raise ValueError('Barras da seleção PV→PQ ausentes: '+', '.join(sorted(missing)))
        return {**selected, 'bus_codes': codes}
    metadata = grid.import_metadata
    if (metadata.get('format') == 'ONS EMS STM v5' and
            metadata.get('association', {}).get('policy', 'sage') == 'sage' and
            'generators' in metadata.get('source_records', {}) and
            all(len(row) == 27 for row in metadata.get('source_records', {}).get('generators', []))):
        return {'complete': True, 'bus_codes': [],
                'source': 'STM SAGE sem tradução; registro de unidade sem marcador PV→PQ',
                'domain': 'Formato v5 de 27 campos; não infere seleção por nome ou tipo dinâmico.'}
    raise ValueError('Bloqueio de Q solicitado sem seleção PV→PQ incorporada. '
                     'Informe a seleção ou escolha explicitamente a regulação de Q neste estudo.')


def apply_reactive_block(grid, controls, phase, prefault_q=None):
    """Mantém o Q físico do N-0 atual, inclusive sem warm start numérico."""
    if not grid.import_metadata.get('parameters', {}).get(phase+'BLOCKQG', False):
        return controls
    selection = reactive_block_selection(grid)
    selected_buses = set(selection['bus_codes'])
    selected = [g for g in grid.generators if g.active and g.bus.active and not g.bus.is_slack
                and str(g.bus.code) in selected_buses]
    values = prefault_q if phase == 'CTG' else {str(g.code): float(g.Q) for g in selected}
    if selected and (values is None or any(str(g.code) not in values for g in selected)):
        raise ValueError('Bloqueio de Q sem geração reativa completa do N-0 deste ponto.')
    frozen = {str(g.code) for g in selected}
    for g in selected:
        value = float(values[str(g.code)])
        if not math.isfinite(value):
            raise ValueError('Q pré-falta não finito: '+str(g.code))
        g.Q = value
        g.control_mode = GeneratorControlMode.Q
    remaining = []
    for control in controls:
        if control['kind'] != 'generator':
            remaining.append(control)
            continue
        pairs = [(g, weight) for g, weight in zip(control['devices'], control['weights']) if str(g.code) not in frozen]
        if not pairs:
            continue
        c = dict(control)
        c['devices'], weights = map(list, zip(*pairs))
        weights = np.asarray(weights, dtype=float)
        if weights.sum() <= 0:
            raise ValueError('Participações remanescentes inválidas após bloqueio de Q: '+c['name'])
        c['weights'] = (weights/weights.sum()).tolist()
        c['value'] = sum(g.Q for g in c['devices'])
        c['min'] = sum(g.Qmin for g in c['devices'])
        c['max'] = sum(g.Qmax for g in c['devices'])
        remaining.append(c)
    grid.import_metadata['reactive_block_execution'] = {
        'phase': phase, 'complete': True, 'selection': selection,
        'generator_q_mvar': {str(g.code): float(g.Q) for g in selected},
        'initial_state': 'N-0 do ponto' if phase == 'CTG' else 'cadastro do caso',
    }
    return remaining


def prepare_control_grid(source, enable_remote=True, phase='PF', *, _consume_private=False):
    """Trabalha numa rede privada. O modelo aberto não recebe a solução.

    ``_consume_private`` transfere uma rede temporária já exclusiva do
    preparador. Os caminhos públicos F5/RMS continuam copiando a entrada.
    """
    from VeraGridEngine.IO.ons_ems.association import check_association_execution
    check_association_execution(source)
    from VeraGridEngine.IO.ons_ems.policy import check_terminal_auxiliaries
    check_terminal_auxiliaries(source)
    from VeraGridEngine.Devices.Injections.shunt import Shunt
    from VeraGridEngine.IO.ons_ems.static_model import copy_static_model
    grid=source if _consume_private else copy_static_model(source)
    prepare_baseline_references(grid)
    grid.import_metadata['pf_adapter_required']=False
    bases={r['code']:r['voltage_pu'] for r in source.import_metadata.get('original_load_models',[])}
    grid.import_metadata['execution_load_models']=[dict(code=l.code,voltage_pu=bases.get(l.code,l.bus.Vm0),
        **{k:getattr(l,k) for k in ('P','Q','Ir','Ii','G','B')}) for l in grid.loads]
    from VeraGridEngine.IO.ons_ems.policy import apply_load_model
    apply_load_model(grid,phase)
    from VeraGridEngine.IO.ons_ems.lcc import prepare_working_loads
    prepare_working_loads(grid)
    rows={f'{bus_id(r[0])}|{r[1]}':r for r in grid.import_metadata.get('source_records',{}).get('generators',[])}
    local_voltage = {}
    for gen in grid.generators:
        if (gen.active and gen.bus.active and gen.control_mode == GeneratorControlMode.V and
                (gen.control_bus is None or gen.control_bus is gen.bus)):
            local_voltage.setdefault(gen.bus.code, []).append(gen)
    groups={}
    controls=[]
    for gen in grid.generators:
        if gen.control_mode!=GeneratorControlMode.V or gen.control_bus is None or gen.control_bus is gen.bus:
            continue
        target=gen.control_bus
        gen.control_bus=None
        gen.control_mode=GeneratorControlMode.Q
        if gen.active and gen.bus.active and enable_remote:
            c=groups.setdefault(target.code,dict(name='Geração remota '+target.code,kind='generator',bus=target.code,
                target=gen.Vset,devices=[],weights=[],value=0.,min=0.,max=0.))
            if not math.isclose(c['target'],gen.Vset,abs_tol=1e-8):
                raise ValueError(f'Alvos diferentes para o controle da barra {target.code}.')
            c['devices'].append(gen)
            weight=float(rows[gen.code][13]) if gen.code in rows else float(gen.Snom)
            c['weights'].append(weight)
            c['value']+=gen.Q
            c['min']+=gen.Qmin
            c['max']+=gen.Qmax
    for shunt in grid.controllable_shunts:
        if shunt.control_mode not in (ShuntControlMode.Continuous,ShuntControlMode.Locked):
            raise ValueError('O adaptador STM atende apenas shunts contínuos ou shunts fixos.')
        fixed=Shunt(name=shunt.name,idtag=shunt.idtag,code=shunt.code,B=shunt.B,G=shunt.G,active=shunt.active)
        grid.add_shunt(shunt.bus,fixed)
        if shunt.control_mode == ShuntControlMode.Continuous and shunt.active and shunt.bus.active:
            target=shunt.control_bus or shunt.bus
            controls.append(dict(name=shunt.name,kind='svc',bus=target.code,target=shunt.Vset,
                value=shunt.B,min=shunt.Bmin,max=shunt.Bmax,device=fixed))
    grid.controllable_shunts.clear()
    shunt_groups = {}
    for c in controls:
        shunt_groups.setdefault(c['bus'], []).append(c)
    held = []
    for target, members in shunt_groups.items():
        if len(members) < 2:
            continue
        if any(not math.isclose(c['target'], members[0]['target'], abs_tol=1e-8) for c in members):
            raise ValueError(f'Alvos SVC diferentes para a mesma barra {target}; confira o cadastro.')
        local_svc = [c for c in members if c['device'].bus.code == target]
        if len(local_svc) > 1:
            raise ValueError(f'Mais de um SVC local na barra {target}; associação não unívoca.')
        if local_svc:
            for c in members:
                if c is not local_svc[0]:
                    held.append(dict(code=c['device'].code, bus=target, initial_b_mvar=c['value'],
                        effective='held_initial_admittance', reason='production862_local_svc_precedence'))
                    controls.remove(c)
        else:
            for c in members:
                c['svc_sharing'] = 'equal_reactive_power'
    for c in groups.values():
        weights=np.asarray(c['weights'])
        if not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
            raise ValueError(f'Participações inválidas na barra {c["bus"]}.')
        if c['bus'] in local_voltage:
            c['local_anchor_codes'] = [g.code for g in local_voltage[c['bus']]]
            c['local_anchor_factor'] = float(weights.sum()/100.)
        c['weights']=(weights/weights.sum()).tolist()
        controls.append(c)
    grid.import_metadata['remote_control_execution'] = dict(schema_version=1,
        contract='production862_shared_reactive_controls_v1', held=held,
        remote_svc_groups={target:[c['device'].code for c in members]
            for target,members in shunt_groups.items() if len(members)>1 and not any(c['device'].bus.code==target for c in members)},
        local_generator_anchors=[dict(bus=c['bus'], local=c['local_anchor_codes'],
            remote=[g.code for g in c['devices']], factor=c['local_anchor_factor'])
            for c in controls if c.get('local_anchor_codes')])
    return grid,controls


def solve_stm_controls(grid,options,logger=None,phase='PF',cancel_callback=None):
    from VeraGridEngine.IO.ons_ems.discrete_controls import solve_discrete
    from VeraGridEngine.IO.ons_ems.lcc import solve_coupled
    from VeraGridEngine.IO.ons_ems.policy import check_execution,check_low_voltage,ignored_single_buses,check_remote_policy
    imported=grid.import_metadata.get('format')=='ONS EMS STM v5'
    if imported:
        grid.import_metadata.pop('discrete_last_solution', None)
        grid.import_metadata.setdefault('effective_pf', {}).pop('control_failure', None)
    candidate,controls=prepare_control_grid(grid,enable_remote=options.control_remote_voltage,phase=phase)
    controls=apply_reactive_block(candidate,controls,phase,
                                 grid.import_metadata.get('region_prefault_q'))
    if imported:
        check_execution(candidate,options,phase)
        grid.import_metadata['effective_discrete'] = copy.deepcopy(candidate.import_metadata.get('effective_discrete', {}))
    settings=copy.deepcopy(options)
    solve=solve_discrete if imported else solve_coupled
    telemetry = {}
    solve_kwargs={'voltage_tolerance':min(options.tolerance,1e-7),'sensitivity_update':'analytic',
                  'telemetry':telemetry,
                  'cancel_callback':cancel_callback}
    if imported:solve_kwargs['phase']=phase
    try:
        result,trace=solve(candidate,settings,controls,**solve_kwargs)
    finally:
        if imported:
            grid.import_metadata['remote_control_execution'] = copy.deepcopy(candidate.import_metadata.get('remote_control_execution', {}))
        if telemetry.get('control_failure'):
            grid.import_metadata.setdefault('effective_pf', {})['control_failure'] = copy.deepcopy(telemetry['control_failure'])
            if logger is not None:
                import json
                logger.add_error('Diagnóstico dos controles de tensão',
                    value=json.dumps(telemetry['control_failure'], ensure_ascii=False))
    if imported:
        grid.import_metadata.setdefault('effective_pf',{})['precision'] = {
            key:copy.deepcopy(telemetry[key]) for key in (
                'pf_requested_tolerance_pu','pf_effective_tolerance_pu',
                'control_requested_tolerance_pu','control_effective_tolerance_pu',
                'precision_refinement_reason','numerical_polish_pu') if key in telemetry}
    check_low_voltage(candidate,result.voltage,phase,ignored_codes=ignored_single_buses(candidate,options))
    if grid.import_metadata.get('format')=='ONS EMS STM v5':check_remote_policy(controls,trace,options.control_Q)
    if 'execution_load_models' in candidate.import_metadata:
        grid.import_metadata['execution_load_models']=copy.deepcopy(candidate.import_metadata['execution_load_models'])
    if 'baseline_reference_states' in candidate.import_metadata:
        grid.import_metadata['baseline_reference_states']=copy.deepcopy(candidate.import_metadata['baseline_reference_states'])
    if 'lcc_last_solution' in candidate.import_metadata:
        grid.import_metadata['lcc_last_solution']=copy.deepcopy(candidate.import_metadata['lcc_last_solution'])
    if result.converged:
        for key in ('discrete_last_solution', 'reactive_block_execution'):
            if key in candidate.import_metadata:
                grid.import_metadata[key]=copy.deepcopy(candidate.import_metadata[key])
    if logger is not None:
        for state in candidate.import_metadata.get('baseline_reference_states', []):
            logger.add_warning('Source-free isolated reference is deenergized only in the private PF state; '
                               'equipment and registered loads are retained', device=state['bus_code'],
                               value=state['retained_q_mvar'], expected_value='Registered MVAr retained')
        logger.add_info('Controles STM ajustados pelo motor AC',value=len(trace))
        allocation = grid.import_metadata.get('remote_control_execution', {})
        if allocation.get('remote_svc_groups') or allocation.get('local_generator_anchors'):
            import json
            logger.add_info('Equações compartilhadas de controle de reativo · referência8.6.2',
                value=json.dumps(allocation, ensure_ascii=False))
        for row in allocation.get('held', []):
            logger.add_warning('SVC remoto mantém admitância inicial pela precedência do SVC local',
                device=row['code'], value=row['initial_b_mvar'], expected_value='MVAr a1pu; contribuição elétrica preservada')
        if trace:
            for control in trace[-1]['controls']:
                if control['limited']:
                    logger.add_warning('Controle de tensão no limite',device=control['name'],
                                       value=control['value'],expected_value='Alvo limitado pela capacidade do controle')
    return result
