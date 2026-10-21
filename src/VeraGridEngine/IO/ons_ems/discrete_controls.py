# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Regulação discreta STM v5 sobre o adaptador AC já existente.

A busca usa passos físicos e novas soluções AC; uma sensibilidade/fatoração
calculada antes de trocar tap ou susceptância não é reutilizada nessa troca.
O domínio atual cobre OLTC de tensão e bancos discretos com estado identificável.
"""
from __future__ import annotations
import copy
import math
from decimal import Decimal
import numpy as np


def active_requests(grid, options, phase='PF'):
    if phase not in ('PF', 'CTG'):
        raise ValueError('Controles discretos: fase deve ser PF ou CTG.')
    effective = grid.import_metadata.get('effective_pf', {})
    explicit = effective.get('phase') == phase and effective.get('source') == 'explicit options'
    result = {'phase': phase}
    for name, suffix, field in [('tap', 'BLKTAP', 'control_taps_modules'),
                                ('phase_shift', 'BLKSHFT', 'control_taps_phase'),
                                ('shunt', 'BLKSHT', 'control_discrete_shunts')]:
        # The caller resolves PRM/defaults into PowerFlowOptions once. Reading
        # the raw PRM here again used to defeat a study's CTG/shunt override.
        result[name] = bool(getattr(options, field))
        result[name+'_source'] = ('explicit options' if explicit else
            phase+suffix if effective.get('phase') == phase and effective.get('source') == 'PRM/defaults'
            else 'study options')
    return result


def _finite(value, code, field):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{code}: campo {field} não numérico.') from None
    if not math.isfinite(value):
        raise ValueError(f'{code}: campo {field} não finito.')
    return value


def _integer(value, code, field):
    parsed = _finite(value, code, field)
    if parsed != int(parsed):
        raise ValueError(f'{code}: campo {field} deve ser inteiro.')
    return int(parsed)


def _band(upper, lower, code):
    lower = _finite(lower, code, 'Vmín'); upper = _finite(upper, code, 'Vmáx')
    if not 0 < lower <= upper:
        raise ValueError(f'{code}: faixa de tensão inválida, requer 0 < Vmín <= Vmáx (pu).')
    return lower, upper


def _resolve_target(grid, original, local, side=None):
    from VeraGridEngine.IO.ons_ems.stm_parser import bus_id
    mapping = {r['original']: r['converted'] for r in grid.import_metadata.get('association', {}).get('buses', [])}
    buses = {b.code: b for b in grid.buses}
    raw = bus_id(original)
    code = mapping.get(raw, raw) if raw != '0' else local.code
    if code not in buses or not buses[code].active:
        raise ValueError(f'Controle discreto: barra controlada {raw} -> {code} ausente/inativa.')
    return buses[code]


def _bank_states(row, code, max_states):
    """Quartetos do formato v5, preservando disponibilidade e sequência zero."""
    if (len(row)-7) % 4 or not 0 < (len(row)-7)//4 <= 8:
        raise ValueError(f'{code}: grupos de shunt v5 exigem 1 a 8 quartetos (estado, quantidade, MVAr, Z0).')
    groups = []
    states = {Decimal(0): ((), False)}
    operations = 0
    for col in range(7, len(row), 4):
        status = _integer(row[col], code, f'grupo {(col-7)//4+1}: estado')
        count = _integer(row[col+1], code, f'grupo {(col-7)//4+1}: quantidade')
        size = _finite(row[col+2], code, f'grupo {(col-7)//4+1}: tamanho MVAr')
        z0 = _finite(row[col+3], code, f'grupo {(col-7)//4+1}: Z0')
        if status not in (0, 1) or not 0 <= count <= 10000 or (count and size == 0):
            raise ValueError(f'{code}: estado/quantidade/tamanho inválido no grupo {(col-7)//4+1}.')
        groups.append({'available': bool(status), 'count': count, 'size_mvar': size, 'z0_pu': z0})
        updated = {}
        for total, (counts, ambiguous) in states.items():
            for n in range(count+1 if status else 1):
                operations += 1
                if operations > max_states*32:
                    raise ValueError(f'{code}: enumeração dos grupos excede o limite explícito de preparação ({max_states*32} combinações); sem truncamento.')
                value = total+Decimal(str(size))*n
                previous = updated.get(value)
                state = counts+(n,)
                updated[value] = (previous[0] if previous else state,
                                  ambiguous or previous is not None or (previous[1] if previous else False))
                if len(updated) > max_states:
                    raise ValueError(f'{code}: estados discretos excedem o limite explícito {max_states}; sem aproximação automática.')
        states = updated
    return groups, states


def validate_discrete_controls(grid, options, phase='PF', max_shunt_states=4096):
    """Devolve descritores ligados à rede privada; não altera posições elétricas."""
    from VeraGridEngine.IO.ons_ems.stm_parser import bus_id, branch_key
    requests = active_requests(grid, options, phase)
    blocked_individual = []
    grid.import_metadata.setdefault('effective_discrete',{})['individual_blocked'] = blocked_individual
    if not any(requests[k] for k in ('tap', 'phase_shift', 'shunt')):
        return []
    if not isinstance(max_shunt_states, int) or max_shunt_states < 1:
        raise ValueError('Limite de estados de bancos deve ser inteiro positivo.')
    if grid.import_metadata.get('format') != 'ONS EMS STM v5':
        raise ValueError('O contrato de controles discretos requer registros STM v5 incorporados.')
    source = grid.import_metadata.get('source_records', {})
    if not source:
        raise ValueError('Controles discretos: registros STM originais não incorporados.')
    actuators = []
    branches = {b.code: b for b in grid.transformers2w}
    p = grid.import_metadata.get('parameters', {})
    policy = grid.import_metadata.get('discrete_control_policy', {})
    method = policy.get('tap_method', 'discrete_steps' if p.get(phase+'DSCRTAP', False) is True else 'continuous_band')
    if method not in ('discrete_steps', 'continuous_band'):
        raise ValueError('Política de tap inválida; use discrete_steps ou continuous_band.')
    for row in source.get('transformers', []):
        code = branch_key(*row[:3]); branch = branches.get(code)
        if len(row) < 25:
            raise ValueError(f'{code}: registro de transformador v5 incompleto.')
        kind = _integer(row[3], code, 'tipo (4º campo)')
        if not int(row[20]) or branch is None or not branch.active or not branch.bus_from.active or not branch.bus_to.active:
            continue
        if kind == 3 and requests['phase_shift']:
            raise ValueError(f'{code}: tipo 3 SOLTC/PhShftMvar: controle de Q e sua banda ainda não qualificados; bloqueie {phase}BLKSHFT ou use opção explícita.')
        if kind not in (1,2,3,4):
            raise ValueError(f'{code}: tipo de transformador {kind} sem contrato automático qualificado.')
        if kind == 4 and requests['phase_shift']:
            from VeraGridEngine.enumerations import TapPhaseControl
            if branch.tap_phase_control_mode == TapPhaseControl.fixed:
                blocked_individual.append({'code':code,'kind':'phase','mode':'Fixed'})
                continue
            if branch.tap_phase_control_mode != TapPhaseControl.Pf:
                raise ValueError(f'{code}: modo individual de fase não corresponde ao contrato STM/MW no terminal De.')
            if policy.get('phase_method','continuous_phase') != 'continuous_phase':
                raise ValueError(f'{code}: somente continuous_phase/MW qualificado; campo 18 não foi qualificado como passo angular discreto.')
            lower = _finite(row[16],code,'limite inferior MW (17º campo)')
            upper = _finite(row[15],code,'limite superior MW (16º campo)')
            minimum = math.radians(_finite(row[14],code,'defasagem mínima graus (15º campo)'))
            maximum = math.radians(_finite(row[13],code,'defasagem máxima graus (14º campo)'))
            if lower>upper or minimum>maximum or min(minimum,maximum)<-math.pi or max(minimum,maximum)>math.pi:
                raise ValueError(f'{code}: banda MW ou limites angulares inválidos.')
            angle = _finite(branch.tap_phase,code,'defasagem atual rad')
            if not minimum-1e-9<=angle<=maximum+1e-9:
                raise ValueError(f'{code}: defasagem atual fora da faixa declarada; não será alterada silenciosamente.')
            target_policy = policy.get('phase_target_policy','lower_fraction')
            fraction = _finite(policy.get('phase_target_fraction',.1),code,'fração de alvo da banda MW')
            if target_policy not in ('band','lower_fraction') or not 0<=fraction<=1:
                raise ValueError(f'{code}: política de alvo MW inválida (band ou lower_fraction, fração entre 0 e 1).')
            actuators.append({'name':branch.name,'code':code,'kind':'phase','device':branch,
                'target_bus':branch.bus_from,'bus':branch.bus_from.code,'lower_mw':lower,'upper_mw':upper,
                'min':minimum,'max':maximum,'phase_method':'continuous_phase','measurement_terminal':'from',
                'source':requests['phase_shift_source'],'error_scale':grid.Sbase,
                'phase_target_policy':target_policy,'phase_target_fraction':fraction})
            continue
        if not requests['tap'] or kind != 2:
            continue
        from VeraGridEngine.enumerations import TapModuleControl
        if branch.tap_module_control_mode == TapModuleControl.fixed:
            blocked_individual.append({'code':code,'kind':'tap','mode':'Fixed'})
            continue
        if branch.tap_module_control_mode != TapModuleControl.Vm:
            raise ValueError(f'{code}: modo individual de tap não corresponde ao contrato STM de tensão.')
        minimum = _finite(row[14], code, 'tap mínimo (15º campo)')
        maximum = _finite(row[13], code, 'tap máximo (14º campo)')
        step = _finite(row[17], code, 'passo de tap (18º campo)')
        if not 0 < minimum <= maximum or step < 0 or (method == 'discrete_steps' and step == 0):
            raise ValueError(f'{code}: faixa/passo de tap inválido; requer 0 < mínimo <= máximo e passo > 0 pu no modo discreto.')
        count = (maximum-minimum)/step if step else 0.
        if method == 'discrete_steps' and (count > 10000 or abs(count-round(count)) > 1e-7):
            raise ValueError(f'{code}: limites não definem uma grade discreta íntegra (máximo-mínimo)/passo.')
        tap = _finite(branch.tap_module, code, 'tap atual')
        if tap < minimum-1e-9 or tap > maximum+1e-9:
            raise ValueError(f'{code}: tap atual fora da faixa; não será ajustado silenciosamente.')
        position = (tap-minimum)/step if step else 0.
        if method == 'discrete_steps' and abs(position-round(position)) > 1e-7:
            raise ValueError(f'{code}: tap atual fora da grade/faixa; não será arredondado silenciosamente.')
        lower, upper = _band(row[15], row[16], code)
        side = _integer(row[12], code, 'lado controlado (13º campo)')
        raw_target = bus_id(row[11])
        if raw_target == '0' and side not in (1, 2):
            raise ValueError(f'{code}: barra 0 exige lado controlado 1 ou 2.')
        target = _resolve_target(grid, raw_target, branch.bus_from if side == 1 else branch.bus_to)
        if target is branch.bus_from: side = 1
        elif target is branch.bus_to: side = 2
        elif side not in (1, 2):
            raise ValueError(f'{code}: barra remota exige lado controlado 1 ou 2 (13º campo).')
        actuators.append({'name': branch.name, 'code': code, 'kind': 'tap', 'device': branch,
            'target_bus': target, 'bus': target.code, 'original_bus': raw_target, 'side': side,
            'lower_pu': lower, 'upper_pu': upper, 'min': minimum, 'max': maximum,
            'step': step, 'position': int(round(position)), 'positions': int(round(count))+1,
            'tap_method': method, 'tap_method_source': 'explicit policy' if 'tap_method' in policy else phase+'DSCRTAP/default',
            'source': requests['tap_source']})
    if requests['shunt']:
        # Terminais possuem susceptância, mas seus registros não fornecem bancos
        # e faixas suficientes para a mesma política de manobra automática.
        for row in source.get('lines', []):
            code = branch_key(*row[:3])
            if len(row) > 24 and int(row[24]) or len(row) > 26 and int(row[26]):
                raise ValueError(f'{code}: controle de shunt terminal (campos 25/27) sem contrato de faixa/grupos qualificado; bloqueie {phase}BLKSHT.')
        shunts = {s.code: s for s in grid.shunts}
        for row in source.get('shunts', []):
            if len(row) < 7:
                raise ValueError('Controle shunt: registro v5 incompleto.')
            if _integer(row[1], 'shunt', 'modo') != 1:
                continue
            code = 'shunt|'+bus_id(row[0]); shunt = shunts.get(code)
            if shunt is None:
                raise ValueError(f'{code}: banco discreto sem equipamento nativo associado.')
            if not shunt.active or not shunt.bus.active:
                continue
            lower, upper = _band(row[2], row[3], code)
            target = _resolve_target(grid, row[4], shunt.bus)
            groups, states = _bank_states(row, code, max_shunt_states)
            value = _finite(shunt.B, code, 'admitância atual MVAr em Vm=1 pu')
            matches = [(b, state) for b, state in states.items() if abs(float(b)-value) <= 1e-7]
            if len(matches) != 1:
                raise ValueError(f'{code}: B atual não corresponde a um estado disponível dos grupos; informe dados/posições consistentes, sem remover bancos em manutenção.')
            counts, ambiguous = matches[0][1]
            given = grid.import_metadata.get('shunt_positions', {}).get(code)
            policy = grid.import_metadata.get('discrete_control_policy', {}).get('shunt_initial_state', 'unique')
            if given is not None:
                counts = tuple(_integer(n, code, 'posição por grupo') for n in given)
                if len(counts) != len(groups) or any(n < 0 or n > (g['count'] if g['available'] else 0) for n, g in zip(counts, groups)) or abs(sum(n*g['size_mvar'] for n,g in zip(counts, groups))-value) > 1e-7:
                    raise ValueError(f'{code}: shunt_positions não confere com B, limites ou disponibilidade.')
                initial_source = 'explicit group positions'
            elif ambiguous and policy != 'equivalent_admittance':
                raise ValueError(f'{code}: B inicial admite múltiplas posições de grupos; informe shunt_positions ou escolha explicitamente discrete_control_policy.shunt_initial_state=equivalent_admittance para estudo estático.')
            else:
                initial_source = 'equivalent admittance (explicit)' if ambiguous else 'unique group positions'
            actuators.append({'name': shunt.name, 'code': code, 'kind': 'shunt', 'device': shunt,
                'target_bus': target, 'bus': target.code, 'original_bus': bus_id(row[4]),
                'lower_pu': lower, 'upper_pu': upper, 'min': float(min(states)), 'max': float(max(states)),
                'groups': groups, 'counts': counts, 'initial_state_source': initial_source,
                'source': requests['shunt_source']})
    for kind, suffix in [('tap','TAPVARIT'),('shunt','SHTVARIT')]:
        if any(a['kind']==kind or kind=='tap' and a['kind']=='phase' for a in actuators):
            delay = _integer(p.get(phase+suffix,0), phase+suffix, 'atraso de controle')
            if delay != 0:
                raise ValueError(f'{phase+suffix}: atraso no Newton não qualificado; o adaptador aplica controle externo após cada AC convergido.')
    if any(a['kind'] in ('tap','shunt') for a in actuators) and p.get(phase+'MIDRANGE',False) is True:
        raise ValueError(f'{phase}MIDRANGE: alvo no centro da faixa não qualificado; política atual regula pela banda declarada.')
    # Uma medição de outra ilha não pode orientar a regulação deste atuador.
    adjacency = {b: set() for b in grid.buses if b.active}
    for b in grid.get_branches(add_hvdc=False, add_vsc=False, add_switch=True):
        if b.active and b.bus_from in adjacency and b.bus_to in adjacency:
            adjacency[b.bus_from].add(b.bus_to); adjacency[b.bus_to].add(b.bus_from)
    for a in actuators:
        start = a['device'].bus_from if a['kind'] in ('tap','phase') else a['device'].bus
        reached = {start}; pending = [start]
        while pending:
            for b in adjacency.get(pending.pop(), ()):
                if b not in reached: reached.add(b); pending.append(b)
        if a['target_bus'] not in reached:
            raise ValueError(f"{a['code']}: medição controlada em outra ilha; regulação automática não aplicável.")
    return actuators


def _snapshot(grid, controls):
    fields = [(g, ('P','Q','Vset','control_mode')) for g in grid.generators]
    fields += [(l, ('P','Q','Ir','Ii','G','B','active')) for l in grid.loads]
    fields += [(s, ('G','B','active')) for s in grid.shunts]
    fields += [(b, ('tap_module','tap_phase','active')) for b in grid.transformers2w]
    fields += [(h, ('active',)) for h in grid.hvdc_lines]
    fields += [(b, ('Vm0','Va0')) for b in grid.buses]
    return ([(device, {f:getattr(device,f) for f in names}) for device,names in fields],
        [c['value'] for c in controls],
        {k:copy.deepcopy(grid.import_metadata.get(k)) for k in ('lcc_last_solution','discrete_last_solution','shunt_positions')})


def _restore(grid, controls, state):
    fields, values, metadata = state
    for device, saved in fields:
        for field, value in saved.items(): setattr(device, field, value)
    for control, value in zip(controls, values): control['value'] = value
    for key, value in metadata.items():
        if value is None: grid.import_metadata.pop(key, None)
        else: grid.import_metadata[key] = copy.deepcopy(value)


def solve_discrete(grid, options, controls, *, phase='PF', max_steps=80, max_shunt_states=4096, power_tolerance_mw=None, **kwargs):
    """Rede privada, coordenação discreta determinística e rollback de tentativas.

    Prioridade: controles contínuos existentes, bancos, depois OLTC. Um ponto
    estacionário fora da banda fica inconclusivo, salvo esgotamento comprovado
    do recurso individual. Não se certifica a existência de solução global.
    """
    from VeraGridEngine.IO.ons_ems.lcc import prepare_working_loads, solve_coupled
    # This diagnostic describes this attempt, never the last automatic run.
    # The registered positions remain the inputs of each independent F5.
    grid.import_metadata.pop('discrete_last_solution', None)
    actuators = validate_discrete_controls(grid, options, phase, max_shunt_states)
    if not actuators:
        result, trace = solve_coupled(grid, options, controls, **kwargs)
        grid.import_metadata['discrete_last_solution'] = dict(
            schema_version=1, phase=phase, status='not_requested',
            ac_converged=bool(result.converged), complete=bool(result.converged),
            controls=[], steps=0, evaluations=0,
            position_source='Registered positions; general and individual permissions preserved')
        return result, trace
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('Controles discretos: limite de passos inválido.')
    prepare_working_loads(grid)
    entry = _snapshot(grid, controls)
    original_modes = {g:g.control_mode for g in grid.generators}
    settings = copy.copy(options)
    settings.control_taps_modules = False; settings.control_taps_phase = False
    settings.control_discrete_shunts = False
    callback = kwargs.get('cancel_callback')
    telemetry = kwargs.get('telemetry')
    tolerance = float(kwargs.get('voltage_tolerance', options.tolerance))
    power_tolerance_mw = options.tolerance*grid.Sbase if power_tolerance_mw is None else float(power_tolerance_mw)
    if not math.isfinite(power_tolerance_mw) or power_tolerance_mw<=0:
        raise ValueError('Tolerância do controle MW deve ser positiva e finita.')
    tolerances = np.array([power_tolerance_mw/grid.Sbase if a['kind']=='phase' else tolerance for a in actuators])
    solve_kwargs = dict(kwargs, sensitivity_cache=None)
    indices = {b.code:i for i,b in enumerate(grid.buses)}
    trace = []
    evaluations = 0
    def ac():
        nonlocal evaluations
        if callback is not None and callback():
            raise RuntimeError('Estudo cancelado durante controles discretos.')
        # Limites de Q devem poder liberar-se após uma nova posição de tap/B.
        for gen, mode in original_modes.items(): gen.control_mode = mode
        evaluations += 1
        if telemetry is not None: telemetry['discrete_ac_calls'] = evaluations
        result, records = solve_coupled(grid, settings, controls, **solve_kwargs)
        if not result.converged or not np.isfinite(result.voltage).all():
            raise RuntimeError('Solução AC discreta incompleta; estado válido anterior preservado.')
        return result, records
    def band(a):
        return (a['lower_mw'],a['upper_mw']) if a['kind']=='phase' else (a['lower_pu'],a['upper_pu'])
    def errors(result):
        ports = {}
        if any(a['kind']=='phase' for a in actuators):
            from VeraGridEngine.IO.ons_ems.pf_results import normalized_flows, ac_branches
            sf,_,_ = normalized_flows(grid,result,{'branch_records':grid.import_metadata.get('branch_records',[])})
            ports = {b.code:float(flow.real) for b,flow in zip(ac_branches(grid),sf)}
        measurements = [ports[a['code']] if a['kind']=='phase' else float(abs(result.voltage[indices[a['bus']]])) for a in actuators]
        error = []
        for value,a in zip(measurements,actuators):
            lower,upper = band(a)
            raw = value-upper if value>upper else value-lower if value<lower else 0.
            if a['kind']=='phase' and a['phase_target_policy']=='lower_fraction':
                if raw and 'active_target_mw' not in a:
                    a['active_target_mw']=lower+a['phase_target_fraction']*(upper-lower)
                if 'active_target_mw' in a: raw=value-a['active_target_mw']
            error.append(raw/a.get('error_scale',1.))
        return measurements,np.array(error)
    def neighbours(a):
        if a['kind']=='phase':
            angle = a['device'].tap_phase
            delta = max(1e-5,min(math.radians(.2),(a['max']-a['min'])*.05))
            return sorted({float(np.clip(angle+d,a['min'],a['max'])) for d in (-delta,delta)}-{angle})
        if a['kind'] == 'tap':
            if a['tap_method']=='continuous_band':
                tap = a['device'].tap_module
                delta = max(1e-5, min(.005, (a['max']-a['min'])*.05))
                return sorted(set(float(np.clip(tap+d,a['min'],a['max'])) for d in (-delta,delta)) - {tap})
            current = int(round((a['device'].tap_module-a['min'])/a['step']))
            return [a['min']+n*a['step'] for n in (current-1,current+1) if 0 <= n < a['positions']]
        candidates = []
        for index, (n,g) in enumerate(zip(a['counts'], a['groups'])):
            if not g['available']: continue
            for count in (n-1,n+1):
                if 0 <= count <= g['count']:
                    counts = list(a['counts']); counts[index] = count
                    candidates.append(tuple(counts))
        return candidates
    def set_position(a, value):
        if a['kind'] == 'tap': a['device'].tap_module = float(value)
        elif a['kind']=='phase': a['device'].tap_phase = float(value)
        else: a['device'].B = sum(n*g['size_mvar'] for n,g in zip(value, a['groups']))
    def position(a):
        return a['device'].tap_module if a['kind']=='tap' else a['device'].tap_phase if a['kind']=='phase' else a['counts']
    def finish(result, continuous, status, vm, error, step, failed):
        observations = []
        for k,(a,v,e) in enumerate(zip(actuators,vm,error)):
            value = a['device'].tap_module if a['kind']=='tap' else a['device'].tap_phase if a['kind']=='phase' else a['device'].B
            limit = abs(value-a['min']) <= 1e-8 or abs(value-a['max']) <= 1e-8
            row = {'name':a['name'], 'code':a['code'], 'kind':a['kind'], 'bus':a['bus'],
                'value':float(value), 'unit':'pu' if a['kind']=='tap' else 'rad' if a['kind']=='phase' else 'MVAr at Vm=1 pu',
                'error_pu':float(e), 'limited':bool(abs(e)>tolerances[k] and limit),
                'at_actuator_bound':bool(limit), 'source':a['source']}
            if a['kind']=='phase':
                row.update(power_mw=v,error_mw=float(e*grid.Sbase),lower_mw=a['lower_mw'],upper_mw=a['upper_mw'],
                    phase_deg=math.degrees(value),phase_method=a['phase_method'],measurement_terminal='from',tolerance_mw=power_tolerance_mw,
                    phase_target_policy=a['phase_target_policy'],phase_target_fraction=a['phase_target_fraction'],
                    target_mw=a.get('active_target_mw'),band_error_mw=(v-a['upper_mw'] if v>a['upper_mw'] else v-a['lower_mw'] if v<a['lower_mw'] else 0.))
            else:
                row.update(voltage_pu=v,lower_pu=a['lower_pu'],upper_pu=a['upper_pu'],tolerance_pu=tolerance)
                if a['kind']=='shunt': row.update(counts=list(a['counts']),initial_state_source=a['initial_state_source'])
                else: row.update(step_pu=a['step'],side=a['side'],tap_method=a['tap_method'],tap_method_source=a['tap_method_source'])
            observations.append(row)
        if status == 'actuator_bound' and any(abs(e)>tol and not r['limited'] for e,tol,r in zip(error,tolerances,observations)):
            status = 'unresolved_discrete_band'
        grid.import_metadata['discrete_last_solution'] = {'schema_version':1, 'phase':phase,
            'status':status, 'ac_converged':True, 'complete':status in ('band_satisfied','actuator_bound'),
            'method':'physical bank steps and selected OLTC method, with new AC solves', 'evaluations':evaluations,
            'steps':step, 'controls':observations, 'failed_probes':failed,
            'domain':'STM v5 OLTC voltage, discrete shunts and type4 phase/MW; static synthetic qualification',
            'dynamic_qualified':False, 'trace':trace}
        grid.import_metadata.setdefault('shunt_positions', {}).update({a['code']:list(a['counts']) for a in actuators if a['kind']=='shunt'})
        final = {'iteration':step, 'max_voltage_error_pu':float(np.max(np.abs(error))),
                 'controls':(continuous[-1]['controls'] if continuous else [])+observations,
                 'discrete_status':status}
        return result, continuous+[final]
    try:
        result, continuous = ac()
        failed = []
        for step in range(max_steps+1):
            vm, error = errors(result)
            score = float(error @ error)
            trace.append({'step':step, 'positions':{a['code']:list(position(a)) if a['kind']=='shunt' else position(a) for a in actuators},
                          'measurements':{a['code']:{'value':v,'unit':'MW' if a['kind']=='phase' else 'pu'} for a,v in zip(actuators,vm)}, 'band_error_squared':score})
            if np.all(np.abs(error) <= tolerances):
                return finish(result, continuous, 'band_satisfied', vm, error, step, failed)
            if step == max_steps:
                return finish(result, continuous, 'step_limit', vm, error, step, failed)
            baseline = _snapshot(grid, controls)
            best = None
            # Bancos antes de taps; grandezas e estados físicos não são relaxados.
            order = sorted(range(len(actuators)), key=lambda k:({'shunt':0,'tap':1,'phase':2}[actuators[k]['kind']], -abs(error[k]), actuators[k]['code']))
            for k in order:
                a = actuators[k]
                if abs(error[k]) <= tolerances[k]: continue
                candidates = neighbours(a)
                original_position = position(a)
                probes = []
                def evaluate(candidate):
                    nonlocal best
                    _restore(grid, controls, baseline)
                    set_position(a, candidate)
                    try:
                        trial, trial_trace = ac()
                        trial_vm, trial_error = errors(trial)
                        trial_score = float(trial_error @ trial_error)
                        probes.append((candidate, trial_vm[k]))
                        if trial_score < score-max(1e-20, score*1e-10) and (best is None or trial_score < best[0]-1e-20):
                            best = (trial_score, k, candidate, trial, trial_trace, _snapshot(grid, controls))
                    except RuntimeError as exc:
                        if callback is not None and callback(): raise
                        failed.append({'step':step, 'code':a['code'], 'candidate':list(candidate) if a['kind']=='shunt' else candidate,
                                       'classification':'numerical_inconclusive', 'reason':str(exc)})
                for candidate in candidates: evaluate(candidate)
                if probes and (a['kind']=='phase' or a['kind']=='tap' and a['tap_method']=='continuous_band'):
                    # Derivada medida por novas soluções AC com todos os controles
                    # e limites. O passo previsto também é resolvido e aceito apenas
                    # se reduzir o erro; não se pressupõe sinal nem monotonicidade.
                    point = max(probes, key=lambda pair:abs(pair[0]-original_position))
                    derivative = (point[1]-vm[k])/(point[0]-original_position)
                    if abs(derivative)>1e-12:
                        lower,upper = band(a)
                        margin = min(1e-6,(upper-lower)*.05)
                        target = lower+margin if error[k]<0 else upper-margin
                        if a['kind']=='phase' and a['phase_target_policy']=='lower_fraction':
                            target = lower+a['phase_target_fraction']*(upper-lower)
                        proposal = float(np.clip(original_position+(target-vm[k])/derivative,a['min'],a['max']))
                        if abs(proposal-original_position)>1e-10 and all(abs(proposal-v)>1e-10 for v,_ in probes):
                            evaluate(proposal)
                if best is not None: break
            _restore(grid, controls, baseline)
            if best is None:
                return finish(result, continuous, 'actuator_bound' if not failed else 'inconclusive_probes', vm, error, step, failed)
            _, k, candidate, result, continuous, accepted = best
            _restore(grid, controls, accepted)
            if actuators[k]['kind']=='shunt': actuators[k]['counts']=candidate
        raise AssertionError('Busca discreta terminou sem classificação.')
    except BaseException:
        _restore(grid, controls, entry)
        raise
