# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Elos LCC: equações de ponte, controle CC e acoplamento ao fluxo AC.

O objeto HVDC identifica o elo na rede; os dados de ponte pertencem ao deck.
Injeções de trabalho são privadas e recalculadas a cada tensão AC, nunca uma
substituição permanente do elo por cargas constantes.
"""
from __future__ import annotations
import math
import copy
import numpy as np


def records(data, parameters=None):
    from VeraGridEngine.IO.ons_ems.stm_parser import bus_id
    rows = data['dclinks']
    if len(rows) % 3:
        raise ValueError('Elo CC incompleto: faltam registros de conversor.')
    result = []
    known_codes = set()
    parameters = parameters or {}
    for h, r, i in zip(rows[::3], rows[1::3], rows[2::3]):
        if len(h) != 13 or len(r) != 14 or len(i) != 14:
            raise ValueError('Layout de elo CC v5 não reconhecido.')
        mode = int(h[3])
        if mode not in range(5):
            raise ValueError(f'Modo CC inválido no polo {h[0]}: {mode}.')
        rec = dict(code=f'{bus_id(r[0])}|{bus_id(i[0])}|DC{h[0]}',
                   pole=str(h[0]), mode=mode, resistance=float(h[4]),
                   order=float(h[5]), voltage=float(h[6]), threshold=float(h[7]),
                   margin=float(h[8]), compounding=float(h[9]),
                   nominal_voltage=float(h[10]), rate=float(h[11]), name=h[12],
                   rectifier=list(r), inverter=list(i))
        rec['active'] = bool(mode and rec['order'])
        rec['converter_type'] = 'CCC' if float(r[12]) or float(i[12]) else 'LCC'
        rec['use_inverter_minimum_gamma'] = bool(parameters.get('ORGDCINVGMIN', False))
        if rec['code'] in known_codes:
            raise ValueError(f'Elo CC duplicado: {rec["code"]}.')
        known_codes.add(rec['code'])
        if rec['active']:
            if rec['order'] < 0:
                raise NotImplementedError(f'{rec["code"]}: inversão de sentido LCC não qualificada.')
            if float(r[4]) or float(i[4]):
                raise NotImplementedError(f'{rec["code"]}: perdas resistivas de ponte não qualificadas.')
            values = [rec[k] for k in ('resistance','order','voltage','threshold','margin','compounding','nominal_voltage','rate')]
            values += [float(v) for row in (r,i) for v in row[1:13]]
            if not all(math.isfinite(v) for v in values) or rec['voltage'] <= 0 or rec['resistance'] < 0:
                raise ValueError(f'{rec["code"]}: parâmetros CC inválidos.')
            for row in (r,i):
                if (int(row[1]) <= 0 or float(row[5]) < 0 or float(row[6]) <= 0 or
                    float(row[7]) <= 0 or float(row[11]) <= 0 or float(row[12]) < 0 or
                    not 0 < float(row[10]) <= float(row[8]) <= float(row[9])):
                    raise ValueError(f'{rec["code"]}: ponte ou faixa de tap inválida.')
        result.append(rec)
    return result


def _bridge_reactive(power, angle, e0, drop, current):
    """Fundamental da corrente durante a comutação; sem ajuste empírico.

    tan(phi) = tan(angle + mu/2) + (mu/sin(mu)-1)/sin(2*angle+mu).
    Esta forma evita subtrair cossenos próximos e conserva o limite mu=0.
    """
    if drop == 0. or current == 0.:
        return power * math.tan(angle), 0.
    cosine = math.cos(angle) - 2 * drop * current / e0
    if not -1. < cosine <= 1.:
        raise RuntimeError('LCC: sobreposição de comutação fora do domínio físico.')
    # acos perto de 1 perde resolução; atan2 conserva o seno positivo.
    end = math.atan2(math.sqrt(max(0., 1.-cosine*cosine)), cosine)
    mu = end-angle
    if not 0. <= mu < math.pi/3:
        raise RuntimeError('LCC: sobreposição >= 60 graus não qualificada.')
    if mu == 0.:
        return power * math.tan(angle), 0.
    if abs(mu) < 1e-3:
        correction = mu*mu/6 + 7*mu**4/360 + 31*mu**6/15120
    else:
        correction = mu/math.sin(mu)-1.
    denominator = math.sin(2*angle+mu)
    if denominator <= 0.:
        raise RuntimeError('LCC: fator de potência fora do domínio de retificação.')
    return power * (math.tan(angle+mu/2)+correction/denominator), mu


def _bridge_control(row, vm, voltage, current, target, side):
    """Tap contínuo e ângulo limitado: equação elétrica em kV e kA."""
    vm, voltage, current = float(vm), float(voltage), float(current)
    n = int(row[1])
    eraw = 3*math.sqrt(2)/math.pi*n*float(row[6])*float(row[7])*vm
    drop = 3/math.pi*n*float(row[5])
    minimum, maximum = math.radians(float(row[3])), math.radians(float(row[2]))
    tap_min, tap_max = float(row[10]), float(row[9])
    if not 0. <= minimum <= maximum < math.pi/2:
        raise RuntimeError(f'LCC: faixa de ângulo {side} não qualificada.')
    target = min(max(float(target), minimum), maximum)
    demand = voltage + drop*current
    tap = eraw*math.cos(target)/demand
    clipped = min(max(tap, tap_min), tap_max)
    limited = bool(clipped != tap)
    angle = target
    if limited:
        cosine = demand*clipped/eraw
        if not 0. <= cosine <= 1.:
            raise RuntimeError(f'LCC: tensão impossível na faixa de tap do {side}.')
        angle = math.acos(cosine)
        if angle < minimum-1e-10 or angle > maximum+1e-10:
            raise RuntimeError(f'LCC: faixa de ângulo e tap do {side} esgotada; transferência de controle não qualificada.')
        angle = min(max(angle, minimum), maximum)
        tap = clipped
    e0 = eraw/tap
    p = voltage*current
    q, mu = _bridge_reactive(p, angle, e0, drop, current)
    return dict(tap=tap, angle=angle, overlap=mu, q=q, voltage_residual_kv=e0*math.cos(angle)-drop*current-voltage,
                tap_limit_active=limited, requested_angle=target, tap_mode='continuous')


def operating_point(record, vm_rectifier, vm_inverter):
    """MW/MVAr, kV, kA e radianos; P positiva sai da barra AC.

    Vdr-Vdi=Rline*Id; modos 1/2 fixam o terminal inversor e 3/4 o
    retificador. Cada ponte ajusta o tap ao ângulo especificado e aplica
    os limites físicos. O máximo do registro é o ângulo especificado no
    dialeto observado; ORGDCINVGMIN seleciona o mínimo do inversor.
    """
    if not record['active']:
        return dict(pf_mw=0., pt_mw=0., qf_mvar=0., qt_mvar=0., losses_mw=0., current_ka=0.)
    r, i = record['rectifier'], record['inverter']
    if not (float(r[12]) or float(i[12])) and record.get('lcc_tap_policy','continuous') != 'continuous':
        raise RuntimeError('LCC: apenas o controle contínuo de tap está qualificado; o passo cadastrado não habilita discretização.')
    if float(r[12]) or float(i[12]):
        from VeraGridEngine.IO.ons_ems.ccc import operating_point as ccc_point
        return ccc_point(record, vm_rectifier, vm_inverter)
    if not all(math.isfinite(v) and v > 0 for v in (vm_rectifier, vm_inverter)):
        raise RuntimeError('LCC energizado sem tensão AC válida em um dos terminais.')
    if float(r[4]) or float(i[4]):
        raise RuntimeError('LCC: perdas resistivas de ponte não qualificadas.')
    mode = int(record['mode'])
    vs, rl = float(record['voltage']), float(record['resistance'])
    if mode not in (1,2,3,4) or not all(math.isfinite(v) for v in (vs,rl,record['order'])) or vs <= 0 or rl < 0:
        raise RuntimeError('LCC: modo ou parâmetros de controle inválidos.')
    current = record['order']/1000 if mode in (2,4) else record['order']/vs
    vi = vs if mode in (1,2) else vs-rl*current
    vr = vi+rl*current
    if mode in (1,3) and record['threshold'] > 0 and vr < record['threshold']:
        raise RuntimeError('LCC: transição potência/corrente abaixo do limiar ainda não qualificada.')
    if min(vi,vr,current) <= 0:
        raise RuntimeError('LCC fora do domínio de tensão/corrente positivo.')
    gamma_target = math.radians(float(i[3] if record.get('use_inverter_minimum_gamma', False) else i[2]))
    retifier = _bridge_control(r, vm_rectifier, vr, current, math.radians(float(r[2])), 'retificador')
    inverter = _bridge_control(i, vm_inverter, vi, current, gamma_target, 'inversor')
    pf, received = vr*current, vi*current
    return dict(pf_mw=pf, pt_mw=-received, qf_mvar=retifier['q'], qt_mvar=inverter['q'],
                losses_mw=rl*current**2, current_ka=current,
                rectifier_voltage_kv=vr, inverter_voltage_kv=vi,
                alpha_rad=retifier['angle'], gamma_rad=inverter['angle'],
                rectifier_tap=retifier['tap'], inverter_tap=inverter['tap'],
                rectifier_overlap_rad=retifier['overlap'], inverter_overlap_rad=inverter['overlap'],
                effective_mode=mode, converter_type='LCC',
                controls={'rectifier':retifier,'inverter':inverter},
                control_residual_mw=((received if mode==1 else pf)-record['order']) if mode in (1,3) else 0.,
                control_residual_ka=(current-record['order']/1000) if mode in (2,4) else 0.,
                balance_residual_mw=pf-received-rl*current**2)


def prepare_working_loads(grid):
    """Criar antes da fotografia elétrica; a restauração inclui estas injeções."""
    from VeraGridEngine.Devices.Injections.load import Load
    known={str(x.code) for x in grid.loads}
    for h in grid.hvdc_lines:
        if not any(r['code']==h.code and r['active'] for r in grid.import_metadata.get('lcc_records',[])):continue
        for side,bus in [('f',h.bus_from),('t',h.bus_to)]:
            code='lcc_work|'+h.code+'|'+side
            if code not in known:
                grid.add_load(bus,Load(name=h.name+' '+side,code=code,active=False));known.add(code)


def solve_coupled(grid, options, controls, **kwargs):
    """Recalcula os conversores em uma rede já privada, mantendo o adaptador AC."""
    from VeraGridEngine.Devices.Injections.load import Load
    from VeraGridEngine.IO.ons_ems.controls_adapter import solve_with_controls
    records_by_code = {r['code']:r for r in grid.import_metadata.get('lcc_records', [])}
    links = [(h,records_by_code[str(h.code)],bool(h.active)) for h in grid.hvdc_lines
             if str(h.code) in records_by_code and records_by_code[str(h.code)]['active']]
    if not links:
        return solve_with_controls(grid,options,controls,**kwargs)
    for h, rec, active in links:
        if rec['mode'] in (2,4) and abs(float(h.Pset))>1e-12:
            raise RuntimeError(f'{h.code}: modo corrente; edite a ordem em A, não Pset em MW. A edição MW não será ignorada.')
    if 'sensitivity_cache' in kwargs:kwargs['sensitivity_cache']=None
    loads = {str(x.code):x for x in grid.loads}
    working = []
    for h,rec,active in links:
        rec=dict(rec,resistance=float(h.r),use_inverter_minimum_gamma=bool(
            grid.import_metadata.get('parameters',{}).get('ORGDCINVGMIN',rec.get('use_inverter_minimum_gamma',False))))
        if rec['mode'] in (1,3):rec=dict(rec,order=float(h.Pset))
        pair=[]
        for side,bus in [('f',h.bus_from),('t',h.bus_to)]:
            key='lcc_work|'+h.code+'|'+side
            load=loads.get(key)
            if load is None:
                load=Load(name=h.name+' '+side,code=key)
                grid.add_load(bus,load)
            load.active=active and bus.active
            pair.append(load)
        h.active=False  # the native fixed-power approximation must not run too
        working.append((h,rec,active,pair))
    v=np.array([complex(b.Vm0)*np.exp(1j*b.Va0) for b in grid.buses])
    indices={b:i for i,b in enumerate(grid.buses)}
    trace=[]
    original_modes={g:g.control_mode for g in grid.generators}
    try:
        for iteration in range(40):
            for g,mode in original_modes.items():g.control_mode=mode
            states=[]
            for h,rec,active,pair in working:
                rec=dict(rec,active=active and bool(rec['order']) and h.bus_from.active and h.bus_to.active)
                state=operating_point(rec,abs(v[indices[h.bus_from]]),abs(v[indices[h.bus_to]]))
                pair[0].P,pair[0].Q=state['pf_mw'],state['qf_mvar']
                pair[1].P,pair[1].Q=state['pt_mw'],state['qt_mvar']
                states.append(state)
            result,ac_trace=solve_with_controls(grid,copy.copy(options),controls,**kwargs)
            trace.extend(ac_trace)
            if not np.all(np.isfinite(result.voltage)):
                raise RuntimeError('Tensão AC inválida durante o acoplamento LCC; último estado válido preservado.')
            mismatch=0.
            for (h,rec,active,pair),state in zip(working,states):
                check=operating_point(dict(rec,active=active and bool(rec['order']) and h.bus_from.active and h.bus_to.active),
                    abs(result.voltage[indices[h.bus_from]]),abs(result.voltage[indices[h.bus_to]]))
                mismatch=max(mismatch,*(abs(check[k]-state[k])/grid.Sbase for k in ('pf_mw','pt_mw','qf_mvar','qt_mvar')))
            # Uma tentativa AC pode recuperar-se na próxima iteração externa.
            # Nenhuma tentativa incompleta pode tornar-se a solução válida.
            if mismatch <= options.tolerance and result.converged:
                for index,h in enumerate(grid.hvdc_lines):
                    k=next((k for k,w in enumerate(working) if w[0] is h),None)
                    if k is None:continue
                    s=states[k]
                    result.Pf_hvdc[index],result.Pt_hvdc[index]=s['pf_mw'],s['pt_mw']
                    result.Qf_hvdc[index],result.Qt_hvdc[index]=s['qf_mvar'],s['qt_mvar']
                    result.hvdc_terminal_q_complete[index]=1
                    result.losses_hvdc[index]=s['losses_mw']
                    result.loading_hvdc[index]=s['pf_mw']/h.rate if h.rate>0 else 0.
                grid.import_metadata['lcc_last_solution']={'mismatch_pu':mismatch,'iterations':iteration+1,
                    'states':{h.code:s for (h,_,_,_),s in zip(working,states)}}
                return result,trace
            v=0.6*result.voltage+0.4*v
        raise RuntimeError('Acoplamento AC/LCC não convergiu; resultado não aprovado.')
    finally:
        for h,_,active,_ in working:h.active=active
