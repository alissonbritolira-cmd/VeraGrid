# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Electrical conversion ported from the developed and tested EMS STM path."""
from __future__ import annotations
import math,hashlib
from pathlib import Path
import numpy as np
from VeraGridEngine import Devices as vg
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.enumerations import GeneratorControlMode,ShuntControlMode,TapModuleControl,TapPhaseControl
from VeraGridEngine.IO.ons_ems.stm_parser import bus_id

def read_stm(path):
    from VeraGridEngine.IO.ons_ems.stm_parser import read_stm as source_reader, SECTION_NAMES
    data,source=source_reader(path)
    return {**{name:[row for row,_ in data[name]] for name in SECTION_NAMES},
            'Sbase':data['Sbase'],'header':data['header'],'sha256':source['sha256']}

def read_prm(path,with_history=False):
    from VeraGridEngine.IO.ons_ems.stm_parser import read_prm as source_reader
    parameters,history,_=source_reader(path)
    previous={};assignments=[]
    for item in history:
        if 'command' not in item:continue
        key=item['command'];assignments.append(dict(item,overrides_line=previous.get(key)))
        previous[key]=item['line']
    return (parameters,assignments) if with_history else parameters

def branch_key(f, t, circuit):
    return f'{bus_id(f)}|{bus_id(t)}|{str(circuit).strip()}'


def branch_records(data, orphan_records=None):
    records = []
    for x in data['lines']:
        records.append(dict(key=branch_key(*x[:3]), f=x[0], t=x[1], id=x[2], kind='line',
            r=float(x[3]), x=float(x[4]), b=float(x[5])/data['Sbase'], rate=float(x[6]),
            rate2=float(x[7]), rate3=float(x[8]), shtf=bool(int(x[9])), gf=float(x[10]), bf=float(x[11]),
            shtt=bool(int(x[12])), gt=float(x[13]), bt=float(x[14]), sf=int(x[15]) == 1,
            st=int(x[16]) == 1, tap=1., phase=0., name=x[22]))
    for x in data['transformers']:
        records.append(dict(key=branch_key(*x[:3]), f=x[0], t=x[1], id=x[2], kind='transformer',
            r=float(x[4]), x=float(x[5]), b=0., rate=float(x[6]), rate2=float(x[7]), rate3=float(x[8]),
            shtf=False, gf=0., bf=0., shtt=False, gt=0., bt=0., sf=int(x[18]) == 1,
            st=int(x[19]) == 1, tap=float(x[9]), phase=float(x[10]), name=x[24],
            ctrl_bus=x[11], ctrl_side=int(x[12]), ctrl=bool(int(x[20])), control_kind=int(x[3]),
            tap_min=float(x[14]), tap_max=float(x[13])))
    for x in data['series_caps']:
        records.append(dict(key=branch_key(*x[:3]), f=x[0], t=x[1], id=x[2], kind='series_cap',
            r=float(x[3]), x=float(x[4]), b=0., rate=float(x[5]), rate2=float(x[6]), rate3=float(x[7]),
            shtf=bool(int(x[8])), gf=float(x[9]), bf=float(x[10]), shtt=bool(int(x[11])),
            gt=float(x[12]), bt=float(x[13]), sf=int(x[14]) == 1, st=int(x[15]) == 1,
            tap=1., phase=0., name=x[17]))
    by_key = {r['key']: r for r in records}
    if len(by_key) != len(records):
        raise ValueError('Há códigos repetidos de ramos no STM.')
    for x in data['line_shunts']:
        key = branch_key(*x[:3])
        if key not in by_key:
            reverse=branch_key(x[1],x[0],x[2])
            if reverse in by_key:
                # Contrato observado: campos F/T são os terminais do ramo
                # cadastrado, mesmo quando a chave auxiliar chega invertida.
                key=reverse
            elif orphan_records is not None:
                orphan_records.append({'code':key,'raw':list(x),
                    'from_bus':bus_id(x[0]),'to_bus':bus_id(x[1]),
                    'active':bool(int(x[3]) or int(x[6])),
                    'reason':'Ramo do registro auxiliar ausente; equipamento não descartado nem ramo fabricado.',
                    'simulation_status':'blocked_missing_branch_association'})
                continue
            else:
                raise ValueError(f'Shunt de terminal sem ramo associado: {key}')
        record = by_key[key]
        for side, offset in (('f', 3), ('t', 6)):
            if int(x[offset]) == 1:
                record['sht'+side] = True
                record['g'+side] += float(x[offset+1])
                record['b'+side] += float(x[offset+2])
    for record in records:
        record['f'], record['t'] = bus_id(record['f']), bus_id(record['t'])
    return records


def open_stm(path, prm_path=None, frequency=60., logger=None, ems_table_path=None,
             planning_kb_path=None, association_policy='sage'):
    logger = logger if logger is not None else Logger()
    data = read_stm(path)
    parameters,assignments=read_prm(prm_path,with_history=True)
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError('Frequência inválida.')
    from VeraGridEngine.IO.ons_ems.vsc import records as vsc_records
    vsc_links=vsc_records(data) if data['vsc'] else []
    if any(record['active'] for record in vsc_links):
        raise NotImplementedError('STM com VSC energizado ainda não foi qualificado; a rede não foi importada parcialmente.')
    from VeraGridEngine.IO.ons_ems.lcc import records as lcc_records
    from VeraGridEngine.IO.ons_ems.association import read_auxiliaries,apply_association
    from VeraGridEngine.IO.ons_ems.policy import parameter_inventory
    links=lcc_records(data,parameters)
    auxiliaries=read_auxiliaries(ems_table_path,planning_kb_path)
    grid = vg.MultiCircuit(name=Path(path).stem, Sbase=data['Sbase'], fbase=frequency)
    grid.comments = 'Rede importada diretamente de STM do SAGE. Consulte os dados de importação para os parâmetros e as limitações.'
    buses, types = {}, {}
    metadata = {'format': 'ONS EMS STM v5', 'source': Path(path).name, 'sha256': data['sha256'],
                'parameters': parameters, 'source_records': data, 'diagnostics': [], 'adjustments': []}
    metadata['parameter_inventory']=parameter_inventory(parameters,assignments)
    metadata['parameter_assignments']=assignments
    if prm_path:
        prm_raw=Path(prm_path).read_bytes()
        try:prm_text=prm_raw.decode('utf-8-sig')
        except UnicodeDecodeError:prm_text=prm_raw.decode('cp1252')
        metadata['parameter_source']={'source':Path(prm_path).name,'sha256':hashlib.sha256(prm_raw).hexdigest(),'text':prm_text}
    metadata['lcc_records']=links
    metadata['vsc_records']=vsc_links
    grid.import_metadata = metadata

    def warning(text):
        metadata['diagnostics'].append(text)
        logger.add_warning(text)

    def bus(code):
        key = bus_id(code)
        if key not in buses:
            raise ValueError(f'Equipamento ligado a barra inexistente: {key}')
        return buses[key]

    aggregations = {}
    for section, factory, add in (
            ('areas', vg.Area, grid.add_area), ('zones', vg.Zone, grid.add_zone),
            ('substations', vg.Substation, grid.add_substation), ('owners',vg.Owner,grid.add_owner)):
        mapping = {}
        for row in data[section]:
            code = bus_id(row[0])
            if code in mapping:
                raise ValueError(f'Código repetido em {section}: {code}')
            obj = factory(name=row[1], code=code)
            add(obj)
            mapping[code] = obj
        aggregations[section] = mapping

    for row in data['buses']:
        if len(row) < 17:
            raise ValueError('Registro de barra STM incompleto.')
        key, kind = bus_id(row[0]), int(row[3])
        if key in buses:
            raise ValueError(f'Número repetido de barra: {key}')
        if kind not in (0, 1, 2, 3, 4):
            raise ValueError(f'Tipo de barra inválido: {key}')
        obj = vg.Bus(name=row[1], code=key, Vnom=float(row[2]), active=kind != 4,
                     is_slack=kind == 3, Vm0=float(row[9]), Va0=np.deg2rad(float(row[10])),
                     vmax=float(row[11]), vmin=float(row[12]))
        grid.add_bus(obj)
        buses[key], types[key] = obj, kind
        for field, section, column in (('area','areas',7),('zone','zones',8),('substation','substations',16)):
            code = bus_id(row[column])
            if code != '0' and code not in aggregations[section]:
                raise ValueError(f'Barra {key}: {field} inexistente ({code}).')
            setattr(obj, field, aggregations[section].get(code))
        if int(row[4]):
            grid.add_shunt(obj, vg.Shunt(name=f'Shunt da barra {key}', code=f'bus_shunt|{key}',
                                        G=float(row[5]), B=float(row[6])))
    for row in data['loads']:
        if len(row) < 9:
            raise ValueError('Registro de carga STM incompleto.')
        key = bus_id(row[0])
        vm = bus(key).Vm0
        zip_values = [float(row[i]) for i in (5, 6, 7, 8)]
        if any(zip_values) and vm <= 0:
            raise ValueError(f'Carga ZIP em {key} sem tensão inicial para normalização.')
        denominator = vm if vm > 0 else 1.
        # As parcelas STM estão em MW/Mvar no ponto informado. A corrente e
        # a admitância nativas são referidas a Vm=1. A comparação pela API
        # confirmou esta transformação em todos os registros do caso MS.
        grid.add_load(bus(key), vg.Load(name=row[12] if len(row) > 12 else f'Carga {key} {row[1]}',
            code=f'{key}|{row[1]}', active=bool(int(row[2])), P=float(row[3]), Q=float(row[4]),
            Ir=zip_values[0]/denominator, Ii=-zip_values[1]/denominator,
            G=zip_values[2]/denominator**2, B=-zip_values[3]/denominator**2))
        if any(zip_values):
            metadata['adjustments'].append({'code':f'{key}|{row[1]}','action':'zip_initial_voltage_normalization','vm_pu':vm})
    for row in data['generators']:
        if len(row) < 18:
            raise ValueError('Registro de gerador STM incompleto.')
        key, control = bus_id(row[0]), bus_id(row[7])
        bus(key)
        vset = float(row[6])
        if vset <= 0:
            target = bus(control if control != '0' else key)
            vset = target.Vm0
            if vset <= 0:
                raise ValueError(f'Gerador {key}|{row[1]} sem tensão especificada ou inicial válida.')
            metadata['adjustments'].append({'code': f'{key}|{row[1]}',
                'action': 'voltage_setting_from_bus', 'vset_pu': vset})
        # Limites de entrada explicitamente escolhidos no PRM. A API de
        # referência limita Vspec, mas conserva a semente Vm da barra.
        lower = float(parameters.get('ORGUNDERVOLT', 0.))
        upper = float(parameters.get('ORGOVERVOLT', math.inf))
        if not math.isfinite(lower) or lower < 0 or upper <= lower or math.isnan(upper):
            raise ValueError('Limites PRM de tensão de entrada inválidos.')
        bounded = min(upper, max(lower, vset))
        if bounded != vset:
            metadata['adjustments'].append({'code': f'{key}|{row[1]}',
                'action': 'voltage_setting_input_bounds', 'original_vset_pu': vset,
                'vset_pu': bounded, 'lower_pu': lower,
                'upper_pu': upper if math.isfinite(upper) else None})
            vset = bounded
        gen = vg.Generator(name=row[26] if len(row) > 26 else f'Gerador {key} {row[1]}',
            code=f'{key}|{row[1]}', P=float(row[2]), Q=float(row[3]), Qmax=float(row[4]), Qmin=float(row[5]),
            vset=vset, Snom=float(row[8]), active=bool(int(row[12])),
            Pmax=float(row[14]), Pmin=float(row[15]),enabled_dispatch=not bool(int(row[17])),
            r1=float(row[20]) if len(row)>25 else 1e-20,x1=float(row[21]) if len(row)>25 else 1e-20,
            r0=float(row[22]) if len(row)>25 else 1e-20,x0=float(row[23]) if len(row)>25 else 1e-20,
            control_mode=GeneratorControlMode.V if types[key] in (2, 3) else GeneratorControlMode.Q)
        if control != '0' and control != key:
            gen.control_bus = bus(control)
        grid.add_generator(bus(key), gen)
    # Production8.6.2 STM mode0 is not an additional injection: two independent
    # R50 GUI probes showed that changing its initial Mvar leaves PF/RMS N0
    # unchanged, while changing BUS Bsht changes both. AC Components shows the
    # fixed bus shunt and no separate mode0 Shunt object. Preserve the complete
    # declaration and the discrepancy with the generic guide, without summing
    # its initial value a second time or manufacturing a connection.
    metadata['fixed_shunt_representation']={
        'schema':1,'variant':'organon_862_STM5_mode0_declaration_BUS_injection',
        'source':'Production8.6.2 R50 actual GUI independent BUS/SHUNTS probes and AC Components; guide PDF212/215 describes fields, not this observed implementation behavior.',
        'schema_note':'Prior files remain readable with original graphs/results. Reimport STM explicitly to apply this representation; opening does not silently rewrite historical injections.'}
    metadata['fixed_shunt_declarations']=[]
    for row in data['shunts']:
        key, mode = bus_id(row[0]), int(row[1])
        if mode not in (0, 1, 2):
            raise ValueError(f'Modo de shunt inválido: {key}')
        if mode==0:
            original_bus=next((r for r in data['buses'] if bus_id(r[0])==key),None)
            metadata['fixed_shunt_declarations'].append(dict(
                bus=key,raw=list(row),declared_initial_mvar=float(row[5]),declared_status=bool(int(row[6])),
                representation='preserved_declaration_not_an_additional_injection',
                electrical_source='BUS Gsht/Bsht/status',bus_in_source=original_bus is not None,
                bus_shunt_status=bool(int(original_bus[4])) if original_bus else None,
                bus_conductance_mw=float(original_bus[5]) if original_bus else None,
                bus_susceptance_mvar=float(original_bus[6]) if original_bus else None,
                represented_by='bus_shunt|'+key if original_bus and int(original_bus[4]) else None))
            warning(f'Shunt fixo {key}: cartão SHUNTS modo0 preservado para consulta; a injeção usa G/B e estado do registro BUS, conforme a referência8.6.2. Não há soma duplicada dos valores.')
            continue
        if mode == 2:
            sizes = [int(row[i+1])*float(row[i+2]) for i in range(7, len(row)-3, 4) if int(row[i]) == 1]
            target = bus_id(row[4])
            shunt = vg.ControllableShunt(name=f'SVC {key}', code=f'svc|{key}', number_of_steps=0,
                B=float(row[5]), Bmin=sum(min(0., x) for x in sizes), Bmax=sum(max(0., x) for x in sizes),
                active=bool(int(row[6])), vset=float(row[2]), control_bus=bus(target if target != '0' else key),
                control_mode=ShuntControlMode.Continuous)
            grid.add_controllable_shunt(bus(key), shunt)
        else:
            grid.add_shunt(bus(key), vg.Shunt(name=f'Shunt {key}', code=f'shunt|{key}',
                                             B=float(row[5]), active=bool(int(row[6]))))
    orphans=[]
    records = branch_records(data,orphans)
    metadata['orphan_terminal_shunts']=orphans
    metadata['orphan_terminal_shunts_contract']=dict(schema_version=1,
        variant='organon_862_unassociated_terminal_auxiliary_not_applied_v1',
        treatment='Raw auxiliary declaration retained; no branch or independent bus injection synthesized.',
        evidence='Production8.6.2 GUI SIN original versus orphan B zero: complete PWF01/PWF20 exports identical.')
    for orphan in orphans:
        orphan.update(effective_active=False,simulation_status='auxiliary_not_applied_missing_branch',
            reason='Registro auxiliar sem ramo elétrico correspondente; conservado integralmente, não convertido em outro equipamento ou injeção.')
        warning(f'Registro auxiliar de shunt terminal {orphan["code"]} sem ramo: preservado e não aplicado, conforme contrato de importação8.6.2. Confira o cadastro; não foi criada injeção adicional.')
    metadata['branch_records']=records
    metadata['missing_inactive_buses']=[]
    missing_codes={r[side] for r in records for side in ('f','t') if r[side] not in buses}
    for key in sorted(missing_codes):
        connected=[r for r in records if key in (r['f'],r['t'])]
        if any(r['s'+('f' if r['f']==key else 't')] or r['kind']=='series_cap' for r in connected):
            raise ValueError(f'Equipamento energizado ligado a barra ausente no STM: {key}.')
        neighbors=[buses[r['t'] if r['f']==key else r['f']].Vnom for r in connected
                   if r['kind']=='line' and (r['t'] if r['f']==key else r['f']) in buses]
        nominal=neighbors[0] if neighbors and len(set(neighbors))==1 else 1.
        placeholder=vg.Bus(name=f'Barra ausente {key} (fora de serviço)',code=key,Vnom=nominal,active=False)
        grid.add_bus(placeholder);buses[key]=placeholder
        metadata['missing_inactive_buses'].append({'code':key,'branches':[r['key'] for r in connected],
            'nominal_placeholder_kv':nominal,'executable':False,'reason':'Todos os terminais desta barra estão abertos; dados originais ausentes.'})
        warning(f'Barra {key} ausente: conexões abertas preservadas; ativação exige dados completos.')
    for record in records:
        f, t = bus(record['f']), bus(record['t'])
        r, x = record['r'], record['x']
        active = record['sf'] and record['st']
        if record['kind'] == 'series_cap' and not active:
            # No formato v5, a abertura do capacitor significa seu bypass.
            # O bypass tem reatância própria de -1e-5 pu no formato atendido.
            # ORGZMIN é um piso geral de impedâncias, não este parâmetro.
            r, x, active = 0., -1e-5, True
            metadata['adjustments'].append({'code': record['key'], 'action': 'series_cap_bypass', 'x_pu': x})
        if not all(math.isfinite(v) for v in (r, x, *(record[key] for key in (
                'b','rate','rate2','rate3','tap','phase','gf','bf','gt','bt')))):
            raise ValueError(f'Parâmetro elétrico não finito no ramo {record["key"]}.')
        if min(record['rate'],record['rate2'],record['rate3'])<0:
            raise ValueError(f'Limite térmico negativo no ramo {record["key"]}.')
        if record['kind'] == 'transformer':
            if record['tap'] <= 0:
                raise ValueError(f'Tap não positivo no transformador {record["key"]}.')
            obj = vg.Transformer2W(bus_from=f, bus_to=t, name=record['name'], code=record['key'],
                HV=max(f.Vnom, t.Vnom), LV=min(f.Vnom, t.Vnom), r=r, x=x, rate=record['rate'],
                tap_module=record['tap'], tap_phase=np.deg2rad(record['phase']), active=active,
                nominal_power=record['rate'])
            # Capability is held by the existing individual native mode; the
            # general GUI permission is resolved separately and starts off.
            if record.get('ctrl') and record.get('control_kind') == 2:
                obj.tap_module_control_mode = TapModuleControl.Vm
            elif record.get('ctrl') and record.get('control_kind') == 4:
                obj.tap_phase_control_mode = TapPhaseControl.Pf
            grid.add_transformer2w(obj)
        else:
            obj = vg.Line(bus_from=f, bus_to=t, name=record['name'], code=record['key'],
                          r=r, x=x, b=record['b'], rate=record['rate'], active=active)
            grid.lines.append(obj)
        if record['rate'] > 0:
            obj.contingency_factor = record['rate3']/record['rate']
            obj.protection_rating_factor = record['rate3']/record['rate']
        for side, terminal in (('f', f), ('t', t)):
            if record['sht'+side] and record['s'+side]:
                grid.add_shunt(terminal, vg.Shunt(name=f'Terminal {record["key"]} {side}',
                    code=f'terminal|{record["key"]}|{side}', G=record['g'+side]*grid.Sbase,
                    B=record['b'+side]*grid.Sbase))
        if record['kind'] == 'line' and record['sf'] != record['st']:
            if complex(r, x) == 0:
                raise ValueError('Linha com terminal aberto e impedância nula.')
            ys, yc = 1/complex(r, x), 1j*record['b']/2
            yin = yc + ys*yc/(ys+yc)
            grid.add_shunt(f if record['sf'] else t, vg.Shunt(name=f'Terminal aberto {record["key"]}',
                code=f'open_end|{record["key"]}', G=yin.real*grid.Sbase, B=yin.imag*grid.Sbase))
    # SAGE pode fornecer Vm=0 numa barra energizada sem medida disponível.
    # É uma semente ausente, não uma condição elétrica de tensão nula.
    # Use o vizinho conectado mais próximo; mantenha a ausência no registro STM.
    missing = {b for b in grid.buses if b.active and b.Vm0 == 0}
    branches = grid.get_branches()
    while missing:
        changes = []
        for target in missing:
            candidates = []
            for branch in branches:
                if not branch.active:
                    continue
                other = branch.bus_to if branch.bus_from is target else (
                    branch.bus_from if branch.bus_to is target else None)
                if other is not None and other.active and other.Vm0 > 0:
                    candidates.append((abs(complex(branch.R, branch.X)), str(other.code), other))
            if candidates:
                other = min(candidates, key=lambda item: item[:2])[2]
                changes.append((target, other.Vm0, other.Va0, str(other.code)))
        if not changes:
            changes = [(b, 1., 0., None) for b in missing]
        for target, vm, va, neighbor in changes:
            target.Vm0, target.Va0 = vm, va
            missing.remove(target)
            metadata['adjustments'].append({'code':str(target.code), 'action':'missing_voltage_seed',
                'neighbor':neighbor, 'vm_pu':vm, 'angle_rad':va})
    if prm_path:
        metadata['prm'] = {'source': Path(prm_path).name, 'sha256': hashlib.sha256(Path(prm_path).read_bytes()).hexdigest()}
    else:
        warning('STM importado sem PRM. Opções adicionais de cálculo não foram fornecidas.')
    for rec in links:
        r,i=rec['rectifier'],rec['inverter']
        link=vg.HvdcLine(bus_from=bus(r[0]),bus_to=bus(i[0]),name=rec['name'],code=rec['code'],
            active=rec['active'],rate=rec['rate'],Pset=rec['order'] if rec['mode'] in (1,3) else 0.,
            r=rec['resistance'],dc_link_voltage=rec['nominal_voltage'],dispatchable=False)
        grid.add_hvdc(link)
    if any(int(row[1]) == 1 for row in data['shunts']):
        warning('Bancos de shunts conservam a posição informada. O chaveamento depende das opções efetivas e da associação dos bancos disponíveis.')
    if any(record.get('ctrl') for record in records):
        warning('Taps e defasadores conservam a posição informada. Confira o controle solicitado e o domínio executável nos dados de importação.')
    if any(gen.control_bus is not None and gen.control_bus is not gen.bus for gen in grid.generators):
        warning('Controle remoto usa o ajuste AC compartilhado, com as participações do STM.')
    metadata['remote_voltage_controls']=any(gen.control_bus is not None and gen.control_bus is not gen.bus for gen in grid.generators)
    from VeraGridEngine.IO.ons_ems.prefault import has_stm_control_actuators
    metadata['pf_adapter_required'] = bool(grid.controllable_shunts or metadata['remote_voltage_controls'] or any(r['active'] for r in links)
        or any(r['active'] for r in orphans)
        or parameters.get('PFMWLDMDL',0) or parameters.get('PFMVARLDMDL',0)
        or has_stm_control_actuators(grid))
    if any(r['active'] for r in links):
        ccc_codes=[r['code'] for r in links if r['active'] and r.get('converter_type')=='CCC']
        reason=('CCC periódico disponível; controles, compounding, transições e equivalência externa em qualificação.' if ccc_codes else
                'Potência e perdas conferidas na referência; reativo do inversor e equivalência completa ainda divergentes.')
        metadata['lcc_qualification']={'qualified':False,'status':'experimental','reason':reason,'ccc_codes':ccc_codes}
        warning('Conversores CC em qualificação: fluxo AC/CC disponível; avaliação de segurança permanece incompleta enquanto a equivalência não for demonstrada.')
    if parameters.get('PFCURRATING'):
        warning('Carregamento STM calculado por corrente, nos dois terminais físicos, conforme PFCURRATING.')
    for parameter in ('PFBLKTAP', 'PFBLKSHT', 'PFBLKSHFT'):
        if parameters.get(parameter) is False:
            warning(f'{parameter}=F: controle solicitado; a execução verificará o modelo, as faixas e a política de cada equipamento.')
    for devices in (grid.buses, grid.generators, grid.loads, grid.shunts, grid.controllable_shunts):
        for device in devices:
            for name in ('Vnom', 'Vm0', 'Va0', 'Vset', 'P', 'Q', 'Ir', 'Ii', 'G', 'B',
                         'Bmin','Bmax','Snom','Pmin', 'Pmax', 'Qmin', 'Qmax'):
                if hasattr(device, name) and not math.isfinite(float(getattr(device, name))):
                    raise ValueError(f'Valor não finito em {device.code}: {name}.')
    if any(b.Vnom<=0 or b.Vm0<0 for b in grid.buses):
        raise ValueError('Tensão nominal não positiva ou semente de tensão negativa.')
    metadata['original_load_models']=[dict(code=l.code,voltage_pu=l.bus.Vm0,
        **{k:getattr(l,k) for k in ('P','Q','Ir','Ii','G','B')}) for l in grid.loads]
    # Equipment identities remain stable even when their bus numbering changes.
    association=apply_association(grid,auxiliaries,association_policy)
    conversion={r['original']:r['converted'] for r in association['buses']}
    for record in records:
        record['f']=conversion[record['f']];record['t']=conversion[record['t']]
    for section,collection,column in [('loads',grid.loads,9),('generators',grid.generators,18)]:
        for row,obj in zip(data[section],collection):
            owner=aggregations['owners'].get(bus_id(row[column])) if len(row)>column else None
            if owner is not None:obj.associate_owner(owner)
    objects_by_code={str(b.code):b for b in grid.get_branches()}
    for row,obj in [(row,objects_by_code.get(branch_key(*row[:3])))
                    for section in ('lines','transformers','series_caps') for row in data[section]]:
        col=19 if obj is not None and obj in grid.lines and len(row)>22 else 22 if obj is not None and obj in grid.transformers2w else 16
        owner=aggregations['owners'].get(bus_id(row[col])) if len(row)>col else None
        if obj is not None and owner is not None:obj.associate_owner(owner)
    return grid
