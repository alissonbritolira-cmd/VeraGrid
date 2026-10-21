# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations
import copy,math
def parameter_inventory(parameters, assignments=None):
    records=[]
    direct={'MVATOL':'tolerance','MAXITER':'max_iter','FLTSTRT':'use_stored_guess',
            'GENVARIT':'control_Q','CURRATING':'thermal metric','OVRLDLV':'thermal level',
            'MWLDMDL':'active load model','MVARLDMDL':'reactive load model'}
    history = assignments or [{'command': name, 'value': value} for name, value in parameters.items()]
    last = {row['command']: index for index, row in enumerate(history)}
    for index, assignment in enumerate(history):
        name, value = assignment['command'], assignment['value']
        phase=next((p for p in ('CPF','CTG','PF') if name.startswith(p)),None)
        suffix=name[len(phase):] if phase else ''
        status='metadata';detail='Sem correspondência executável nesta integração.'
        if name in ('ORGUNDERVOLT','ORGOVERVOLT'):
            status='supported';detail='Limite de Vspec aplicado na importação; conserva Vm inicial e registra ajustes.'
        elif name=='ORGDCINVGMIN':
            status='conditional'
            detail=('Alvo mínimo de extinção do inversor.' if value else 'Alvo cadastrado de extinção do inversor.')+' Executado pelo adaptador LCC/CCC no domínio qualificado do equipamento; não certifica o caso completo.'
        if phase in ('PF','CTG'):
            if suffix in direct:
                status='supported';detail=direct[suffix]
                if suffix=='GENVARIT' and value not in (0,parameters.get(phase+'MAXITER',80)):
                    status='blocked';detail='Atraso intermediário de atuação de Q não qualificado.'
            elif suffix in ('BLKTAP','BLKSHT','BLKSHFT'):
                status='supported' if value is True else 'conditional'
                detail='Posição informada, controle bloqueado.' if value is True else 'Controle solicitado; verificar o contrato por equipamento e a execução.'
            elif suffix in ('DSCRTAP','TAPVARIT','SHTVARIT','MIDRANGE'):
                status='conditional';detail='Semântica depende do atuador; confira controles discretos e comandos recusados.'
            elif suffix in ('DSTSLACK','AREACNTRL','BLOCKQG','CHKCTRVLT','PRLTRFTAP','USEFIXSHT'):
                status='inactive' if value is False else 'blocked'
                detail='Comando desativado.' if value is False else 'Controle solicitado sem implementação qualificada.'
                if suffix == 'BLOCKQG' and value is True:
                    status='conditional'
                    detail='Bloqueio PV→PQ nas barras selecionadas; exige seleção efetiva e Q inicial do próprio ponto.'
            elif suffix=='VTHRS':
                status='conditional';detail='CP/I abaixo do limiar é recusado; transição para Z não implementada.'
        if last[name] != index:
            status='superseded';detail='Substituído por uma atribuição posterior no PRM.'
        records.append({'command':name,'requested':value,'phase':phase,'status':status,'detail':detail,
                        'line':assignment.get('line'),'effective_assignment':last[name] == index,
                        'overrides_line':assignment.get('overrides_line')})
    return records


def pf_options(grid,options=None,phase='PF'):
    from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
    from VeraGridEngine.enumerations import SolverType
    if 'branch_records' not in grid.import_metadata:
        source=grid.import_metadata.get('source_records',{})
        if all(k in source for k in ('lines','transformers','series_caps','line_shunts','Sbase')):
            from VeraGridEngine.IO.ons_ems.stm_parser import branch_records
            rows=branch_records(source)
            mapping={r['original']:r['converted'] for r in grid.import_metadata.get('association',{}).get('buses',[])}
            for r in rows:r['f']=mapping.get(r['f'],r['f']);r['t']=mapping.get(r['t'],r['t'])
            grid.import_metadata['branch_records']=rows
    parameters=grid.import_metadata.get('parameters',{})
    settings=copy.copy(options) if options is not None else PowerFlowOptions(solver_type=SolverType.NR)
    iteration_value=parameters.get(phase+'MAXITER',settings.max_iter) if options is None else settings.max_iter
    try:iteration_count=float(iteration_value)
    except (TypeError,ValueError) as error:
        raise ValueError('Número de iterações efetivo precisa ser inteiro positivo e finito.') from error
    if isinstance(iteration_value,bool) or not math.isfinite(iteration_count) or iteration_count<1 or not iteration_count.is_integer():
        raise ValueError('Número de iterações efetivo precisa ser inteiro positivo e finito; não é truncado.')
    settings.max_iter=int(iteration_count)
    if options is None:
        settings.tolerance=float(parameters.get(phase+'MVATOL',settings.tolerance))
        settings.use_stored_guess=not parameters.get(phase+'FLTSTRT',False)
        delay=parameters.get(phase+'GENVARIT',0)
        settings.control_Q=delay < settings.max_iter
        settings.control_remote_voltage=bool(grid.import_metadata.get('remote_voltage_controls'))
        settings.control_taps_modules=parameters.get(phase+'BLKTAP',True) is False
        settings.control_taps_phase=parameters.get(phase+'BLKSHFT',True) is False
        settings.control_discrete_shunts=parameters.get(phase+'BLKSHT',True) is False
    if not math.isfinite(settings.tolerance) or settings.tolerance<=0 or settings.max_iter<1:
        raise ValueError('Tolerância ou número de iterações PRM inválido.')
    effective={'phase':phase,'precedence':'explicit API/interface options > selected PRM > native defaults',
        'source':'explicit options' if options is not None else 'PRM/defaults',
        'control_origins':copy.deepcopy(grid.import_metadata.get('pf_control_origins',{})),
        'options':{name:getattr(getattr(settings,name),'value',getattr(settings,name))
                   for name in settings.__slots__},'commands':parameter_inventory(parameters,grid.import_metadata.get('parameter_assignments'))}
    option_commands={'MVATOL':'tolerance','MAXITER':'max_iter','FLTSTRT':'use_stored_guess',
        'GENVARIT':'control_Q','BLKTAP':'control_taps_modules','BLKSHFT':'control_taps_phase',
        'BLKSHT':'control_discrete_shunts','DSTSLACK':'distributed_slack'}
    if options is not None:
        for record in effective['commands']:
            if record['phase']==phase and record['effective_assignment']:
                suffix=record['command'][len(phase):]
                if suffix in option_commands:
                    record['execution_source']='explicit options'
                    record['overridden_by']=option_commands[suffix]
                    record['effective_value']=getattr(settings,option_commands[suffix])
    grid.import_metadata['effective_pf']=effective
    return settings


def check_terminal_auxiliaries(grid):
    """One contract for F5, control preparation and contingency copies."""
    orphan_contract=grid.import_metadata.get('orphan_terminal_shunts_contract',{})
    auxiliary_contract=(orphan_contract.get('schema_version')==1 and
        orphan_contract.get('variant')=='organon_862_unassociated_terminal_auxiliary_not_applied_v1')
    branch_codes={b.code for b in grid.get_branches_iter(add_vsc=False,add_hvdc=False)}
    unresolved=[]
    for row in grid.import_metadata.get('orphan_terminal_shunts',[]):
        code=str(row.get('code',''));parts=code.split('|')
        reverse='|'.join((parts[1],parts[0],parts[2])) if len(parts)==3 else code
        still_missing=code not in branch_codes and reverse not in branch_codes
        if (auxiliary_contract and still_missing and row.get('effective_active') is False
                and row.get('simulation_status')=='auxiliary_not_applied_missing_branch'):
            continue
        if row.get('active',True):unresolved.append(row)
    if unresolved:
        codes=', '.join(str(row.get('code','sem identificador')) for row in unresolved)
        raise ValueError('Shunt terminal ativo sem associação a ramo elétrico: '+codes+
                         '. Resolva a associação ou reimporte o STM para revisar o contrato dos registros auxiliares. O cadastro histórico foi preservado.')


def check_execution(grid,options,phase='PF'):
    """Falhas explícitas evitam apresentar um PRM parcialmente executado como equivalente."""
    from VeraGridEngine.IO.ons_ems.association import check_association_execution
    check_association_execution(grid)
    fixed_contract=grid.import_metadata.get('fixed_shunt_representation',{})
    if fixed_contract.get('schema')!=1:
        rows=grid.import_metadata.get('source_records',{}).get('shunts',[])
        old_codes={'shunt|'+str(row[0]).strip() for row in rows if len(row)>1 and int(row[1])==0}
        legacy=[s.code for s in grid.shunts if s.code in old_codes]
        if legacy:
            raise ValueError('Representação histórica de shunt fixo STM modo0 como injeção adicional: '+
                             ', '.join(legacy)+'. Reimporte o STM para aplicar o contrato explícito BUS/SHUNTS da referência8.6.2. '
                             'O cadastro e as curvas históricos foram preservados; nenhum valor foi migrado silenciosamente.')
    check_terminal_auxiliaries(grid)
    p=grid.import_metadata.get('parameters',{})
    effective=grid.import_metadata.get('effective_pf',{})
    explicit=effective.get('phase')==phase and effective.get('source')=='explicit options'
    links={r['code']:r for r in grid.import_metadata.get('lcc_records',[])}
    from VeraGridEngine.enumerations import HvdcControlType
    for h in grid.hvdc_lines:
        if h.control_mode!=HvdcControlType.type_1_Pset or h.angle_droop:
            raise ValueError(f'Controle por ângulo não qualificado para o LCC importado: {h.code}.')
        if not math.isfinite(h.r) or h.r<0 or not math.isfinite(h.Pset):
            raise ValueError(f'Resistência ou ordem de potência CC inválida: {h.code}.')
        if h.active and (h.code not in links or not links[h.code]['active']):
            raise ValueError(f'Elo CC ativado sem modelo energizado qualificado: {h.code}. Reimporte seus registros completos; não há conversão para injeção fixa.')
        if h.active and links[h.code]['mode'] in (2,4) and h.Pset:
            raise ValueError(f'Elo em controle de corrente: Pset não representa sua ordem em A ({h.code}).')
        for field,default in [('Vset_f',1.),('Vset_t',1.),('min_firing_angle_f',-1.),('max_firing_angle_f',1.),
                              ('min_firing_angle_t',-1.),('max_firing_angle_t',1.),('loss_factor',0.)]:
            if h.active and getattr(h,field)!=default:
                raise ValueError(f'{h.code}: {field} pertence ao modelo HVDC genérico; o conversor STM usa seus registros de ponte, sem fallback.')
    missing={r['code'] for r in grid.import_metadata.get('missing_inactive_buses',[])}
    if any(b.active and b.code in missing for b in grid.buses) or any(
            b.active and (b.bus_from.code in missing or b.bus_to.code in missing) for b in grid.get_branches()):
        raise ValueError('Barra originalmente ausente foi ativada; forneça seus dados elétricos antes de calcular.')
    if options.distributed_slack:
        raise ValueError('Slack distribuída STM não qualificada neste adaptador.')
    from VeraGridEngine.IO.ons_ems.discrete_controls import active_requests,validate_discrete_controls
    actuators=validate_discrete_controls(grid,options,phase)

    individual_blocked=copy.deepcopy(grid.import_metadata.get('effective_discrete',{}).get('individual_blocked',[]))
    grid.import_metadata['effective_discrete']={'phase':phase,'requests':active_requests(grid,options,phase),
        'individual_blocked':individual_blocked,
        'controls':[{'code':a['code'],'kind':a['kind'],'bus':a['bus'],'source':a['source'],
                     'measurement_unit':'MW' if a['kind']=='phase' else 'pu',
                     'actuator_unit':'rad' if a['kind']=='phase' else 'pu' if a['kind']=='tap' else 'MVAr at Vm=1 pu',
                     'actuator_min':a['min'],'actuator_max':a['max'],
                     **{key:a[key] for key in ('lower_pu','upper_pu','lower_mw','upper_mw','tap_method',
                         'tap_method_source','phase_method','phase_target_policy','phase_target_fraction',
                         'measurement_terminal','error_scale') if key in a}} for a in actuators],
        'execution_status':'not_executed' if actuators else 'not_requested'}
    if p.get(phase+'BLOCKQG') is True:
        from VeraGridEngine.IO.ons_ems.stm_controls import reactive_block_selection
        selection = reactive_block_selection(grid)
        executed = grid.import_metadata.get('reactive_block_execution', {})
        if selection['bus_codes'] and not (executed.get('phase') == phase and executed.get('complete')):
            raise ValueError(f'{phase}BLOCKQG: seleção reconhecida, mas o bloqueio de Q ainda não foi aplicado.')
    for suffix in ('DSTSLACK','AREACNTRL','CHKCTRVLT','PRLTRFTAP','USEFIXSHT'):
        if p.get(phase+suffix) is True and not (explicit and suffix=='DSTSLACK' and not options.distributed_slack):
            raise ValueError(f'{phase+suffix}: controle importado não qualificado.')
    delay=p.get(phase+'GENVARIT',0)
    maximum=p.get(phase+'MAXITER',options.max_iter)
    if 0<delay<maximum and not explicit:
        raise ValueError(f'{phase}GENVARIT: atraso intermediário de Q não qualificado.')


def ignored_single_buses(grid,options):
    if not options.ignore_single_node_islands:return set()
    from VeraGridEngine.IO.ons_ems.pf_results import ac_branches
    connected={b for branch in ac_branches(grid) if branch.active and branch.bus_from.active and branch.bus_to.active
               for b in (branch.bus_from,branch.bus_to)}
    return {b.code for b in grid.buses if b.active and b not in connected}


def check_remote_policy(controls,trace,enforce_limits):
    if enforce_limits or not trace:return
    names={c['name'] for c in controls if c['kind']=='generator'}
    limited=[r['name'] for r in trace[-1]['controls'] if r['name'] in names and r['limited']]
    if limited:raise ValueError('Desligamento dos limites de Q remotos não qualificado; controles saturados: '+', '.join(limited))


def check_low_voltage(grid,voltage,phase='PF',ignored_codes=()):
    threshold=grid.import_metadata.get('parameters',{}).get(phase+'VTHRS',0)
    if not threshold:return
    vm={b:abs(v) for b,v in zip(grid.buses,voltage)}
    bad=[l.code for l in grid.loads if l.active and l.bus.active and l.bus.code not in ignored_codes and vm[l.bus]<threshold and
         any((l.P,l.Q,l.Ir,l.Ii)) and not l.code.startswith('lcc_work|')]
    if bad:raise ValueError(f'{phase}VTHRS: transição CP/I para Z necessária e não qualificada: {bad}.')


def apply_load_model(grid,phase='PF'):
    parameters=grid.import_metadata.get('parameters',{})
    modes=[int(parameters.get(phase+suffix,0)) for suffix in ('MWLDMDL','MVARLDMDL')]
    if any(m not in range(4) for m in modes):raise ValueError('Modelo de carga PRM inválido.')
    original={x['code']:x for x in grid.import_metadata.get('execution_load_models',grid.import_metadata.get('original_load_models',[]))}
    for load in grid.loads:
        if load.code not in original:continue
        row=original[load.code];v=row['voltage_pu']
        for sign,mode,fields in [(1,modes[0],('P','Ir','G')),(-1,modes[1],('Q','Ii','B'))]:
            p,i,z=(row[f] for f in fields)
            if mode:
                total=p+sign*i*v+sign*z*v*v
                values=(total,0.,0.) if mode==3 else (0.,sign*total/v,0.) if mode==2 else (0.,0.,sign*total/v**2)
            else:values=(p,i,z)
            for f,value in zip(fields,values):setattr(load,f,value)


def normalize_pf_results(grid,result,phase='PF'):
    """Mesmos terminais e mesma métrica usados pela avaliação estática da região."""
    import numpy as np
    from VeraGridEngine.IO.ons_ems.pf_results import normalized_flows,ac_branches
    records=grid.import_metadata.get('branch_records',[])
    sf,st,by_code=normalized_flows(grid,result,{'branch_records':records})
    buses={b:i for i,b in enumerate(grid.buses)}
    p=grid.import_metadata.get('parameters',{})
    current=bool(p.get(phase+'CURRATING',False));level=int(p.get(phase+'OVRLDLV',1))
    if level not in (1,2,3):raise ValueError('Nível térmico PRM deve ser 1, 2 ou 3.')
    for index,branch in enumerate(ac_branches(grid)):
        row=by_code.get(branch.code,{})
        rate=float(branch.rate if level==1 else branch.rate*branch.contingency_factor if level==3 else row.get('rate2',branch.rate))
        vf,vt=abs(result.voltage[buses[branch.bus_from]]),abs(result.voltage[buses[branch.bus_to]])
        values=[abs(sf[index]),abs(st[index])]
        if current:values=[values[0]/max(vf,1e-12),values[1]/max(vt,1e-12)]
        result.loading[index]=max(values)/rate if rate else 0.
        vf_complex=result.voltage[buses[branch.bus_from]];vt_complex=result.voltage[buses[branch.bus_to]]
        result.If[index]=np.conj(sf[index]/(grid.Sbase*vf_complex)) if abs(vf_complex)>1e-12 else 0j
        result.It[index]=np.conj(st[index]/(grid.Sbase*vt_complex)) if abs(vt_complex)>1e-12 else 0j
    result.Sf[:],result.St[:]=sf,st
    result.losses[:]=sf+st
    grid.import_metadata['effective_thermal']={'phase':phase,'metric':'current' if current else 'MVA',
                                             'level':level,'terminals':'maximum of both physical terminals'}
    check_dc_operational_limits(grid,result)
    effective=grid.import_metadata.get('effective_discrete',{})
    discrete=grid.import_metadata.get('discrete_last_solution',{})
    if effective.get('controls'):
        if discrete.get('phase')==phase:
            effective.update(execution_status=discrete.get('status','unknown'),
                controls_complete=bool(discrete.get('complete',False)),ac_converged=bool(result.converged),
                evaluations=discrete.get('evaluations'),resolved_controls=copy.deepcopy(discrete.get('controls',[])))
        else:
            effective.update(execution_status='not_executed',controls_complete=False,ac_converged=bool(result.converged))


def set_hvdc_operational_limit(grid, code, value, unit='MW', origin='', scope='pole', identifier=None):
    """Limite do estudo, independente da potência nominal do equipamento.

    A restrição não muda a ordem do conversor para obter convergência. O fluxo
    resolve o estado solicitado e registra violações; a região usa a mesma
    potência máxima em módulo entre os dois terminais para classificá-lo.
    """
    code=str(code)
    if code not in {str(h.code) for h in grid.hvdc_lines}:
        raise ValueError(f'Elo/polo CC inexistente: {code}.')
    if unit != 'MW':
        raise ValueError('Limite de potência CC deve ser informado em MW; não é corrente nem potência nominal.')
    if scope not in ('pole','link'):
        raise ValueError('Cadastre o limite de cada polo/elo. Limite agregado de conjunto exige composição e regra explícitas.')
    records=grid.import_metadata.setdefault('hvdc_operational_limits',{})
    if value is None:
        records.pop(code,None)
    else:
        value=float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Limite operativo CC deve ser positivo e finito.')
        if not str(origin).strip():
            raise ValueError('Informe a origem do limite operativo CC.')
        records[code]={'value':value,'unit':unit,'origin':str(origin).strip(),
            'equipment_code':code,'scope':scope,'identifier':str(identifier or code),
            'metric':'maximum_absolute_terminal_power'}
    previous=grid.import_metadata.get('effective_dc_limits')
    if previous is not None:previous['valid_for_current_limits']=False
    grid.import_metadata['hvdc_limits_revision']=grid.import_metadata.get('hvdc_limits_revision',0)+1
    return hvdc_operational_limit(grid,code)


def hvdc_operational_limit(grid, code, explicit_limits=None):
    """Prioridade: limite explícito do estudo, cadastro incorporado, ausente."""
    code=str(code)
    explicit_limits=explicit_limits or {}
    if code in explicit_limits:
        row={'value':explicit_limits[code],'unit':'MW','origin':'Limite explícito da região',
             'equipment_code':code,'scope':'pole','identifier':code}
    else:
        row=copy.deepcopy(grid.import_metadata.get('hvdc_operational_limits',{}).get(code,{}))
    value=row.get('value')
    if value is not None:
        value=float(value)
        if not math.isfinite(value) or value <= 0 or row.get('unit','MW') != 'MW':
            raise ValueError(f'Limite operativo CC inválido para {code}.')
    row.update(value=value,unit='MW',equipment_code=code,defined=value is not None)
    row.setdefault('origin','Não cadastrado');row.setdefault('scope','pole');row.setdefault('identifier',code)
    return row


def check_dc_operational_limits(grid,result):
    """Diagnóstico operativo não é o status de convergência do Newton."""
    import numpy as np
    checks=[]
    for index, link in enumerate(grid.hvdc_lines):
        record=hvdc_operational_limit(grid,link.code)
        power=max(abs(float(result.Pf_hvdc[index])),abs(float(result.Pt_hvdc[index])))
        loading=power/record['value'] if record['defined'] else None
        active=bool(link.active and link.bus_from.active and link.bus_to.active)
        status='inactive' if not active else ('missing_limit' if loading is None else 'violated' if loading > 1. else 'within_limit')
        checks.append(dict(record,power_mw=power,loading_pu=loading,status=status))
        result.loading_hvdc[index]=loading if loading is not None else (0. if not active else np.nan)
    complete=all(r['status']!='missing_limit' for r in checks)
    grid.import_metadata['effective_dc_limits']={'metric':'maximum_absolute_terminal_power','unit':'MW',
        'checks':checks,'complete':complete,'violated':any(r['status']=='violated' for r in checks),
        'valid_for_current_limits':True,'revision':grid.import_metadata.get('hvdc_limits_revision',0)}
    states={str(link.code):{'active':bool(link.active and link.bus_from.active and link.bus_to.active),'pf_mw':float(result.Pf_hvdc[index]),
                           'pt_mw':float(result.Pt_hvdc[index])}
            for index,link in enumerate(grid.hvdc_lines)}
    check_dc_operational_groups(grid,states)
    return checks


def set_hvdc_operational_group(grid, identifier, members, value, unit='MW', origin='',
                               scope='set', aggregation='absolute_terminal_sum'):
    """Limite agregado com composição e regra declaradas no próprio estudo."""
    identifier=str(identifier).strip()
    if not identifier:raise ValueError('Informe a identificação do conjunto CC.')
    groups=grid.import_metadata.get('hvdc_operational_groups',{})
    if value is None:
        groups.pop(identifier,None)
    else:
        members=[str(code).strip() for code in members]
        if not members or len(set(members))!=len(members):
            raise ValueError('Selecione polos/elos distintos para compor o conjunto CC.')
        unknown=set(members)-{str(h.code) for h in grid.hvdc_lines}
        if unknown:raise ValueError('Membro CC inexistente: '+', '.join(sorted(unknown))+'.')
        if scope not in ('link','set'):raise ValueError('Escopo agregado deve ser elo ou conjunto.')
        if aggregation not in ('absolute_terminal_sum','signed_terminal_sum'):
            raise ValueError('Escolha a regra de soma em módulo ou soma algébrica por terminal.')
        value=float(value)
        if unit!='MW' or not math.isfinite(value) or value<=0:
            raise ValueError('Limite do conjunto CC deve ser positivo e finito, em MW.')
        if not str(origin).strip():raise ValueError('Informe a origem do limite do conjunto CC.')
        groups[identifier]={'identifier':identifier,'scope':scope,'members':members,
            'aggregation':aggregation,'value':value,'unit':unit,'origin':str(origin).strip()}
    grid.import_metadata['hvdc_operational_groups']=groups
    previous=grid.import_metadata.get('effective_dc_groups')
    if previous is not None:previous['valid_for_current_limits']=False
    grid.import_metadata['hvdc_limits_revision']=grid.import_metadata.get('hvdc_limits_revision',0)+1
    return copy.deepcopy(groups.get(identifier))


def check_dc_operational_groups(grid,states):
    """Cada terminal é somado separadamente; estados ausentes impedem aprovação.

    states: code -> {active, pf_mw, pt_mw}. Em soma algébrica, o sinal é o do
    terminal físico cadastrado; em soma em módulo, sentidos contrários não se
    cancelam. O maior módulo entre os dois terminais define o carregamento.
    """
    constraints=[]
    for identifier,definition in grid.import_metadata.get('hvdc_operational_groups',{}).items():
        row=copy.deepcopy(definition) if isinstance(definition,dict) else {'raw_definition':copy.deepcopy(definition)}
        row.setdefault('identifier',str(identifier))
        row.update(equipment_code=str(identifier),power_mw=None,loading_pu=None,missing_members=[])
        value=row.get('value');members=row.get('members',[]);rule=row.get('aggregation')
        valid=(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>0 and
               row.get('unit')=='MW' and row.get('scope') in ('link','set') and
               bool(str(row.get('origin','')).strip()) and isinstance(members,list) and bool(members) and
               all(isinstance(code,str) and bool(code.strip()) for code in members) and
               len(set(members))==len(members) and
               rule in ('absolute_terminal_sum','signed_terminal_sum'))
        if not valid:
            row['status']='invalid_definition';constraints.append(row);continue
        pf=[];pt=[];active=[]
        for code in members:
            state=states.get(str(code))
            if not isinstance(state,dict) or not isinstance(state.get('active'),bool):
                row['missing_members'].append(str(code));continue
            if not state['active']:
                pf.append(0.);pt.append(0.);continue
            active.append(str(code))
            try:
                f=float(state['pf_mw']);t=float(state['pt_mw'])
                if not math.isfinite(f) or not math.isfinite(t):raise ValueError()
            except (KeyError,TypeError,ValueError):
                row['missing_members'].append(str(code));continue
            pf.append(abs(f) if rule=='absolute_terminal_sum' else f)
            pt.append(abs(t) if rule=='absolute_terminal_sum' else t)
        row['active_members']=active
        if row['missing_members']:
            row['status']='missing_state'
        else:
            power=max(abs(math.fsum(pf)),abs(math.fsum(pt)))
            row.update(power_mw=power,loading_pu=power/value,
                       terminal_from_mw=math.fsum(pf),terminal_to_mw=math.fsum(pt),
                       status='inactive' if not active else 'violated' if power>value else 'within_limit')
        constraints.append(row)
    missing=any(r['status'] in ('invalid_definition','missing_state') for r in constraints)
    result={'constraints':constraints,'missing':missing,'complete':not missing,
            'violated':any(r['status']=='violated' for r in constraints),
            'valid_for_current_limits':True,'revision':grid.import_metadata.get('hvdc_limits_revision',0)}
    grid.import_metadata['effective_dc_groups']=result
    return result
