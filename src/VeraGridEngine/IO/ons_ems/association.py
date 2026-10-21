# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Associação explícita das tabelas EMS e planejamento, sem busca por nomes parecidos."""
from __future__ import annotations
import hashlib
import math
import re
from pathlib import Path
import shlex


def _read(path):
    raw=Path(path).read_bytes()
    try:text=raw.decode('utf-8-sig');encoding='utf-8-sig' if raw.startswith(b'\xef\xbb\xbf') else 'utf-8'
    except UnicodeDecodeError:text=raw.decode('cp1252');encoding='cp1252'
    return text,dict(source=Path(path).name,sha256=hashlib.sha256(raw).hexdigest(),text=text,encoding=encoding)


def _embedded_read(item):
    """Reutiliza a tabela escolhida e incorporada, verificando seus bytes por hash."""
    text=item['text'];expected=item['sha256']
    for encoding in dict.fromkeys((item.get('encoding','utf-8'),'utf-8','utf-8-sig','cp1252')):
        try:raw=text.encode(encoding)
        except (UnicodeEncodeError,LookupError):continue
        if hashlib.sha256(raw).hexdigest()==expected:
            return text,dict(item,encoding=encoding)
    raise ValueError(f'Tabela incorporada {item.get("source","?")}: conteúdo não corresponde ao hash selecionado.')


def _read_ems_blocks(text,source,result):
    """END encerra um bloco, não a tabela inteira do formato encontrado no acervo."""
    kinds=('conventional_generators','renewable_generators',
           'synchronous_high_voltage_associations','dynamic_pole_associations')
    blocks=[];block=0;summary=None
    result['ems_blocks']=blocks
    result['synchronous_associations']=[]
    result['pole_associations']=[]
    result['generator_identifiers']={}
    result['generator_identifier_groups']=[]
    lines=text.splitlines(keepends=True)
    for number,line in enumerate(lines,1):
        content=line.split('!',1)[0].strip()
        if not content or content.startswith(('/','#')):continue
        if content.upper()=='END':
            if summary is not None:summary['end_line']=number
            block+=1;summary=None
            continue
        if block and summary is None and _ems_footer_prose(content):
            # O leitor anterior parava no END. Preservar prosa não autoriza
            # ocultar um bloco: qualquer cabeçalho/campo numérico posterior
            # exige um contrato de tabela e continua sendo erro de leitura.
            for offset,raw in enumerate(lines[number-1:],number):
                tail=raw.split('!',1)[0].strip()
                if not tail or tail.startswith(('/','#')):continue
                if (tail.upper()=='END' or tail.startswith('(')
                        or any(c.isdigit() or c in '=,()' for c in tail)):
                    raise ValueError(f'EMSTable, linha {offset}: rodapé textual seguido de registro/cabeçalho não reconhecido.')
            result['ems_trailing_text']={'start_line':number,'after_block':block,
                'text':''.join(lines[number-1:]),'source':source['source'],
                'source_sha256':source['sha256'],'status':'legacy trailing text; metadata only; not executed'}
            break
        if block>=len(kinds):
            raise ValueError(f'EMSTable, linha {number}: bloco adicional sem contrato reconhecido.')
        if summary is None:
            summary={'index':block+1,'kind':kinds[block],'start_line':number,
                     'end_line':None,'headers':[],'records':0,
                     'use':'bus association' if block<2 else 'metadata only; not executed'}
            blocks.append(summary)
        if content.startswith('('):
            summary['headers'].append({'line':number,'text':content});continue
        fields=shlex.split(content)
        if len(fields)==1 and fields[0].isdigit():
            if block!=0 or summary['records'] or 'ems_format_version' in result:
                raise ValueError(f'EMSTable, linha {number}: cabeçalho de formato fora do primeiro bloco.')
            result['ems_format_version']=int(fields[0]);continue
        origin={'line':number,'block':block+1,'block_kind':kinds[block],
                'source':source['source'],'source_sha256':source['sha256']}
        if block<2:
            if len(fields) not in (2,3) or not fields[1].isdigit():
                raise ValueError(f'EMSTable, linha {number}: registro de associação de barra não reconhecido no bloco {block+1}.')
            name,code=fields[:2]
            try:units=int(fields[2]) if len(fields)==3 else None
            except ValueError as exc:
                raise ValueError(f'EMSTable, linha {number}: quantidade inválida para {name}.') from exc
            if units is not None and units<=0:raise ValueError(f'EMSTable: quantidade inválida para {name}.')
            value={'bus':str(int(code)),'equivalent_units':units,**origin}
            # The conventional table in the supplied decks also records an
            # explicit generator identifier and its quantity in the comment:
            # ! <legacy bus>, [<identifier>(<quantity>), ...]
            # Temporary STM bus names/numbers do not erase that identity.
            # Retain candidates, including conflicts; never match prefixes,
            # similar plant names or just electrical/geographic proximity.
            if block==0 and '!' in line:
                identity=re.fullmatch(r'\s*(\d+)\s*,\s*\[([^\]]+)\]\s*',line.split('!',1)[1].strip())
                if identity:
                    legacy_bus,body=identity.groups()
                    parsed=[re.fullmatch(r'\s*([A-Za-z0-9_:.-]+)\((\d+)\)\s*',item)for item in body.split(',')]
                    if parsed and all(parsed):
                        identifiers=[item[1]for item in parsed]
                        quantities=[int(item[2])for item in parsed]
                        complete=(len(set(identifiers))==len(identifiers)and all(q>0 for q in quantities)
                                  and units is not None and sum(quantities)==units)
                        group_record=dict(value,table_bus_name=name,legacy_bus=legacy_bus,
                            identifiers=identifiers,quantities=quantities,
                            comment_quantity_total=sum(quantities),comment_group_complete=complete,
                            raw=line.rstrip('\r\n'))
                        result['generator_identifier_groups'].append(group_record)
                        for identifier,quantity in zip(identifiers,quantities):
                            result['generator_identifiers'].setdefault(identifier,[]).append(dict(
                                value,table_bus_name=name,legacy_bus=legacy_bus,
                                source_identifier=identifier,comment_quantity=quantity,
                                comment_quantity_total=sum(quantities),comment_group_complete=complete,
                                raw=line.rstrip('\r\n')))
            if name not in result['ems']:
                result['ems'][name]=dict(value,entries=[dict(value)])
            else:
                previous=result['ems'][name];previous['entries'].append(value)
                if len({e['bus'] for e in previous['entries']})>1:
                    previous['candidates']=list(previous['entries'])
                if len({e['equivalent_units'] for e in previous['entries']})>1:
                    previous['equivalent_units']=None
        elif block==2:
            if len(fields)!=5:
                raise ValueError(f'EMSTable, linha {number}: associação síncrona/alta exige cinco campos.')
            try:group=int(fields[2]);lower=float(fields[3]);upper=float(fields[4])
            except ValueError as exc:
                raise ValueError(f'EMSTable, linha {number}: grupo/limites da associação síncrona inválidos.') from exc
            if not math.isfinite(lower) or not math.isfinite(upper) or lower>upper:
                raise ValueError(f'EMSTable, linha {number}: faixa da associação síncrona inválida.')
            result['synchronous_associations'].append({'bus_name':fields[0],
                'high_voltage_bus_name':fields[1],'group':group,'lower':lower,'upper':upper,
                'status':'metadata only; not executed',**origin})
        else:
            if len(fields)!=3:
                raise ValueError(f'EMSTable, linha {number}: associação de polo DYN exige três campos.')
            result['pole_associations'].append({'from_identifier':fields[0],
                'to_identifier':fields[1],'dynamic_pole':fields[2],
                'status':'metadata only; not executed',**origin})
        summary['records']+=1


def _ems_footer_prose(content):
    """Somente prosa sem campos estruturais pode seguir um END legado."""
    return (len(content.split())>=3 and any(c.isalpha() for c in content)
            and not any(c.isdigit() or c in '=,()' for c in content))


def read_auxiliaries(ems_path=None,planning_path=None,*,embedded_files=None):
    """Conteúdo e hashes ficam no estudo. Não há procura na pasta do programa."""
    if embedded_files is not None and (ems_path is not None or planning_path is not None):
        raise ValueError('Escolha arquivos externos ou tabelas incorporadas explicitamente, sem misturar origens.')
    embedded_files=embedded_files or {}
    result={'ems':{},'planning':{},'files':{}}
    if ems_path or 'EMSTable' in embedded_files:
        text,result['files']['EMSTable']=_embedded_read(embedded_files['EMSTable']) if 'EMSTable' in embedded_files else _read(ems_path)
        _read_ems_blocks(text,result['files']['EMSTable'],result)
    if planning_path or 'PlanningKB' in embedded_files:
        text,result['files']['PlanningKB']=_embedded_read(embedded_files['PlanningKB']) if 'PlanningKB' in embedded_files else _read(planning_path)
        parent=None
        for number,line in enumerate(text.splitlines(),1):
            content=line.split('!',1)[0].strip()
            if not content or content.startswith(('(','/','#')):continue
            if content.split()[0]=='-9999':break
            fields=shlex.split(content)
            if len(fields) in (8,9):
                code=str(int(fields[0]))
                if code in result['planning']:raise ValueError(f'PlanningKB: barra repetida {code}.')
                parent={'bus':code,'name':fields[1],'type':int(fields[2]),'area':int(fields[3]),
                        'voltage_attribute':float(fields[4]),'mw_attribute':float(fields[5]),
                        'unit_size_attribute':float(fields[6]),'neighbors_expected':int(fields[7]),
                        'coincident_bus':fields[8] if len(fields)==9 else '0','neighbors':[],'line':number}
                result['planning'][code]=parent
            elif len(fields)==5 and parent is not None:
                parent['neighbors'].append({'bus':str(int(fields[0])),'name':fields[1],
                    'circuit':fields[2],'type':int(fields[3]),'impedance_attribute':float(fields[4])})
            else:raise ValueError(f'PlanningKB, linha {number}: registro não reconhecido.')
        for row in result['planning'].values():
            if len(row['neighbors'])!=row['neighbors_expected']:
                raise ValueError(f'PlanningKB: vizinhança incompleta da barra {row["bus"]}.')
    return result


def _unit_quantity_evidence(identifier,source_bus,destination,auxiliaries):
    """Use a complete exact-identity list; a bus total is never a unit count."""
    candidates=[r for r in auxiliaries.get('generator_identifiers',{}).get(identifier,())
                if r['legacy_bus']==source_bus]
    if not candidates or any(not r.get('comment_group_complete')for r in candidates):return None
    signatures={(r['bus'],r['comment_quantity'])for r in candidates}
    if len(signatures)!=1 or next(iter(signatures))[0]!=destination:return None
    return dict(schema_version=1,quantity=next(iter(signatures))[1],source_identifier=identifier,
                legacy_bus=source_bus,planning_bus=destination,
                rule='Exact STM generator identifier and legacy bus in a complete positive-quantity EMSTable list; list sum equals declared bus total.',
                entries=candidates,electrical_network_changed=False)


def apply_association(grid,auxiliaries,policy='sage'):
    """Identificação não prova equivalência. A redução AC tem domínio explícito.

    planning_sections preserva a rede elétrica inteira, inclusive seções que
    apontam à mesma barra de planejamento. planning_merge é uma transformação
    explícita e restrita; não aplica heurísticas nem remove ligações internas.
    """
    if policy not in ('sage','planning_sections','planning_merge'):
        raise ValueError('Política de associação não reconhecida.')
    if policy!='sage' and not auxiliaries['files']:
        raise ValueError('Selecione tabelas auxiliares para converter a identificação.')
    metadata=grid.import_metadata
    from VeraGridEngine.IO.ons_ems.stm_parser import bus_id
    unit_sources={f'{bus_id(r[0])}|{r[1]}':r for r in metadata.get('source_records',{}).get('generators',[])}
    generators_by_bus={}
    for generator in grid.generators:generators_by_bus.setdefault(generator.bus,[]).append(generator)
    metadata['association']={'schema_version':4,'policy':policy,'auxiliaries':auxiliaries,'buses':[],
                             'units':[],'circuits':[],'groups':[],'poles':[],'devices':[],
                             'aggregation_contracts':[], 'diagnostics':[],
                             'schema_note':'Schema4 separates complete exact-identifier per-generator quantities from bus totals. Older saved mappings remain historical and readable; reimport STM with the selected embedded/original tables to build this evidence. No automatic remapping of an opened case.'}
    report=metadata['association'];destinations={};groups={};ems_by_bus={}
    for bus in grid.buses:
        ems=auxiliaries['ems'].get(bus.name.strip())
        if ems and ems.get('candidates'):
            report['diagnostics'].append(f'{bus.code}: destinos conflitantes para {bus.name.strip()}; associação pendente por unidade.')
            if policy!='sage':
                raise ValueError(f'EMSTable: destinos conflitantes para {bus.name.strip()}; forneça associação explícita por unidade.')
            ems=None
        direct=auxiliaries['planning'].get(str(bus.code).split('.')[0])
        identity_evidence=None
        if not ems and not direct and len(generators_by_bus.get(bus,()))==1:
            generator=generators_by_bus[bus][0];source=unit_sources.get(generator.code)
            identifier=source[26].strip() if source and len(source)>26 else ''
            candidates=auxiliaries.get('generator_identifiers',{}).get(identifier,[])
            if candidates:
                # This fallback qualifies one individual unit only. Conflicts
                # remain unresolved for SAGE identity; a requested planning
                # conversion cannot choose a destination arbitrarily.
                signatures={(v['bus'],v.get('equivalent_units'),v['comment_quantity']) for v in candidates}
                if len(signatures)!=1 or next(iter(signatures))[1:]!=(1,1):
                    message=(f'{bus.code}: identificador de gerador {identifier} com destinos/quantidades '
                             'incompatíveis na EMSTable, linhas '+', '.join(str(v['line']) for v in candidates)+'.')
                    report['diagnostics'].append(message)
                    if policy!='sage':raise ValueError(message)
                else:
                    ems=dict(candidates[0]);identity_evidence={
                        'source_identifier':identifier,'STM_generator_code':generator.code,
                        'source':'EMSTable conventional-generator explicit identifier comment',
                        'entries':candidates,'quantity':1,'match':'exact; one generator at this bus',
                        'electrical_network_changed':False,'dynamic_aggregation_qualified':False}
        ems_by_bus[bus]=ems or {}
        destination=ems['bus'] if ems else direct['bus'] if direct else str(bus.code)
        method=('EMSTable exact generator identifier' if identity_evidence else
                'EMSTable exact' if ems else 'PlanningKB direct' if direct else 'unchanged')
        destinations[bus]=destination
        groups.setdefault(destination,[]).append(bus)
        report['buses'].append({'original':str(bus.code),'original_name':bus.name,
            'section':str(bus.code).partition('.')[2] or '0','planning_bus':destination,
            'converted':str(bus.code),'method':method,'equivalent_units':ems.get('equivalent_units') if ems else None,
            'generator_identifier_evidence':identity_evidence})
        knowledge=auxiliaries['planning'].get(destination)
        if knowledge and knowledge['voltage_attribute']<900 and not math.isclose(bus.Vnom,knowledge['voltage_attribute'],rel_tol=.01):
            report['diagnostics'].append(f'{bus.code}: tensão {bus.Vnom} kV difere do atributo {knowledge["voltage_attribute"]} da tabela.')
    original={bus:str(bus.code) for bus in grid.buses}
    for g in grid.generators:
        source=unit_sources.get(g.code)
        ems=ems_by_bus[g.bus]
        quantity_evidence=(_unit_quantity_evidence(source[26].strip(),bus_id(source[0]),destinations[g.bus],auxiliaries)
                           if source and len(source)>26 else None)
        report['units'].append({'original':g.code,'original_bus':original[g.bus],
            'original_group':source[16] if source else None,'original_unit':g.code.partition('|')[2],'converted':g.code,
            'equivalent_units':ems.get('equivalent_units'),
            'quantity_per_generator':quantity_evidence['quantity']if quantity_evidence else None,
            'quantity_evidence':quantity_evidence,
            'planning_bus':destinations[g.bus],'p_mw':g.P,'pmin_mw':g.Pmin,'pmax_mw':g.Pmax,
            'qmin_mvar':g.Qmin,'qmax_mvar':g.Qmax,'snom_mva':g.Snom,
            'status':g.active,'bus_status':g.bus.active,'dispatch_available':g.enabled_dispatch,
            'reference':g.bus.is_slack,'control_mode':str(g.control_mode),
            'control_original_bus':original.get(g.control_bus or g.bus),
            'rms_model_applied':bool(getattr(g.rms_model,'children',[])),
            'source_generator_identifier':source[26] if source and len(source)>26 else None,
            'stepup_parameters_metadata_only':{'r_pu':float(source[9]),'x_pu':float(source[10]),'tap':float(source[11])} if source else None,
            'dynamic_association':'individual original bus/group; no dynamic aggregation'})
    for b in grid.get_branches(add_hvdc=True,add_vsc=True):
        report['circuits'].append({'original':b.code,'converted':b.code,
            'from_original':original[b.bus_from],'to_original':original[b.bus_to],
            'from_planning':destinations[b.bus_from],'to_planning':destinations[b.bus_to],
            'r_pu':getattr(b,'R',None),'x_pu':getattr(b,'X',None),
            'tap':getattr(b,'tap_module',None),'rate_mva':b.rate,
            'active':b.active,'idtag':b.idtag,'kind':str(b.device_type),
            'contingency_factor':b.contingency_factor,
            'control_original_bus':original.get(getattr(b,'control_bus',None)),
            'units':{'r_pu':'pu','x_pu':'pu','rate_mva':'MVA','tap':'pu'}})
    planned_codes={}
    if policy=='planning_merge':
        planned_codes={bus:destination for destination,members in groups.items() for bus in members}
    elif policy=='planning_sections':
        for destination,members in groups.items():
            members.sort(key=lambda bus:original[bus])
            planned_codes.update({bus:destination if len(members)==1 else f'{destination}.{index+1}'
                                  for index,bus in enumerate(members)})
        if len(set(planned_codes.values()))!=len(planned_codes):
            raise ValueError('Conversão geraria códigos de barra repetidos; identifique explicitamente as seções conflitantes.')
    if policy=='planning_merge':
        # A identificação comum não prova equipotencialidade.
        for destination,buses in groups.items():
            contract=_qualify_merge(grid,destination,buses,original)
            if contract is not None:report['aggregation_contracts'].append(contract)
        replacements={b:members[0] for members in groups.values() for b in members}
        for device in grid.get_all_elements_iter():
            for field in ('bus','bus_from','bus_to','control_bus','regulation_bus'):
                value=getattr(device,field,None)
                if value in replacements:setattr(device,field,replacements[value])
        grid.buses=[members[0] for members in groups.values()]
        for destination,members in groups.items():members[0].code=destination
    elif policy=='planning_sections':
        for bus,code in planned_codes.items():bus.code=code
    converted={original[b]:b.code if b in grid.buses else groups[destinations[b]][0].code for b in original}
    for item in report['buses']:item['converted']=converted[item['original']]
    for item in report['units']:item['converted_bus']=converted[item['original_bus']]
    for item in report['circuits']:
        item['from_converted']=converted[item['from_original']];item['to_converted']=converted[item['to_original']]
    for destination,members in groups.items():
        report['groups'].append({'planning_bus':destination,'original_buses':[original[b] for b in members],
            'aggregation':policy=='planning_merge' and len(members)>1})
    for item in report['units']:
        item['control_converted_bus']=converted.get(item['control_original_bus'])
    for item in report['circuits']:
        item['control_converted_bus']=converted.get(item['control_original_bus'])
    for contract in report['aggregation_contracts']:
        for row in contract['branches']:
            row['outside_converted']=converted[row['outside_original']]
    missing={row['code']:row for row in metadata.get('missing_inactive_buses',[])}
    for row in report['buses']:
        row['in_source']=row['original'] not in missing
        row['activation_qualified']=row['original'] not in missing
        if row['original'] in missing:
            row['missing_reason']=missing[row['original']]['reason']
    _complete_shared_map(grid,report,original,converted)
    # Metadata follows electrical references; origin records remain untouched.
    for rec in metadata.get('lcc_records',[]):
        rec['rectifier'][0]=converted[rec['rectifier'][0].strip()]
        rec['inverter'][0]=converted[rec['inverter'][0].strip()]
    if len(set(str(b.code) for b in grid.buses))!=len(grid.buses):
        raise ValueError('Conversão gerou códigos de barra repetidos.')
    check_association_execution(grid)
    return report

def _close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)


def _control_mode(device):
    mode=getattr(device,'control_mode','')
    return str(getattr(mode,'value',mode))


def _injection_devices(grid):
    return {(kind,str(device.code)):device for collection,kind in
        ((grid.generators,'generator'),(grid.loads,'load'),(grid.shunts,'shunt'),
         (grid.controllable_shunts,'controlled_shunt')) for device in collection}


def _member_injections(grid,buses,original):
    entries=[]
    for (kind,code),device in _injection_devices(grid).items():
        if device.bus not in buses:continue
        entries.append({'kind':kind,'code':code,'original_bus':original[device.bus],
            'active':bool(device.active),'dispatch_available':bool(getattr(device,'enabled_dispatch',False)),
            'control_mode':_control_mode(device),
            'control_bus':original.get(getattr(device,'control_bus',None)),
            'vset_pu':getattr(device,'Vset',None)})
    return entries


def _injection_signature(entries,devices,member,weight):
    result={}
    for item in entries:
        if item['original_bus']!=member or not item['active']:continue
        device=devices[(item['kind'],item['code'])]
        key=(item['kind'],item['control_mode'])
        values=result.setdefault(key,{})
        for field in ('P','Q','Ir','Ii','G','B','Snom','Pmin','Pmax','Qmin','Qmax','Bmin','Bmax'):
            if not hasattr(device,field):continue
            # O Q cadastrado de uma máquina PV é chute, não injeção imposta.
            if field=='Q' and item['kind']=='generator' and item['control_mode']=='V':continue
            values[field]=values.get(field,0.)+float(getattr(device,field))/weight
    return result


def _same_signatures(a,b):
    return a.keys()==b.keys() and all(a[k].keys()==b[k].keys() and
        all(_close(a[k][field],b[k][field]) for field in a[k]) for k in a)


def _qualify_merge(grid,destination,buses,original):
    """Redução AC exata de folhas homotéticas, com todos os ramos individuais.

    Para Y_i=w_i Y_0 e I_i(V)=w_i I_0(V), o mesmo V satisfaz cada KCL
    individual e o KCL somado. A igualdade deve continuar verdadeira depois
    de qualquer edição ou contingência. Nenhum transformador é substituído
    por impedância ajustada e nenhuma unidade é fundida.
    """
    if len(buses)<2:return None
    first=buses[0]
    for bus in buses[1:]:
        if any(getattr(bus,key)!=getattr(first,key) for key in
               ('Vnom','active','is_slack','Vm0','Va0','Vmin','Vmax')):
            raise ValueError(f'Agregação {destination}: atributos elétricos diferentes; use seções preservadas.')
    if not first.active or first.is_slack:
        raise ValueError(f'Agregação {destination}: seções inativas ou de referência devem permanecer individuais.')
    members=set(buses);all_branches=grid.get_branches(add_hvdc=True,add_vsc=True)
    internal=[b for b in all_branches if b.bus_from in members and b.bus_to in members]
    if internal:
        codes=', '.join(str(b.code) for b in internal)
        raise ValueError(f'Agregação {destination}: circuito interno {codes} requer equivalente qualificado.')
    entries=_member_injections(grid,members,original)
    if any(x['kind']=='controlled_shunt' for x in entries):
        raise ValueError(f'Agregação {destination}: shunt controlado requer qualificação individual.')
    for generator in grid.generators:
        if generator.bus not in members:continue
        if getattr(generator.rms_model,'children',[]):
            raise ValueError(f'Agregação {destination}: modelo RMS aplicado em {generator.code}; preserve seções.')
        if generator.active and generator.control_bus not in (None,generator.bus):
            raise ValueError(f'Agregação {destination}: controle remoto de {generator.code} requer qualificação individual.')
    targets={float(g.Vset) for g in grid.generators if g.bus in members and g.active and _control_mode(g)=='V'}
    if len(targets)>1:
        raise ValueError(f'Agregação {destination}: controles de geração incompatíveis.')
    devices=_injection_devices(grid);branches=[];weights={};outside=None;reference=None
    for bus in buses:
        linked=[b for b in all_branches if b.bus_from is bus or b.bus_to is bus]
        if len(linked)!=1 or not linked[0].active:
            raise ValueError(f'Agregação {destination}: seção {original[bus]} não é folha energizada com ramo único.')
        branch=linked[0]
        if branch not in grid.lines and branch not in grid.transformers2w:
            raise ValueError(f'Agregação {destination}: tipo de ramo {branch.code} não qualificado.')
        remote=branch.bus_to if branch.bus_from is bus else branch.bus_from
        if outside is not None and remote is not outside:
            raise ValueError(f'Agregação {destination}: seções ligam barras externas diferentes; não são equipotenciais.')
        outside=remote;impedance=complex(branch.R,branch.X)
        if impedance==0:
            raise ValueError(f'Agregação {destination}: ramo ideal {branch.code} exige redução topológica própria.')
        signature={'kind':'transformer' if branch in grid.transformers2w else 'line',
            'side':'from' if branch.bus_from is bus else 'to',
            'tap_module':float(getattr(branch,'tap_module',1.)),
            'tap_phase':float(getattr(branch,'tap_phase',0.)),
            'G':float(getattr(branch,'G',0.)),'B':float(getattr(branch,'B',0.))}
        if reference is None:reference=(impedance,signature)
        ratio=reference[0]/impedance
        if ratio.real<=0 or abs(ratio.imag)>1e-10*max(1.,abs(ratio.real)):
            raise ValueError(f'Agregação {destination}: impedâncias não proporcionais no ramo {branch.code}.')
        weight=float(ratio.real);weights[original[bus]]=weight
        if any(signature[field]!=reference[1][field] for field in ('kind','side','tap_module','tap_phase')):
            raise ValueError(f'Agregação {destination}: tap, fase ou orientação diferente em {branch.code}.')
        if any(not _close(signature[field]/weight,reference[1][field]) for field in ('G','B')):
            raise ValueError(f'Agregação {destination}: admitância não proporcional em {branch.code}.')
        branches.append({'code':str(branch.code),'member':original[bus],'outside_original':original[remote],
            'r_pu':float(branch.R),'x_pu':float(branch.X),'active':True,**signature})
    base=_injection_signature(entries,devices,original[first],weights[original[first]])
    for bus in buses[1:]:
        if not _same_signatures(base,_injection_signature(entries,devices,original[bus],weights[original[bus]])):
            raise ValueError(f'Agregação {destination}: injeções/limites de {original[bus]} não proporcionais às admitâncias; use seções preservadas.')
    return {'planning_bus':destination,'domain':'AC folhas homotéticas; unidades e ramos individuais',
        'original_buses':[original[b] for b in buses],'weights':weights,'branches':branches,
        'devices':entries,'dynamic_qualified':False,
        'dynamic_reason':'Estados, inércias e controles exigem qualificação temporal na etapa 3.'}


def check_association_execution(grid):
    """Recusa alterações que invalidam a redução sem restaurar o cadastro antigo."""
    association=grid.import_metadata.get('association',{})
    contracts=association.get('aggregation_contracts',[])
    if not contracts:
        if any(row.get('aggregation') for row in association.get('groups',[])):
            raise ValueError('Deck agregado anterior sem contrato elétrico; reimporte com seções preservadas.')
        return
    devices=_injection_devices(grid)
    branches={str(b.code):b for b in grid.get_branches(add_hvdc=True,add_vsc=True)}
    buses={str(b.code):b for b in grid.buses}
    for contract in contracts:
        destination=contract['planning_bus'];merged=buses.get(destination)
        if merged is None or not merged.active or merged.is_slack:
            raise ValueError(f'Agregação {destination}: estado/referência mudou; reimporte com seções preservadas.')
        linked={str(b.code) for b in branches.values() if b.bus_from is merged or b.bus_to is merged}
        if linked!={r['code'] for r in contract['branches']}:
            raise ValueError(f'Agregação {destination}: conjunto de ramos mudou; preserve as seções.')
        injected={key for key,device in devices.items() if device.bus is merged}
        if injected!={(r['kind'],r['code']) for r in contract['devices']}:
            raise ValueError(f'Agregação {destination}: conjunto de injeções mudou; preserve as seções.')
        for row in contract['branches']:
            branch=branches.get(row['code'])
            if branch is None or branch.active!=row['active']:
                raise ValueError(f'Agregação {destination}: topologia do ramo {row["code"]} mudou; a redução não cobre esta contingência.')
            if any(not _close(getattr(branch,field,default),row[target]) for field,target,default in
                   (('R','r_pu',0.),('X','x_pu',0.),('G','G',0.),('B','B',0.),
                    ('tap_module','tap_module',1.),('tap_phase','tap_phase',0.))):
                raise ValueError(f'Agregação {destination}: parâmetros do ramo {row["code"]} mudaram; reimporte com seções preservadas.')
            own=branch.bus_from if row['side']=='from' else branch.bus_to
            remote=branch.bus_to if row['side']=='from' else branch.bus_from
            if own is not merged or str(remote.code)!=row['outside_converted']:
                raise ValueError(f'Agregação {destination}: terminais do ramo {row["code"]} mudaram.')
        for row in contract['devices']:
            device=devices.get((row['kind'],row['code']))
            if device is None or device.bus is not merged or bool(device.active)!=row['active']:
                raise ValueError(f'Agregação {destination}: estado/associação de {row["code"]} mudou.')
            if _control_mode(device)!=row['control_mode']:
                raise ValueError(f'Agregação {destination}: controle de {row["code"]} mudou.')
            if hasattr(device,'Vset') and not _close(device.Vset,row['vset_pu']):
                raise ValueError(f'Agregação {destination}: alvo de tensão de {row["code"]} mudou.')
            if getattr(device,'control_bus',None) not in (None,merged):
                raise ValueError(f'Agregação {destination}: controle remoto de {row["code"]} mudou.')
        members=contract['original_buses'];weights=contract['weights']
        base=_injection_signature(contract['devices'],devices,members[0],weights[members[0]])
        for member in members[1:]:
            if not _same_signatures(base,_injection_signature(contract['devices'],devices,member,weights[member])):
                raise ValueError(f'Agregação {destination}: redispatch/carga/limites romperam a proporcionalidade; use seções preservadas.')


def _complete_shared_map(grid,report,original,converted):
    """Mapa persistente reutilizável por estudos estáticos e associação RMS."""
    reverse={}
    for key,code in converted.items():reverse.setdefault(code,[]).append(key)
    report['reverse_buses']=reverse
    original_by_identity={id(bus):key for bus,key in original.items()}
    aggregated_origins={}
    for contract in report['aggregation_contracts']:
        for row in contract['devices']:
            aggregated_origins.setdefault((row['kind'],row['code']),row['original_bus'])
    for (kind,code),device in _injection_devices(grid).items():
        origin=aggregated_origins.get((kind,code))
        if origin is None:origin=original_by_identity.get(id(device.bus))
        report['devices'].append({'kind':kind,'original':code,'converted':code,
            'original_bus':origin,'converted_bus':str(device.bus.code),'idtag':device.idtag,
            'active':bool(device.active),'control_mode':_control_mode(device),
            'control_converted_bus':str(device.control_bus.code) if getattr(device,'control_bus',None) is not None else None})
    for record in grid.import_metadata.get('lcc_records',[]):
        rectifier=record['rectifier'][0].strip();inverter=record['inverter'][0].strip()
        report['poles'].append({'original':record['code'],'converted':record['code'],
            'pole':record['pole'],'mode':record['mode'],'active':record['active'],
            'rectifier_original_bus':rectifier,'inverter_original_bus':inverter,
            'rectifier_converted_bus':converted.get(rectifier,rectifier),
            'inverter_converted_bus':converted.get(inverter,inverter),
            'nominal_voltage_kv':record['nominal_voltage'],'nominal_power_mw':record['rate'],
            'operational_limit_reference':'import_metadata.hvdc_operational_limits',
            'limit_key':record['code'],'limit_policy':'not inferred from nominal power'})
    report['group_units']={}
    for row in report['units']:
        key=f'{row["original_bus"]}|{row["original_group"]}'
        report['group_units'].setdefault(key,[]).append(row['original'])
    report['dynamic_bindings']=[{'original_unit':row['original'],
        'original_bus':row['original_bus'],'original_group':row['original_group'],
        'planning_bus':row['planning_bus'],'converted_bus':row['converted_bus'],
        'converted_unit':row['converted'],'equivalent_units':row['equivalent_units'],
        'reference':row['reference'],'dispatch_available':row['dispatch_available'],
        'aggregation_qualified':False,'status':'association only; simulation qualification separate'}
        for row in report['units']]
    report['source_hashes']={key:value['sha256'] for key,value in report['auxiliaries']['files'].items()}
    report['source_hashes']['STM']=grid.import_metadata.get('sha256')
    report['electrical_qualification']={'policy':report['policy'],
        'domain':'unchanged electrical network' if not report['aggregation_contracts'] else 'AC homothetic leaf reduction',
        'generation_units_merged':False,'dynamic_aggregation_qualified':False}
