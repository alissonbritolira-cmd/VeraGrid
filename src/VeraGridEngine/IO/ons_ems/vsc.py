"""Registros VSC multiterminais: parsing rastreável não é qualificação elétrica.

Fonte: equipamentos/hvdc/data.html e equipamentos/hvdc/vsc.html, manual
consultado em 11/10/2026. Os coeficientes A/B/C e a base CC devem ter suas
unidades conferidas antes da conversão para o modelo nativo.
"""
from __future__ import annotations
import math


CONVERTER_FIELDS=('node','ac_bus','positive_status','negative_status','p_mode','p_positive',
    'p_negative','q_mode','q_set','loss_a','loss_b','loss_c','s_max_mva','i_max_ka',
    'q_max_mvar','q_min_mvar','v_max_pu','v_min_pu','reactor_r_pu','reactor_x_pu',
    'filter_b_pu','transformer_r_pu','transformer_x_pu','ac_base_kv','limit_correction',
    'grounded','series_levels','converter_type')


def records(data):
    rows=data.get('vsc',[]) if isinstance(data,dict) else data
    result=[];cursor=0
    while cursor<len(rows):
        raw=list(rows[cursor]);cursor+=1
        if len(raw)!=4:raise ValueError('VSC: cabeçalho requer ID, nome, status e tipo.')
        identifier,name,status,kind=raw
        try:status=int(status);kind=int(kind)
        except ValueError as error:raise ValueError('VSC: status/tipo inválidos.') from error
        if status not in (0,1) or kind not in (0,1):raise ValueError('VSC: status e tipo devem ser 0 ou 1.')
        link={'id':str(identifier),'name':name,'active':bool(status),'bipolar':bool(kind),
            'raw_header':raw,'converters':[],'lines':[],'simulation_status':'not_qualified'}
        for section in ('converters','lines'):
            terminated=False
            while cursor<len(rows):
                raw=list(rows[cursor]);cursor+=1
                if len(raw)==1 and raw[0]=='0':terminated=True;break
                if section=='converters':
                    if len(raw) not in (26,27,28):raise ValueError(f'VSC {identifier}: conversor requer 26 campos e N/tipo opcionais.')
                    full=raw+(['1'] if len(raw)==26 else [])
                    if len(full)==27:full.append('0')
                    row=dict(zip(CONVERTER_FIELDS,full));row['raw']=raw
                    for key in CONVERTER_FIELDS:
                        if key in ('node','ac_bus'):continue
                        try:row[key]=float(row[key])
                        except ValueError as error:raise ValueError(f'VSC {identifier}: {key} inválido.') from error
                        if not math.isfinite(row[key]):raise ValueError(f'VSC {identifier}: {key} não finito.')
                    for key in ('positive_status','negative_status','grounded','converter_type'):
                        if row[key] not in (0,1):raise ValueError(f'VSC {identifier}: {key} deve ser 0 ou 1.')
                        row[key]=int(row[key])
                    if row['p_mode'] not in (1,2) or row['q_mode'] not in (1,2):raise ValueError(f'VSC {identifier}: controle P/Q desconhecido.')
                    if row['series_levels']<1 or row['series_levels']!=int(row['series_levels']):raise ValueError(f'VSC {identifier}: N inválido.')
                    if not 0<=row['limit_correction']<=1:raise ValueError(f'VSC {identifier}: LPCF fora de [0,1].')
                    if row['q_min_mvar']>row['q_max_mvar'] or row['v_min_pu']>row['v_max_pu']:raise ValueError(f'VSC {identifier}: limites invertidos.')
                    if row['ac_base_kv']<=0:raise ValueError(f'VSC {identifier}: base CA deve ser positiva.')
                    row['code']=f'VSC|{identifier}|{row["node"]}'
                else:
                    if len(raw)!=6:raise ValueError(f'VSC {identifier}: linha CC requer De, Para, status e Rccp/Rccn/Rccr.')
                    row=dict(zip(('from_node','to_node','active','positive_r_pu','negative_r_pu','return_r_pu'),raw));row['raw']=raw
                    row['active']=int(row['active'])
                    if row['active'] not in (0,1):raise ValueError(f'VSC {identifier}: status de linha inválido.')
                    for key in ('positive_r_pu','negative_r_pu','return_r_pu'):
                        row[key]=float(row[key])
                        if not math.isfinite(row[key]) or row[key]<0:raise ValueError(f'VSC {identifier}: resistência CC inválida.')
                link[section].append(row)
            if not terminated:raise ValueError(f'VSC {identifier}: terminador 0 ausente em {section}.')
        nodes={r['node'] for r in link['converters']}
        if len(nodes)!=len(link['converters']):raise ValueError(f'VSC {identifier}: nó conversor duplicado.')
        for row in link['lines']:
            if row['from_node'] not in nodes or row['to_node'] not in nodes:
                raise ValueError(f'VSC {identifier}: linha CC referencia nó sem conversor.')
        link['qualification_requirements']=[
            {'field':'dc_voltage_base_kv','reason':'Base física CC para converter pu, corrente e limites em kA; não deduzida da base CA.'},
            {'field':'loss_a/loss_b/loss_c units and base','reason':'Confirmar unidades e base dos fatores antes de mapear IEC alpha1/2/3 nativos.'},
            {'field':'limit_correction and transitions','reason':'Qualificar transições Smax/Imax, Q/V e política LPCF.'},
            {'field':'same-case native reference','reason':'Tensões CA/CC, P/Q, correntes, perdas e controles no mesmo caso energizado.'}]
        result.append(link)
    if len({r['id'] for r in result})!=len(result):raise ValueError('VSC: identificador de elo duplicado.')
    return result