# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Recover persisted RMS data through native I/O; never run a simulation.

Committed regular rows can be recovered after an interrupted writer. Complete
irregular records require its final result snapshot. Missing state is disclosed,
never synthesized, and a recovered execution is historical until revalidated.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zlib
import numpy as np
from VeraGridEngine.Simulations.Rms.trajectory_store import RmsTrajectoryStore, RmsStoredRows, MINIMUM_FREE_BYTES


def inspect_recovery(directory):
    store = RmsTrajectoryStore(directory)
    with store.reader() as connection:
        connection.execute('BEGIN')
        metadata = {key: json.loads(value) for key,value in connection.execute('SELECT key,value FROM metadata')}
        groups = [dict(zip(('idtag','planned_rows','columns','block_rows','committed','status'), row))
                  for row in connection.execute('SELECT id,n_rows,n_cols,block_rows,committed,status FROM rms_groups')]
        keys = {row[0] for row in connection.execute('SELECT key FROM large_metadata')}
    study = metadata.get('study', {})
    definitions = study.get('groups')
    variables = study.get('variables')
    if not isinstance(definitions,list) or not definitions or not isinstance(variables,list) or not variables:
        raise ValueError('Cadastro de grupos/variáveis ausente na recuperação.')
    expected = {str(row['idtag']) for row in definitions}
    if len(expected)!=len(definitions) or len({g['idtag'] for g in groups})!=len(groups):
        raise ValueError('Identidades repetidas na recuperação.')
    for group in groups:
        if (group['idtag'] not in expected or any(type(group[k]) is not int for k in
                ('planned_rows','columns','block_rows','committed'))
                or group['columns']!=len(variables) or not 1<=group['block_rows']<=512
                or not 0<=group['committed']<=group['planned_rows']):
            raise ValueError('Fronteira confirmada contradiz as dimensões/identidades do estudo.')
    snapshot = metadata.get('input_snapshot', {})
    if snapshot.get('member')!='inputs.veragrid' or not isinstance(snapshot.get('sha256'),str):
        raise ValueError('Entradas originais não foram confirmadas; não é possível reconstruir o estudo.')
    return dict(directory=str(store.directory),metadata=metadata,groups=groups,
                has_result_snapshot='result_snapshot' in keys,
                has_base='result_base' in keys,
                recovered_scope=('Resultado final persistido; conferir contagens e integridade'
                    if 'result_snapshot' in keys else
                    'Somente prefixos regulares; estados irregulares e históricos finais não foram confirmados. Não qualifica segurança.'))


def _clock_and_view(store, group):
    times=[];cursor=0
    with store.reader() as connection:
        rows=connection.execute('SELECT row_start,n_rows,hash,payload FROM clocks WHERE group_id=? AND row_start<? ORDER BY row_start',
                                (group['idtag'],group['committed']))
        for start,count,digest,payload in rows:
            if (type(count) is not int or start!=cursor or not 0<count<=group['block_rows']
                    or cursor+count>group['committed']):
                raise ValueError('Relógio RMS não cobre continuamente a fronteira confirmada.')
            decoder=zlib.decompressobj();raw=decoder.decompress(payload,count*8+1)
            if len(raw)!=count*8 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or hashlib.sha256(raw).hexdigest()!=digest:
                raise ValueError('Integridade do relógio RMS inválida.')
            selected=np.frombuffer(raw,dtype='<f8')
            if not np.all(np.isfinite(selected)) or np.any(selected<0):
                raise ValueError('Relógio RMS não finito/negativo.')
            times.extend(selected.tolist());cursor+=count
        if cursor!=group['committed'] or np.any(np.diff(times)<=0):
            raise ValueError('Relógio RMS incompleto ou fora de ordem.')
    return times,RmsStoredRows.from_committed(store,group['idtag'],
        (group['planned_rows'],group['columns']),group['block_rows'],group['committed'])


def _validate_variable_metadata(metadata, study):
    """Reject contradictory channel identities before native curve reconstruction."""
    definitions=metadata.get('variables')
    original=study['variables']
    if (not isinstance(definitions,list) or len(definitions)!=len(original)
            or any(not isinstance(row,dict) or type(row.get('uid'))is not int
                   or not isinstance(row.get('name'),str) for row in definitions)):
        raise ValueError('Invalid RMS recovery variable definitions.')
    identities=[row['uid'] for row in definitions]
    if (len(set(identities))!=len(identities) or
            [(str(row['uid']),row['name']) for row in definitions] !=
            [(str(row['uid']),row['name']) for row in original]):
        raise ValueError('RMS recovery channel identities differ from the preserved study.')
    indices=metadata.get('uid2idx')
    if (not isinstance(indices,dict) or set(indices)!=set(map(str,identities)) or
            any(type(value)is not int for value in indices.values()) or
            sorted(indices.values())!=list(range(len(identities)))):
        raise ValueError('Invalid RMS recovery variable index map.')
    labels=metadata.get('vars_glob_name2uid')
    # Native global labels include parameter identities as well as exported
    # trajectory columns; those extra parameter entries must remain available.
    if (not isinstance(labels,dict) or any(not isinstance(key,str) or type(value)is not int
            for key,value in labels.items()) or not set(identities).issubset(set(labels.values()))):
        raise ValueError('Invalid RMS recovery global variable map.')


def export_recovery(directory, destination, *, checkpoint=lambda:None, progress=lambda message:None):
    """Export an independent transactional native file from preserved inputs."""
    from VeraGridEngine.IO.file_open import FileOpen
    from VeraGridEngine.IO.file_save import FileSave,FileSavingOptions
    from VeraGridEngine.Simulations.Rms.rms_results import RmsResults
    from VeraGridEngine.Simulations.driver_template import DriverToSave
    from VeraGridEngine.basic_structures import Logger
    from VeraGridEngine.enumerations import SimulationTypes
    from VeraGridEngine.Utils.Symbolic.symbolic import Var
    import pandas as pd
    info=inspect_recovery(directory)
    location=Path(info['directory']);target=Path(destination).resolve()
    if target.suffix.lower()!='.veragrid' or target.is_relative_to(location):
        raise ValueError('Grave um novo .veragrid fora da pasta original de recuperação.')
    if target.exists():
        raise ValueError('Choose a new destination; existing studies are preserved')
    if shutil.disk_usage(target.parent).free<MINIMUM_FREE_BYTES:
        raise ValueError('Destino abaixo da reserva de512MiB; recuperação original preservada.')
    checkpoint();progress('Conferindo as entradas preservadas…')
    snapshot=location/'inputs.veragrid';digest=hashlib.sha256()
    with snapshot.open('rb') as stream:
        while chunk:=stream.read(1024**2):checkpoint();digest.update(chunk)
    if digest.hexdigest()!=info['metadata']['input_snapshot']['sha256']:
        raise ValueError('As entradas preservadas mudaram ou estão corrompidas. Nenhum resultado foi recuperado.')
    opener=FileOpen(str(snapshot));grid=opener.open()
    if grid is None or opener.logger.has_errors():
        raise ValueError('Não foi possível abrir o caso original preservado.')
    study=info['metadata']['study']
    if str(grid.idtag)!=str(study['case_idtag']):
        raise ValueError('O caso preservado não corresponde à identidade da execução.')
    checkpoint();progress('Conferindo os prefixos e metadados confirmados…')
    store=RmsTrajectoryStore(location);store._cancel_reader=checkpoint
    try:
        saved=store.read_json_stream('result_snapshot') if info['has_result_snapshot'] else None
        base=store.read_json_stream('result_base') if saved is None else saved['metadata']
        if base is None:raise ValueError('Metadados base não foram confirmados; dados originais preservados.')
        metadata=copy.deepcopy(saved['metadata'] if saved is not None else base)
        _validate_variable_metadata(metadata,study)
        groups=study['groups'];by_id={row['idtag']:row for row in info['groups']}
        options=study['options'];step=float(options['time_step']);duration=float(options['simulation_time'])
        if not np.isfinite(step) or step<=0 or not np.isfinite(duration) or duration<0:
            raise ValueError('Horizonte/passo original inválido.')
        planned=int(np.ceil(duration/step))+1
        if any(g['planned_rows']!=planned for g in info['groups']):
            raise ValueError('Dimensões da recuperação diferem das opções originais.')
        # Upstream BE currently exposes ceil(horizon / step) macro boundaries.
        # Retain that declared axis; the committed clocks below remain primary.
        times=np.arange(planned)*step
        variables=[Var(name=str(row['name']),uid=int(row['uid'])) for row in metadata['variables']]
        results=RmsResults(time_array=pd.DatetimeIndex(pd.to_datetime(times*1e9)),
            rms_events_group_names=np.asarray([row['name'] for row in groups],dtype=str),
            rms_events_group_idtags=np.asarray([row['idtag'] for row in groups],dtype=str),variables=variables,
            uid2idx={int(k):int(v) for k,v in metadata['uid2idx'].items()},
            vars_glob_name2uid={str(k):int(v) for k,v in metadata['vars_glob_name2uid'].items()},
            devices_vars_info={},has_event_group_results=np.zeros(len(groups),dtype=bool),
            declared_event_groups=metadata.get('declared_event_groups'),allocate_values=False)
        counts=[];clocks=[]
        for index,definition in enumerate(groups):
            checkpoint();group=by_id.get(definition['idtag']);count=0 if group is None else group['committed']
            if count:
                clock,view=_clock_and_view(store,group);results.values.attach(index,view,count)
            else:clock=[]
            counts.append(count);clocks.append(clock)
        expected=metadata.get('valid_samples',[])
        complete_metadata=(saved is not None and expected==counts)
        if not any(counts):
            raise ValueError('Nenhuma amostra regular foi confirmada no banco. Se a cauda foi salva em um .veragrid, abra esse arquivo; os originais foram preservados.')
        if saved is not None and not complete_metadata:
            raise ValueError('O snapshot final declara amostras que não foram confirmadas em blocos. Abra o .veragrid salvo durante a sessão, se disponível; a recuperação não descarta os estados/históricos persistidos para simular um resultado completo.')
        if saved is not None:
            for field in ('accepted_substeps','boundary_left_limits','numerical_state_projections'):
                metadata[field]=saved['irregular'][field]
        if not complete_metadata:
            # A missing state stream is disclosed, never invented as a complete trajectory.
            for field in ('accepted_substeps','boundary_left_limits','numerical_state_projections',
                          'topology_history','mode_history','protection_history','fault_history'):
                metadata[field]=[[] for _ in groups]
            metadata['execution_diagnostics']=[dict(status='recovered_regular_prefix_only',
                reason='Recuperado somente o prefixo regular confirmado. Estados irregulares/históricos não foram confirmados; não qualifica segurança.',
                unpersisted_original_samples=(expected[i] if i<len(expected) else None),
                physical_instability_demonstrated=False) for i in range(len(groups))]
        metadata['valid_samples']=counts;metadata['regular_sample_times_s']=clocks
        results.has_event_group_results=np.asarray(counts)>0
        results.well_initialized=np.asarray(saved['well_initialized'],bool) if complete_metadata else np.zeros(len(groups),bool)
        results.converged=np.asarray(saved['converged'],bool) if complete_metadata else np.zeros(len(groups),bool)
        results.converged &= np.asarray(counts) == planned
        results.cancelled=(np.asarray(saved['cancelled'],bool) if complete_metadata and 'cancelled' in saved else
            np.asarray([row.get('status')=='cancelled' for row in metadata.get('execution_diagnostics',[{} for _ in groups])]))
        results.recovered=np.asarray(counts)>0
        results.trajectory_directory=str(location)
        results._stored_dynamic_metadata=metadata;results.restore_dynamic_metadata(grid)
        results.runtime_storage=dict(recovery=dict(directory=str(location),scope='complete_persisted_snapshot' if complete_metadata else 'regular_prefix_only',
            original_inputs_sha256=digest.hexdigest(),originals_preserved=True,integrator_restart=False),experimental=True)
        logger=Logger();logger.add_warning('Execução recuperada como histórico; conferir a cobertura antes de reutilizar.',value=info['recovered_scope'])
        wrapped=DriverToSave('Recuperação RMS',SimulationTypes.RmsDynamic_run,results,logger)
        # Upstream DriverToSave has no input_context slot. Keep the same context
        # in persisted RMS metadata rather than widening the shared driver API.
        context=copy.deepcopy(study.get('input_identity') or {})
        context.update(status='recovered',reason='Recuperação de dados confirmados; revalidação do caso e dos critérios exigida.',
                       recovery=results.runtime_storage['recovery'])
        results.runtime_storage['input_context']=context
        checkpoint();progress('Gravando recuperação transacional; originais preservados…')
        json_files={key:value for key,value in opener.json_files.items() if key=='gui_config'}
        writer=FileSave(grid,str(target),options=FileSavingOptions(sessions_data=[wrapped],dictionary_of_json_files=json_files))
        log=writer.save()
        if log.has_errors() or not target.is_file():raise ValueError('A recuperação não foi confirmada; conserve os arquivos originais.')
        return dict(destination=str(target),counts=counts,complete_persisted_metadata=complete_metadata,
                    scope='complete_persisted_snapshot' if complete_metadata else 'regular_prefix_only')
    finally:
        store.close()
