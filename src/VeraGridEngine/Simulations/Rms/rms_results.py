# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import json
import numpy as np
from typing import List, Dict, Optional


from VeraGridEngine.Simulations.results_template import ResultsTemplate, ResultsProperty
from VeraGridEngine.basic_structures import Vec, DateVec, StrVec, Mat, BoolVec
from VeraGridEngine.enumerations import StudyResultsType, DeviceType
from VeraGridEngine.Utils.Symbolic.symbolic import Var
from VeraGridEngine.Devices.types import ALL_DEV_TYPES


class RmsResults(ResultsTemplate):
    LOCAL_RESULTS_DECLARATIONS = (
        ResultsProperty(name='time_array', tpe=DateVec, old_names=list(), expandable=False),
        ResultsProperty(name='values', tpe=Vec, old_names=list(), expandable=False),
        ResultsProperty(name='rms_events_group_names', tpe=StrVec, old_names=list(), expandable=False),
        ResultsProperty(name='rms_events_group_idtags', tpe=StrVec, old_names=list(), expandable=False),
        ResultsProperty(name='has_event_group_results', tpe=BoolVec, old_names=list(), expandable=False),
        ResultsProperty(name='well_initialized', tpe=BoolVec, old_names=list(), expandable=False),
        ResultsProperty(name='converged', tpe=BoolVec, old_names=list(), expandable=False),
        ResultsProperty(name='cancelled', tpe=BoolVec, old_names=list(), expandable=False),
        ResultsProperty(name='recovered', tpe=BoolVec, old_names=list(), expandable=False),
        ResultsProperty(name='dynamic_metadata_json', tpe=StrVec, old_names=list(), expandable=False),
    )

    __slots__ = (
        "nt",
        "nv",
        "ng",
        "rms_events_group_names",
        "rms_events_group_idtags",
        "has_event_group_results",
        "well_initialized",
        "converged",
        "cancelled",
        "recovered",
        "trajectory_directory",
        "archive_chunked",
        "dynamic_metadata_json",
        "_stored_dynamic_metadata",
        "runtime_storage",
        "variables",
        "uid2vars_glob_name",
        "devices_vars_info",
        "initial_parameter_value_maps",
        "parameter_value_maps",
        "uid2idx",
        "vars_glob_name2uid",
        "variable_array",
        "values",
        "valid_samples",
        "execution_diagnostics",
        "topology_history",
        "mode_history",
        "protection_history",
        "integration_performance",
        "accepted_substeps",
        "regular_sample_times_s",
        "boundary_left_limits",
        "numerical_state_projections",
        "fault_history",
        "state_count",
        "declared_event_groups",
        "study_projection_context",
        "runtime_parameter_identities",
    )

    def __init__(self,
                 time_array: DateVec,
                 rms_events_group_names: StrVec,
                 rms_events_group_idtags: StrVec,
                  variables: List[Var],
                 uid2idx: Dict[int, int],
                 vars_glob_name2uid: Dict[str, int],
                 devices_vars_info: Dict[ALL_DEV_TYPES, List[Var]],
                 initial_parameter_value_maps: Optional[List[Dict[str, float]]] = None,
                 parameter_value_maps: Optional[List[Dict[str, float]]] = None,
                 has_event_group_results: Optional[StrVec] = None,
                 declared_event_groups: Optional[List[Dict[str, object]]] = None,
                 allocate_values: bool = True):
        """
        Build the RMS dynamic results container.

        :param time_array: Ordered simulation time samples.
        :param rms_events_group_names: Ordered declared RMS event-group names.
        :param rms_events_group_idtags: Ordered declared RMS event-group idtags.
        :param variables: Exported RMS variables stored in ``values``.
        :param uid2idx: Mapping from variable uid to the ``values`` column index.
        :param vars_glob_name2uid: Mapping from global variable label to variable uid.
        :param devices_vars_info: Device-to-variable mapping used to rebuild the results tree.
        :param initial_parameter_value_maps: Per-event-group initial parameter scalar maps exported for GUI plotting.
        :param parameter_value_maps: Per-event-group final parameter scalar maps exported for GUI plotting.
        :param has_event_group_results: Boolean mask telling which event-group columns contain actual simulation data.
        :return: None.

        New driver runs allocate only the selected event groups and preserve
        the full declaration catalogue separately. Historical full-group arrays
        remain readable and are never silently compacted when saved again.
        """
        ResultsTemplate.__init__(
            self,
            name='RMS simulation',
            available_results=[],
            time_array=time_array,
            clustering_results=None,
            study_results_type=StudyResultsType.RmsSimulation
        )

        self.nt = len(time_array)
        self.nv = len(variables)
        self.state_count=None
        self.ng = len(rms_events_group_names)
        self.declared_event_groups = declared_event_groups
        self.study_projection_context = None
        self.runtime_parameter_identities = None

        self.rms_events_group_names = rms_events_group_names
        self.rms_events_group_idtags = rms_events_group_idtags

        # A selected group may fail before producing any accepted samples.
        # Plotting must not turn its placeholder buffer into an actual curve.
        if has_event_group_results is None:
            self.has_event_group_results = np.ones(self.ng, dtype=bool)
        else:
            self.has_event_group_results = np.array(has_event_group_results, dtype=bool)

        self.well_initialized = np.zeros(self.ng, dtype=bool)
        self.converged = np.zeros(self.ng, dtype=bool)
        self.cancelled = np.zeros(self.ng, dtype=bool)
        self.recovered = np.zeros(self.ng, dtype=bool)
        self.trajectory_directory = ''
        self.archive_chunked = not allocate_values
        self.dynamic_metadata_json = np.array([""], dtype=str)
        self._stored_dynamic_metadata = None
        self.runtime_storage = None

        self.variables = variables
        self.uid2vars_glob_name = {uid: name for name, uid in vars_glob_name2uid.items()}
        self.devices_vars_info: Dict[ALL_DEV_TYPES, List[Var]] = devices_vars_info
        if initial_parameter_value_maps is None:
            self.initial_parameter_value_maps: List[Dict[str, float]] = list()
            group_index: int
            for group_index in range(self.ng):
                self.initial_parameter_value_maps.append(dict())
        else:
            self.initial_parameter_value_maps = list(initial_parameter_value_maps)
        if parameter_value_maps is None:
            self.parameter_value_maps: List[Dict[str, float]] = list()
            group_index: int
            for group_index in range(self.ng):
                self.parameter_value_maps.append(dict())
        else:
            self.parameter_value_maps = list(parameter_value_maps)
        self.uid2idx: Dict[int, int] = uid2idx
        self.vars_glob_name2uid = vars_glob_name2uid
        self.variable_array = np.array([self.uid2vars_glob_name[var.uid] for var in variables], dtype=str)
        if allocate_values:
            # Preserve constructor compatibility for existing native readers.
            self.values = np.zeros((self.nt, self.nv, self.ng), dtype=float)
        else:
            from VeraGridEngine.Simulations.Rms.scenario_values import RmsScenarioValues
            self.values = RmsScenarioValues((self.nt, self.nv, self.ng))
        self.valid_samples = np.full(self.ng, self.nt, dtype=int)
        self.execution_diagnostics = [dict(status='not_executed') for _ in range(self.ng)]
        self.topology_history = [[] for _ in range(self.ng)]
        self.mode_history = [[] for _ in range(self.ng)]
        self.protection_history = [[] for _ in range(self.ng)]
        self.integration_performance = [{} for _ in range(self.ng)]
        self.accepted_substeps=[[] for _ in range(self.ng)]
        self.regular_sample_times_s=[None for _ in range(self.ng)]
        self.boundary_left_limits=[[] for _ in range(self.ng)]
        self.numerical_state_projections=[[] for _ in range(self.ng)]
        self.fault_history=[[] for _ in range(self.ng)]
        self.prepare_for_saving()

    def _build_dynamic_metadata_payload(self, include_values=True) -> Dict[str, object]:
        """
        Build the serialized dynamic metadata payload.

        :return: JSON-safe metadata dictionary.
        """
        payload: Dict[str, object] = dict()
        payload['schema_version'] = (7 if self.runtime_parameter_identities is not None else
            6 if self.study_projection_context is not None else 5 if self.declared_event_groups is not None else 4)
        if self.runtime_parameter_identities is not None:
            payload['runtime_parameter_identities']=self.runtime_parameter_identities
            payload['runtime_parameter_identity_contract']='prepared_runtime_vector_v1'
        if self.study_projection_context is not None:
            payload['study_projection_context']=self.study_projection_context
            payload['channel_projection_contract']='execution_input_snapshot_v1'
        if self.declared_event_groups is not None:
            payload['group_axis_contract'] = 'selected_scenarios_with_full_declaration_catalogue_v1'
            payload['declared_event_groups'] = self.declared_event_groups
        if self.runtime_storage is not None:
            payload['runtime_storage_diagnostics'] = (self.runtime_storage.diagnostics()
                if callable(getattr(self.runtime_storage, 'diagnostics', None)) else self.runtime_storage)
        payload['state_count']=self.state_count
        payload['valid_samples'] = self.valid_samples.tolist()
        payload['execution_diagnostics'] = self.execution_diagnostics
        payload['topology_history'] = self.topology_history
        payload['mode_history'] = self.mode_history
        payload['protection_history'] = self.protection_history
        payload['integration_performance'] = self.integration_performance
        payload['fault_history']=self.fault_history
        payload['accepted_trajectory_schema']=2
        payload['regular_sample_times_s']=self.regular_sample_times_s
        payload['numerical_projection_schema']=1
        for field in ('accepted_substeps','boundary_left_limits','numerical_state_projections'):
            payload[field]=[[dict(time_s=float(row['time_s']),
                                 **({'values':np.asarray(row['values']).tolist()} if include_values else {}),
                                 **({'right_partner':dict(row['right_partner'])} if 'right_partner' in row else {}))
                             for row in group] for group in getattr(self,field)]
            for source_group,target_group in zip(getattr(self,field),payload[field]):
                for source,target in zip(source_group,target_group):
                    for key in ('proof_schema','observation_only','observational_variable_uids',
                                'stage','projected_state_indices','max_state_correction'):
                        if key in source:target[key]=source[key]
        payload["variables"] = [{"uid": int(var.uid), "name": str(var.name)} for var in self.variables]
        payload["uid2idx"] = {str(uid): int(idx) for uid, idx in self.uid2idx.items()}
        payload["vars_glob_name2uid"] = {str(name): int(uid) for name, uid in self.vars_glob_name2uid.items()}
        payload["devices_vars_info"] = [
            {
                "device_idtag": str(device.idtag),
                "variables": [{"uid": int(var.uid), "name": str(var.name)} for var in variables],
            }
            for device, variables in self.devices_vars_info.items()
        ]
        # The GUI parameter reconstruction needs the event-group initial scalar
        # baseline after a results archive is loaded again. Persisting the maps
        # keeps post-load plotting consistent with live-session plotting.
        payload["initial_parameter_value_maps"] = [
            {str(key): float(value) for key, value in parameter_map.items()}
            for parameter_map in self.initial_parameter_value_maps
        ]
        payload["parameter_value_maps"] = [
            {str(key): float(value) for key, value in parameter_map.items()}
            for parameter_map in self.parameter_value_maps
        ]
        return payload

    def archive_dynamic_metadata(self):
        """Native writer uses headers; explicit legacy JSON export still includes all states."""
        return self._build_dynamic_metadata_payload(include_values=False)

    def parse_saved_data(self, grid, data_dict, logger=None):
        from VeraGridEngine.basic_structures import Logger
        from VeraGridEngine.IO.veragrid.result_store import ChunkedArray
        self._stored_dynamic_metadata=data_dict.get('rms_metadata_payload')
        values=data_dict.get('values')
        # The native generic parser materializes ndarray-compatible objects.
        # Keep the existing development tiles lazy while restoring its metadata.
        metadata={k:v for k,v in data_dict.items() if k!='rms_metadata_payload' and
                  not (k=='values' and isinstance(values,ChunkedArray))}
        super().parse_saved_data(grid,metadata,logger if logger is not None else Logger())
        if isinstance(values,ChunkedArray):
            self.values=values
            self.archive_chunked=True

    def prepare_for_saving(self) -> None:
        """
        Refresh the serialized dynamic metadata before the result archive is written.

        :return: None.
        """
        self.dynamic_metadata_json = np.array(
            [json.dumps(self._build_dynamic_metadata_payload(), separators=(",", ":"))],
            dtype=str,
        )

    def restore_dynamic_metadata(self, grid) -> None:
        """
        Restore dynamic variable metadata from the serialized payload.

        :param grid: Current circuit used to resolve device idtags.
        :return: None.
        """
        # Upstream restores arrays before consolidating the empty result shell.
        self.consolidate_after_loading()
        if self._stored_dynamic_metadata is None and (len(self.dynamic_metadata_json) == 0 or str(self.dynamic_metadata_json[0]) == ""):
            return
        else:
            pass

        metadata = self._stored_dynamic_metadata if self._stored_dynamic_metadata is not None else json.loads(str(self.dynamic_metadata_json[0]))
        schema = metadata.get('schema_version', 1)
        if isinstance(schema,bool) or schema not in (1,2,3,4,5,6,7):
            raise ValueError('Unsupported RMS metadata schema: '+str(schema))
        self.runtime_parameter_identities=None
        if schema==7:
            identities=metadata.get('runtime_parameter_identities')
            if (metadata.get('runtime_parameter_identity_contract')!='prepared_runtime_vector_v1'
                    or not isinstance(identities,dict) or identities.get('version')!=1
                    or identities.get('origin')!='prepared_runtime_parameter_vector'
                    or not isinstance(identities.get('case_idtag'),str)
                    or not isinstance(identities.get('parameters'),dict)):
                raise ValueError('Invalid RMS execution parameter identity map.')
            for uid,row in identities['parameters'].items():
                if (not isinstance(uid,str) or not uid.isdecimal() or int(uid)<=0
                        or not isinstance(row,dict) or not isinstance(row.get('parameter_name'),str)
                        or type(row.get('owner_resolved'))is not bool
                        or type(row.get('name_unique_in_device'))is not bool
                        or (row['owner_resolved']and(not isinstance(row.get('device_idtag'),str)or not row['device_idtag']))
                        or (not row['owner_resolved']and row.get('device_idtag')is not None)):
                    raise ValueError('Invalid RMS execution parameter identity: '+str(uid))
            self.runtime_parameter_identities=identities
        self.study_projection_context=None
        if schema == 6 or (schema==7 and metadata.get('study_projection_context')is not None):
            context=metadata.get('study_projection_context')
            if (metadata.get('channel_projection_contract')!='execution_input_snapshot_v1'
                    or not isinstance(context,dict) or context.get('schema')!=1
                    or not all(k in context for k in ('channel_mapping','device_bus_idtags','bus_voltage_uids','rotors','equivalent_generator_idtags','initial_components','Sbase','fBase'))):
                raise ValueError('Invalid RMS channel projection context.')
            self.study_projection_context=context
        self.declared_event_groups = None
        if schema == 5 or (schema in (6,7) and metadata.get('declared_event_groups') is not None):
            if metadata.get('group_axis_contract') != 'selected_scenarios_with_full_declaration_catalogue_v1':
                raise ValueError('Unsupported RMS group-axis contract.')
            catalogue = metadata.get('declared_event_groups')
            if not isinstance(catalogue, list) or any(
                    not isinstance(row, dict) or not isinstance(row.get('idtag'), str)
                    or not row['idtag'] or not isinstance(row.get('name'), str)
                    or type(row.get('selected')) is not bool for row in catalogue):
                raise ValueError('Invalid RMS declaration catalogue.')
            identities = [row['idtag'] for row in catalogue]
            selected = [row for row in catalogue if row['selected']]
            if (len(set(identities)) != len(identities)
                    or [row['idtag'] for row in selected] != list(map(str, self.rms_events_group_idtags))
                    or [row['name'] for row in selected] != list(map(str, self.rms_events_group_names))):
                raise ValueError('RMS selected scenario axis differs from its declaration catalogue.')
            self.declared_event_groups = catalogue
        state_count=metadata.get('state_count')
        if state_count is not None and (isinstance(state_count,bool) or not isinstance(state_count,int) or not 0<=state_count<=self.nv):
            raise ValueError('Invalid RMS state count.')
        self.state_count=state_count
        counts = metadata.get('valid_samples')
        if counts is None:
            # A failed legacy buffer does not disclose where its accepted
            # trajectory ended. Preserve the buffer, without claiming coverage.
            counts = [self.nt if bool(done) else 0 for done in self.converged]
        if len(counts) != self.ng or any(isinstance(n, bool) or int(n) != n or not 0 <= n <= self.nt for n in counts):
            raise ValueError('Invalid RMS accepted-sample coverage.')
        self.valid_samples = np.asarray(counts, dtype=int)
        self.regular_sample_times_s=metadata.get('regular_sample_times_s',[None for _ in range(self.ng)])
        if len(self.regular_sample_times_s)!=self.ng:
            raise ValueError('Invalid RMS regular clock groups.')
        for index, raw_times in enumerate(self.regular_sample_times_s):
            if raw_times is None:continue # Historical nanosecond clock, explicitly unavailable.
            times=np.asarray(raw_times,dtype=float)
            if (times.shape!=(int(self.valid_samples[index]),) or not np.all(np.isfinite(times)) or
                np.any(times<0) or np.any(np.diff(times)<=0)):
                raise ValueError('Invalid RMS regular accepted clock.')
            self.regular_sample_times_s[index]=times.tolist()
        self.runtime_storage = metadata.get('runtime_storage_diagnostics')
        self.execution_diagnostics = metadata.get('execution_diagnostics', [
            dict(status='legacy_completed' if done else 'legacy_coverage_unknown') for done in self.converged])
        self.topology_history = metadata.get('topology_history', [[] for _ in range(self.ng)])
        self.mode_history = metadata.get('mode_history', [[] for _ in range(self.ng)])
        self.protection_history = metadata.get('protection_history', [[] for _ in range(self.ng)])
        self.integration_performance = metadata.get('integration_performance', [{} for _ in range(self.ng)])
        self.fault_history=metadata.get('fault_history',[[] for _ in range(self.ng)])
        trajectory_schema=metadata.get('accepted_trajectory_schema',0)
        if isinstance(trajectory_schema,bool) or trajectory_schema not in (0,1,2):
            raise ValueError('Unsupported RMS accepted trajectory schema.')
        self.accepted_substeps=metadata.get('accepted_substeps',[[] for _ in range(self.ng)])
        self.boundary_left_limits=metadata.get('boundary_left_limits',[[] for _ in range(self.ng)])
        # Explicit backwards migration: earlier schemas have no diagnostic
        # projection stream. Preserve their original trajectories unchanged.
        self.numerical_state_projections=metadata.get('numerical_state_projections',[[] for _ in range(self.ng)])
        projection_schema=metadata.get('numerical_projection_schema',0)
        if type(projection_schema) is not int or projection_schema not in (0,1):
            raise ValueError('Unsupported RMS numerical projection schema.')
        if schema>=4 and projection_schema!=1:
            raise ValueError('RMS schema4/5 requires the numerical projection contract.')
        for groups in (self.accepted_substeps,self.boundary_left_limits,self.numerical_state_projections):
            if len(groups)!=self.ng:raise ValueError('Invalid RMS accepted trajectory groups.')
            for group in groups:
                # This upstream slice retains irregular records in the native
                # JSON metadata rather than adding the optional lazy collection.
                previous=-float('inf')
                for row in group:
                    timestamp=float(row['time_s']);values=np.asarray(row['values'],float)
                    if not np.isfinite(timestamp) or timestamp<0 or timestamp<=previous or values.shape!=(self.nv,) or not np.all(np.isfinite(values)):
                        raise ValueError('Invalid RMS accepted substep or event left limit.')
                    row['values']=values;previous=timestamp
        # Validate companion identities against the actual accepted coverage,
        # including restored historical clocks, before any channel is queried.
        for index in range(self.ng):
            self.get_accepted_trajectory(index,column=slice(0,0),include_left_limits=True)
        for records in (self.execution_diagnostics,self.topology_history,self.mode_history,
                        self.protection_history,self.integration_performance,self.fault_history):
            if len(records) != self.ng:
                raise ValueError('Invalid RMS per-scenario metadata.')
        variable_defs = metadata.get("variables", list())
        var_by_uid: Dict[int, Var] = dict()
        variable_def: Dict[str, object]
        for variable_def in variable_defs:
            uid = int(variable_def["uid"])
            var_by_uid[uid] = Var(name=str(variable_def["name"]), uid=uid)

        self.variables = [var_by_uid[int(variable_def["uid"])] for variable_def in variable_defs]
        self.uid2idx = {int(uid): int(idx) for uid, idx in metadata.get("uid2idx", dict()).items()}
        self.vars_glob_name2uid = {
            str(name): int(uid) for name, uid in metadata.get("vars_glob_name2uid", dict()).items()
        }
        self.uid2vars_glob_name = {uid: name for name, uid in self.vars_glob_name2uid.items()}
        self.variable_array = np.array([var.name for var in self.variables], dtype=str)

        device_by_idtag = {str(device.idtag): device for device in grid.get_all_elements_iter()}
        self.devices_vars_info = dict()
        device_record: Dict[str, object]
        for device_record in metadata.get("devices_vars_info", list()):
            device = device_by_idtag.get(str(device_record.get("device_idtag", "")), None)
            if device is not None:
                device_vars = list()
                variable_record: Dict[str, object]
                for variable_record in device_record.get("variables", list()):
                    uid = int(variable_record["uid"])
                    variable = var_by_uid.get(uid, None)
                    if variable is None:
                        variable = Var(name=str(variable_record["name"]), uid=uid)
                        var_by_uid[uid] = variable
                    else:
                        pass
                    device_vars.append(variable)
                self.devices_vars_info[device] = device_vars
            else:
                pass

        self.parameter_value_maps = [
            {str(key): float(value) for key, value in parameter_map.items()}
            for parameter_map in metadata.get("parameter_value_maps", list())
        ]
        self.initial_parameter_value_maps = [
            {str(key): float(value) for key, value in parameter_map.items()}
            for parameter_map in metadata.get("initial_parameter_value_maps", list())
        ]

        if len(self.initial_parameter_value_maps) == self.ng:
            pass
        else:
            self.initial_parameter_value_maps = list()
            group_index: int
            for group_index in range(self.ng):
                self.initial_parameter_value_maps.append(dict())

    def event_group_index(self, idtag: str) -> int:
        """Resolve the result axis by identity for compact and historical runs."""
        matches = np.flatnonzero(np.asarray(self.rms_events_group_idtags, dtype=str) == str(idtag))
        if len(matches) != 1:
            raise ValueError('Sequência sem resultado RMS identificado de forma única: ' + str(idtag))
        return int(matches[0])

    def get_var(self, uid: int) -> Var:
        """
        
        :param uid: 
        :return: 
        """
        idx = self.uid2idx[uid]
        return self.variables[idx]

    def get_valid_sample_count(self, group_idx: int) -> int:
        """Return accepted coverage, including partial and restored studies."""
        if not 0 <= group_idx < self.ng:
            raise IndexError('RMS event group outside the results.')
        return int(self.valid_samples[group_idx])

    def get_mode_curve(self, group_idx: int, device_idtag: str, parameter_name: str, times):
        """Reconstruct actual latched modes, including SPS actions, after reload."""
        self.get_valid_sample_count(group_idx)
        key = str(device_idtag)+':'+str(parameter_name)
        changes = [(r['time_s'], r['values'][key]) for r in self.mode_history[group_idx]
                   if key in r['values']]
        if not changes:
            return None
        axis = np.asarray(times, dtype=float)
        baseline = self.initial_parameter_value_maps[group_idx].get(key)
        if baseline is None and (not len(axis) or changes[0][0] > axis[0]+1e-12):
            return None
        output = np.full(len(axis), changes[0][1] if baseline is None else baseline, dtype=float)
        from VeraGridEngine.Simulations.Rms.accepted_trajectory import right_limit_mask
        for time_s, value in changes:
            output[right_limit_mask(axis,time_s)] = float(value)
        return output

    def get_accepted_trajectory(self, group_idx: int, column=None, include_left_limits=False):
        """Regular output plus every accepted local boundary, without padding.

        Left/right pairs share the event time and preserve discontinuities.
        The pre-event state at t=0 is retained in metadata, outside the
        simulated right-limit trajectory. Legacy regular arrays stay intact.
        """
        count=self.get_valid_sample_count(group_idx)
        raw_times=self.regular_sample_times_s[group_idx]
        times=(np.asarray(raw_times,float) if raw_times is not None else
               np.asarray((self.time_array[:count]-self.time_array[0])/np.timedelta64(1,'s'),float))
        if (times.shape!=(count,) or not np.all(np.isfinite(times)) or
            np.any(times<0) or np.any(np.diff(times)<=0)):
            raise ValueError('RMS regular clock does not match accepted coverage or chronology.')
        values=self.values[:count,slice(None) if column is None else column,group_idx]
        extras=self.accepted_substeps[group_idx]
        limits=self.boundary_left_limits[group_idx]
        left=[row for row in limits if row['time_s']>0] if include_left_limits else []
        if not extras and not left:return times,values
        right_records=extras
        right_times=np.asarray([row['time_s'] for row in right_records],float)
        from VeraGridEngine.Simulations.Rms.accepted_trajectory import align_event_companions
        times,right_times=align_event_companions(times,right_times,left,legacy_nanoseconds=raw_times is None)
        all_times=np.concatenate((times,right_times,np.asarray([row['time_s'] for row in left],float)))
        def rows(records):
            if column is None:return np.asarray([row['values'] for row in records],float).reshape((-1,self.nv))
            # Legacy/in-memory records also select before conversion; no unused columns.
            if not records:return np.empty((0,)+np.shape(self.values[:0,column,group_idx])[1:])
            return np.asarray([np.asarray(row['values'])[column] for row in records],float)
        all_values=np.concatenate((values,rows(right_records),rows(left)),axis=0)
        sides=np.concatenate((np.ones(len(times)+len(extras),int),np.zeros(len(left),int)))
        order=np.lexsort((sides,all_times))
        all_times,all_values=all_times[order],all_values[order]
        if not np.all(np.isfinite(all_times)) or np.any(np.diff(all_times)<0):
            raise ValueError('Invalid RMS accepted trajectory chronology.')
        if not include_left_limits and np.any(np.diff(all_times)<=0):
            raise ValueError('Duplicate RMS accepted right-limit samples.')
        return all_times,all_values

    def export_accepted_scenario(self,group_idx):
        """Complete accepted trajectory for a DSA point, without display thinning.

        Parameters and ownership are retained so raw channels remain readable
        after loading, even if they were not chosen as security criteria.
        """
        times,values=self.get_accepted_trajectory(group_idx,include_left_limits=True)
        if not np.all(np.isfinite(values)):raise ValueError('Accepted RMS trajectory contains non-finite values.')
        owners={}
        for device,variables in self.devices_vars_info.items():
            for variable in variables:
                owners.setdefault(variable.uid,[]).append(dict(device_idtag=device.idtag,code=device.code,name=device.name))
        variables=[dict(uid=str(v.uid),name=v.name,owners=owners.get(v.uid,[]),
                        role=('unknown' if self.state_count is None else 'state' if i<self.state_count else 'algebraic'),
                        units='model_native') for i,v in enumerate(self.variables)]
        boundaries=[]
        for row in self.boundary_left_limits[group_idx]:
            timestamp=float(row['time_s'])
            if timestamp<=0:continue # The initial equilibrium is retained separately.
            at=int(np.searchsorted(times,timestamp,side='left'))
            if at+1>=len(times) or times[at]!=timestamp or times[at+1]!=timestamp:
                raise ValueError('Accepted RMS event pair lost its stored identity.')
            entry=dict(time_s=timestamp,left_sample_index=at,right_sample_index=at+1)
            for key in ('right_partner','proof_schema','observation_only','observational_variable_uids'):
                if key in row:entry[key]=row[key]
            boundaries.append(entry)
        return dict(schema_version=1,time_s=times.tolist(),variables=variables,state_count=self.state_count,
            accepted_boundaries=boundaries,
            numerical_projection_schema=1,
            numerical_state_projections=[dict(time_s=float(row['time_s']),
                values=np.asarray(row['values']).tolist(),stage=row.get('stage'),
                projected_state_indices=list(row.get('projected_state_indices',())),
                max_state_correction=row.get('max_state_correction'))
                for row in self.numerical_state_projections[group_idx]],
            values_real=np.real(values).tolist(),
            values_imag=np.imag(values).tolist() if np.iscomplexobj(values) and np.any(np.imag(values)!=0) else None,
            initial_parameters=dict(self.initial_parameter_value_maps[group_idx]),
            final_parameters=dict(self.parameter_value_maps[group_idx]),
            declared_group_idtag=str(self.rms_events_group_idtags[group_idx]),
            complete=bool(self.converged[group_idx]),initialized=bool(self.well_initialized[group_idx]),
            clock_origin='solver_float64_with_retained_pair_identity' if self.regular_sample_times_s[group_idx] is not None else 'legacy_datetime_ns_with_unique_event_companions',
            sampling='every accepted macro/internal step, including physical event left/right pairs; no resampling; numerical bound projection inputs retained separately')

    def get_devices_dict_tree(self) -> Dict[DeviceType, Dict[ALL_DEV_TYPES, List[Var]]]:
        """

        :return:
        """
        tree: Dict[DeviceType, Dict[ALL_DEV_TYPES, List[Var]]] = dict()

        for elm, var_list in self.devices_vars_info.items():

            elm_by_type_dict = tree.get(elm.device_type, None)

            if elm_by_type_dict is None:
                tree[elm.device_type] = {elm: var_list}
            else:
                elm_by_type_dict[elm] = var_list

        return tree

    def get_parameter_value(self, group_idx: int, device_idtag: str, parameter_name: str) -> float | None:
        """
        Get one exported parameter value for one event group.

        :param group_idx: Event-group index.
        :param device_idtag: Stable device identifier.
        :param parameter_name: Canonical symbolic parameter name.
        :return: Parameter scalar value, or ``None``.
        """
        if group_idx >= 0 and group_idx < len(self.parameter_value_maps):
            parameter_key: str = str(device_idtag) + ":" + str(parameter_name)
            return self.parameter_value_maps[group_idx].get(parameter_key, None)
        else:
            return None

    def get_initial_parameter_value(self, group_idx: int, device_idtag: str, parameter_name: str) -> float | None:
        """
        Get one exported initial parameter value for one event group.

        :param group_idx: Event-group index.
        :param device_idtag: Stable device identifier.
        :param parameter_name: Canonical symbolic parameter name.
        :return: Initial parameter scalar value, or ``None``.
        """
        if group_idx >= 0 and group_idx < len(self.initial_parameter_value_maps):
            parameter_key: str = str(device_idtag) + ":" + str(parameter_name)
            return self.initial_parameter_value_maps[group_idx].get(parameter_key, None)
        else:
            return None


    def get_vars_data(self, var_list: List[Var], group_idx: int = 0) -> Mat:
        """
        Get time, vars data matrix for the vars selection
        :param var_list: List of vars
        :param group_idx: group index
        :return: data (time, vars)
        """
        data = np.empty((self.nt, len(var_list)), dtype=float)

        for i, var in enumerate(var_list):
            idx = self.uid2idx[var.uid]
            data[:, i] = self.values[:, idx, group_idx]

        return data

    # def mdl(self, result_type: ResultTypes) -> ResultsTable:
    #     """
    #     Export the results as a ResultsTable for plotting.
    #     """
    #     if result_type == ResultTypes.RmsSimulationReport:
    #         return ResultsTable(
    #             data=np.array(self.values),
    #             index=np.array(pd.to_datetime(self.time_array).astype(str), dtype=np.str_),
    #             columns=self.variable_array,
    #             title="Rms Simulation Results",
    #             units=self.units,
    #             idx_device_type=DeviceType.TimeDevice,
    #             cols_device_type=DeviceType.NoDevice
    #         )
    #
    #     else:
    #         raise Exception(f"Result type not understood: {result_type}")

    def consolidate_after_loading(self) -> None:
        """
        Refresh the dynamic dimensions after disk restore.

        :return: None.
        """
        self.nt = int(self.values.shape[0]) if self.values.ndim >= 1 else 0
        self.nv = int(self.values.shape[1]) if self.values.ndim >= 2 else 0
        self.ng = int(self.values.shape[2]) if self.values.ndim >= 3 else 0
        if len(self.cancelled) == 0:
            self.cancelled = np.zeros(self.ng, dtype=bool)
        if len(self.recovered) == 0:
            self.recovered = np.zeros(self.ng, dtype=bool)

        if len(self.rms_events_group_names) == 0 and len(self.rms_events_group_idtags) > 0:
            self.rms_events_group_names = np.array(self.rms_events_group_idtags, dtype=str)
        else:
            pass
