# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Estados aceitos entre saídas regulares e limites laterais dos eventos."""
import numpy as np


def right_limit_mask(times,boundary):
    """Apply a retained mode to the right side, keeping the paired left side."""
    axis=np.asarray(times,float)
    left=np.r_[np.diff(axis)==0,False] if len(axis) else np.empty(0,bool)
    equal=np.abs(axis-float(boundary))<=64*np.finfo(float).eps*max(1.,abs(float(boundary)))
    return ((axis>float(boundary))|equal)&~(equal&left)


def record_local_step(solver,time_s,values,macro_target,left_limit=None,boundary_metadata=None):
    local=abs(time_s-macro_target)>1e-12
    if left_limit is not None:
        solver.boundary_left_limits.append(dict(time_s=float(time_s),values=left_limit.copy(),
            right_partner=dict(kind='local' if local else 'regular',
                index=len(solver.accepted_substeps) if local else int(solver.accepted_samples)),
            **(boundary_metadata or {})))
    if local:
        solver.accepted_substeps.append(dict(time_s=float(time_s),values=values.copy()))


def align_event_companions(regular_times,local_times,left_limits,legacy_nanoseconds=False):
    """Derive a paired axis without rewriting any stored clock or state.

    New captures name the accepted right sample by index. Historical datetime
    clocks lost subnanosecond precision: only a unique companion within that
    known storage resolution can be recovered. This is not general resampling.
    """
    regular=np.asarray(regular_times,float).copy()
    local=np.asarray(local_times,float).copy()
    used=set()
    for row in left_limits:
        timestamp=float(row['time_s'])
        partner=row.get('right_partner')
        if partner is not None:
            kind=partner.get('kind');index=partner.get('index')
            if (kind not in ('regular','local') or isinstance(index,bool) or
                not isinstance(index,(int,np.integer))):
                raise ValueError('Invalid RMS event companion identity.')
            axis=regular if kind=='regular' else local
            if not 0<=index<len(axis):raise ValueError('RMS event companion outside accepted coverage.')
            # record_local_step uses this clock protocol to choose macro output.
            clock_resolution=1e-9 if legacy_nanoseconds and kind=='regular' else 1e-12
            if abs(axis[index]-timestamp)>clock_resolution+64*np.finfo(float).eps*max(1.,abs(timestamp)):
                raise ValueError('RMS event companion clock contradicts its identity.')
        else:
            candidates=[]
            for kind,axis in (('regular',regular),('local',local)):
                tolerance=(1e-9+64*np.finfo(float).eps*max(1.,abs(timestamp))
                           if legacy_nanoseconds and kind=='regular' else 0.)
                candidates.extend((kind,int(index)) for index in
                                  np.flatnonzero(np.abs(axis-timestamp)<=tolerance))
            if len(candidates)!=1:
                raise ValueError('RMS event has no unique accepted clock companion.')
            kind,index=candidates[0];axis=regular if kind=='regular' else local
        identity=(kind,index)
        if identity in used:raise ValueError('Multiple RMS left limits name the same companion.')
        used.add(identity);axis[index]=timestamp
    merged=np.sort(np.concatenate((regular,local)))
    if not np.all(np.isfinite(merged)) or np.any(np.diff(merged)<=0):
        raise ValueError('Duplicate or invalid RMS accepted right-limit clocks.')
    return regular,local


def last_accepted_time(solver):
    regular=float(solver.t[solver.accepted_samples-1]) if solver.accepted_samples else None
    extra=solver.accepted_substeps[-1]['time_s'] if solver.accepted_substeps else None
    return max(value for value in (regular,extra) if value is not None) if regular is not None or extra is not None else None
