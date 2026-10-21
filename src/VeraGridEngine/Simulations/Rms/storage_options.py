# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Native option snapshot helper extracted from the development study_state module."""
import hashlib
import json
import math
from enum import Enum
import numpy as np


def _value(value):
    if isinstance(value, Enum):return _value(value.value)
    if isinstance(value,np.ndarray):
        content=json.dumps(value.tolist(),sort_keys=True,default=str).encode() if value.dtype.hasobject else value.tobytes()
        return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(content).hexdigest())
    if isinstance(value,np.generic):return _value(value.item())
    if isinstance(value,complex):return [value.real,value.imag]
    if isinstance(value,float) and math.isfinite(value) and value.is_integer():return int(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,dict):return {str(k):_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_value(v) for v in value]
    if hasattr(value,'idtag'):return dict(idtag=str(value.idtag))
    if hasattr(value,'uid'):return dict(uid=int(value.uid))
    if hasattr(value,'get_data'):return _value(value.get_data())
    if hasattr(value,'to_dict'):return _value(value.to_dict())
    return str(value)


def options_snapshot(options):
    if options is None:return None
    names=[]
    for cls in type(options).__mro__:
        slots=getattr(cls,'__slots__',())
        names.extend((slots,) if isinstance(slots,str) else slots)
    values={n:getattr(options,n) for n in names if n not in ('__weakref__','__dict__') and hasattr(options,n)}
    values.update(getattr(options,'__dict__',{}))
    for name in ('_idtag','_name','_code','_rdfid','device_type','comment','action','selected_to_merge','diff_changes'):
        values.pop(name,None)
    # Opções aninhadas são configuração, não uma referência a idtag aleatório.
    from VeraGridEngine.Simulations.options_template import OptionsTemplate
    def option_value(value):
        if isinstance(value,OptionsTemplate):return options_snapshot(value)
        if isinstance(value,dict):return {str(k):option_value(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [option_value(v) for v in value]
        return _value(value)
    return option_value(values)
