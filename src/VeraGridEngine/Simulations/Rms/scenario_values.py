# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""RMS result axes over accepted solver buffers, without a second trajectory.

The solver has stopped before a group is attached. Selection returns independent
arrays and unaccepted/unexecuted samples are NaN. Native persistence still writes
the complete declared shape in bounded blocks; no channel or state is removed.
"""
from __future__ import annotations

import copy
import math
import numpy as np


class RmsScenarioValues:
    """Read-only [time, variable, scenario] projection of completed buffers."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        if len(self.shape) != 3 or any(type(n) is not int or n < 0 for n in self.shape):
            raise ValueError('Invalid RMS scenario result shape.')
        self.dtype = np.dtype(float)
        self.ndim = 3
        self.size = math.prod(self.shape)
        self.nbytes = self.size * self.dtype.itemsize
        self._groups = [None] * self.shape[2]

    def __len__(self):
        return self.shape[0]

    def attach(self, group, values, accepted_samples):
        """Transfer ownership of the accepted prefix, preserving every Float64."""
        if not 0 <= group < self.shape[2] or self._groups[group] is not None:
            raise ValueError('RMS result group already attached or out of range.')
        if (values.ndim != 2 or values.shape[1] != self.shape[1]
                or values.dtype != self.dtype or not 0 <= accepted_samples <= self.shape[0]
                or values.shape[0] < accepted_samples):
            raise ValueError('RMS solver buffer contradicts accepted result coverage.')
        # Do not convert array-like storage here: future bounded backings retain
        # the same selection protocol. The current ndarray prefix is only a view.
        retained = (values.accepted_prefix(accepted_samples)
                    if callable(getattr(values, 'accepted_prefix', None))
                    else values[:accepted_samples, :])
        if isinstance(retained, np.ndarray):
            retained = retained.view()
            retained.flags.writeable = False
        self._groups[group] = retained

    @property
    def retained_bytes(self):
        return sum(getattr(v, 'resident_bytes', v.size * v.dtype.itemsize)
                   for v in self._groups if v is not None)

    def __getitem__(self, item):
        selectors = item if isinstance(item, tuple) else (item,)
        if any(v is Ellipsis for v in selectors):
            at = next(i for i, v in enumerate(selectors) if v is Ellipsis)
            selectors = selectors[:at] + (slice(None),) * (4-len(selectors)) + selectors[at+1:]
        selectors += (slice(None),) * (3-len(selectors))
        if len(selectors) != 3:
            raise IndexError('Invalid RMS result dimension count.')
        indices, scalar_axes = [], []
        for axis, (selector, length) in enumerate(zip(selectors, self.shape)):
            if isinstance(selector, (int, np.integer)) and not isinstance(selector, (bool, np.bool_)):
                index = int(selector)
                if index < 0:
                    index += length
                if not 0 <= index < length:
                    raise IndexError('RMS result index out of range.')
                indices.append(np.array([index], dtype=int))
                scalar_axes.append(axis)
            elif isinstance(selector, slice):
                indices.append(np.arange(*selector.indices(length), dtype=int))
            elif (axis == 1 and isinstance(selectors[0], slice)
                  and isinstance(selectors[2], (int, np.integer))
                  and not isinstance(selectors[2], (bool, np.bool_))):
                chosen = np.asarray(selector)
                if chosen.ndim != 1 or (chosen.size and chosen.dtype.kind not in 'iu'):
                    return np.asarray(self)[item]
                if np.any(chosen < -length) or np.any(chosen >= length):
                    raise IndexError('RMS result channel out of range.')
                chosen = chosen.astype(int, copy=False)
                indices.append(np.where(chosen < 0, chosen+length, chosen))
            else:
                # Preserve NumPy's full advanced-index semantics for explicit
                # callers; normal GUI channels and archive blocks use above.
                return np.asarray(self)[item]
        result = np.full(tuple(len(v) for v in indices), np.nan, dtype=self.dtype)
        if result.size:
            for target_group, source_group in enumerate(indices[2]):
                source = self._groups[source_group]
                if source is None:
                    continue
                positions = np.flatnonzero(indices[0] < source.shape[0])
                if positions.size:
                    selected = source[np.ix_(indices[0][positions], indices[1])]
                    result[positions, :, target_group] = selected
        return np.squeeze(result, axis=tuple(scalar_axes)) if scalar_axes else result

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError('A combined RMS array requires explicit materialization.')
        values = self[:, :, :]
        return values if dtype is None else values.astype(dtype, copy=False)

    def __deepcopy__(self, memo):
        # Snapshot requests keep the standard independent-copy semantics.
        result = type(self)(self.shape)
        memo[id(self)] = result
        result._groups = copy.deepcopy(self._groups, memo)
        return result
