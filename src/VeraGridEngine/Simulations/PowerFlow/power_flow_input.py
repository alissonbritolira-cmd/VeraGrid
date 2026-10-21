# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Identity of the electrical inputs used to obtain a power-flow snapshot."""
from __future__ import annotations

from enum import Enum
import hashlib
import json
import numpy as np

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.Parents.editable_device import EditableDevice
from VeraGridEngine.enumerations import PrpCat


def _electrical_value(value):
    """Encode registered input values without following device-reference cycles."""
    if isinstance(value, EditableDevice):
        return [value.device_type.value, value.idtag]
    if isinstance(value, Enum):
        return [type(value).__name__, value.value]
    if isinstance(value, np.ndarray):
        return _electrical_value(value.tolist())
    if isinstance(value, np.generic):
        return _electrical_value(value.item())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, (list, tuple)):
        return [_electrical_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _electrical_value(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return _electrical_value(value.to_dict())
    if hasattr(value, "get_data"):
        return _electrical_value(value.get_data())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported power-flow input: {type(value).__name__}")


def power_flow_input_fingerprint(grid: MultiCircuit, t_idx: int | None = None) -> str:
    """Hash ordered PF inputs, including connections, without models or UI state.

    PF property declarations provide the electrical data. Connection properties
    are tagged as topology rather than PF; generator dispatch bounds also affect
    distributed slack. Include those explicitly. Array order is significant for
    matching the result vectors back to the circuit.
    """
    connection_properties = {"bus", "bus_from", "bus_to", "reducible", "Pmin", "Pmax"}
    groups = ([grid.buses]
              + grid.get_branch_lists(add_vsc=True, add_hvdc=True, add_switch=True)
              + grid.get_injection_devices_lists())
    payload = ["power-flow-input-v1", grid.Sbase, grid.fBase]
    for devices in groups:
        entries = []
        for device in devices:
            values = [
                [prop.name, _electrical_value(device.get_property_value(prop, t_idx))]
                for prop in device.property_list
                if PrpCat.PF in prop.category or prop.name in connection_properties
            ]
            entries.append([device.device_type.value, device.idtag, values])
        payload.append(entries)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
