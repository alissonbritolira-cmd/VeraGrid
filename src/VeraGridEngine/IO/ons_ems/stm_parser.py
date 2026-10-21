# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Read ONS EMS STM version 5 and retain its source records.

The developed converter and power-flow adapters apply supported network and
control representations. Unsupported requests are diagnosed explicitly. PRM
assignments remain inspectable data; reading them never executes a script.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from VeraGridEngine import Devices as dev
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.enumerations import GeneratorControlMode


SECTION_NAMES = ('buses', 'loads', 'generators', 'shunts', 'lines', 'line_shunts',
                 'transformers', 'series_caps', 'dclinks', 'vsc', 'areas',
                 'zones', 'owners', 'substations')


def _read_text(path: str | Path) -> tuple[str, dict]:
    """Read text without running commands or discovering neighbouring files."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp1252')
    return text, {'source': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'text': text}


def _without_comment(line: str) -> str:
    """STM slash comments can occur after a quoted name containing a slash."""
    quoted = False
    index = 0
    while index < len(line):
        if line[index] == "'":
            if quoted and index + 1 < len(line) and line[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif not quoted and line[index] in ('/', '!'):
            return line[:index].strip()
        index += 1
    return line.strip()


def bus_id(value: str) -> str:
    """Preserve SAGE bus sections without treating 100.1 as the integer 100."""
    value = str(value).strip()
    if not re.fullmatch(r'\d+(?:\.\d+)?', value):
        raise ValueError(f'Invalid STM bus/area identifier: {value!r}')
    number, _, section = value.partition('.')
    return str(int(number)) + ('.' + str(int(section)) if section and int(section) else '')


def _number(row: list[str], column: int, context: str) -> float:
    try:
        value = float(row[column].replace('D', 'E').replace('d', 'e'))
    except (ValueError, IndexError) as error:
        raise ValueError(f'{context}: missing or invalid number in field {column + 1}.') from error
    if not math.isfinite(value):
        raise ValueError(f'{context}: non-finite number in field {column + 1}.')
    return value


def _integer(row: list[str], column: int, context: str) -> int:
    value = _number(row, column, context)
    if not value.is_integer():
        raise ValueError(f'{context}: field {column + 1} must be an integer.')
    return int(value)


def _flag(row: list[str], column: int, context: str) -> bool:
    value = _integer(row, column, context)
    if value not in (0, 1):
        raise ValueError(f'{context}: field {column + 1} must be 0 or 1.')
    return bool(value)


def read_stm(path: str | Path) -> tuple[dict, dict]:
    """Read all 14 positional sections, retaining every original source line."""
    text, source = _read_text(path)
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != '5':
        raise ValueError('Only ONS EMS STM version 5 is supported.')
    try:
        base = float(lines[1].split()[-1])
    except (ValueError, IndexError) as error:
        raise ValueError('STM header has no valid MVA base.') from error
    if not math.isfinite(base) or base <= 0:
        raise ValueError('STM MVA base must be positive and finite.')
    sections, current = [], []
    for line_number, line in enumerate(lines[3:], 4):
        if re.match(r'^\s*0\s*/', line):
            sections.append(current)
            current = []
            continue
        clean = _without_comment(line)
        if clean:
            try:
                fields = next(csv.reader([clean], quotechar="'", skipinitialspace=True, strict=True))
            except csv.Error as error:
                raise ValueError(f'STM line {line_number}: malformed quoted record.') from error
            current.append(([field.strip() for field in fields], line_number))
    if current or len(sections) < len(SECTION_NAMES):
        raise ValueError('STM has incomplete sections; each section must end with 0 /.')
    if any(sections[len(SECTION_NAMES):]):
        raise NotImplementedError('Additional nonempty STM sections are not supported.')
    data = dict(zip(SECTION_NAMES, sections))
    data['Sbase'] = base
    data['header'] = lines[2]
    return data, source


def read_prm(path: str | Path | None) -> tuple[dict, list[dict], dict | None]:
    """Keep command order and repeated assignments; never evaluate PRM text."""
    if path is None:
        return {}, [], None
    text, source = _read_text(path)
    parameters, history = {}, []
    for line_number, line in enumerate(text.splitlines(), 1):
        clean = _without_comment(line)
        if not clean:
            continue
        match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)', clean)
        if match is None:
            history.append({'line': line_number, 'text': line, 'status': 'preserved, not applied'})
            continue
        key, value = match[1].upper(), match[2].strip()
        if value.upper() in ('T', 'F', 'Y', 'N'):
            value = value.upper() in ('T', 'Y')
        else:
            try:
                numeric = float(value.replace('D', 'E').replace('d', 'e'))
            except ValueError:
                pass
            else:
                if not math.isfinite(numeric):
                    raise ValueError(f'PRM line {line_number}: {key} must be finite.')
                value = numeric
        parameters[key] = value
        history.append({'command': key, 'value': value, 'line': line_number,
                        'status': 'preserved, not applied'})
    return parameters, history, source




def inspect_stm(path: str | Path, prm_path: str | Path | None = None,
                frequency_hz: float = 60.0) -> dict:
    """Return a complete read-only source inventory, not a partial circuit."""
    data, source = read_stm(path)
    parameters, assignments, prm_source = read_prm(prm_path)
    return {'schema': 1, 'format': 'ONS EMS STM v5', 'stm': source, 'prm': prm_source,
            'parameters': parameters, 'assignments': assignments, 'frequency_hz': frequency_hz,
            'section_records': {name: [{'line': line, 'fields': row} for row, line in data[name]]
                                for name in SECTION_NAMES},
            'unsupported_features': unsupported_features(data),
            'diagnostics': ['Read-only source inspection. No circuit is imported and no simulation is enabled.'],
            'applied_gui_options': {}}




def unsupported_features(data: dict) -> list[str]:
    """Substantive devices are imported through the developed PF adapters.

    Input errors are reported by those adapters; presence alone is not a
    reason to replace an import by source-only inspection.
    """
    return []


def branch_key(f, t, circuit):
    from VeraGridEngine.IO.ons_ems.network_converter import branch_key as implementation
    return implementation(f,t,circuit)


def branch_records(data, orphan_records=None):
    from VeraGridEngine.IO.ons_ems.network_converter import branch_records as implementation
    return implementation(data,orphan_records)


def open_stm(path: str | Path, prm_path: str | Path | None = None,
             frequency_hz: float = 60.0, logger: Logger | None = None) -> dev.MultiCircuit:
    """Import the complete network through the developed, tested EMS path."""
    from VeraGridEngine.IO.ons_ems.network_converter import open_stm as convert
    data,source=read_stm(path)
    # Preserve the prototype's descriptive malformed-record errors when
    # delegating to conversion functions that index positional source fields.
    for section, minimum in (('buses',17),('loads',9),('generators',18),('shunts',7),
                             ('lines',23),('line_shunts',9),('transformers',25),
                             ('series_caps',18),('areas',2),('zones',2),
                             ('owners',2),('substations',2)):
        for row,line in data[section]:
            if len(row)<minimum:
                raise ValueError(f'STM {section} line {line}: expected at least {minimum} fields.')
    grid=convert(path,prm_path=prm_path,frequency=frequency_hz,logger=logger)
    parameters,history,prm_source=read_prm(prm_path)
    grid.import_metadata.update(schema=2,format='ONS EMS STM v5',stm=source,prm=prm_source,
        parameters=parameters,assignments=history,frequency_hz=frequency_hz,
        section_records={name:[{'line':line,'fields':row} for row,line in data[name]]
                         for name in SECTION_NAMES},unsupported_features=[],applied_gui_options={})
    return grid
