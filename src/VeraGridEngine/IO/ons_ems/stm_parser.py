# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Import the AC, fixed-control subset of ONS EMS STM version 5.

This is a network conversion, not an ONS EMS study interpreter. Unsupported
electrical representations are rejected rather than silently omitted. A PRM
is retained for consultation; reading it does not execute its commands.
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


def unsupported_features(data: dict) -> list[str]:
    """Report source features requiring adapters outside this first AC slice."""
    reasons = []
    for section in ('series_caps', 'dclinks', 'vsc'):
        if data[section]:
            reasons.append(f'{section}: {len(data[section])} records need an additional electrical adapter.')
    switched = sum(len(row) > 1 and row[1] not in ('0', '0.0') for row, _ in data['shunts'])
    if switched:
        reasons.append(f'Switched shunts/SVCs: {switched} records need a control adapter.')
    automatic = sum(len(row) > 20 and row[20] not in ('0', '0.0') for row, _ in data['transformers'])
    if automatic:
        reasons.append(f'Automatic transformer controls: {automatic} records are not supported.')
    remote = sum(len(row) > 7 and bus_id(row[7]) not in ('0', bus_id(row[0]))
                 for row, _ in data['generators'])
    if remote:
        reasons.append(f'Remote generator voltage controls: {remote} records are not supported.')
    for section, columns in (('lines', (15, 16)), ('transformers', (18, 19))):
        count = sum(len(row) > columns[1] and row[columns[0]] != row[columns[1]] for row, _ in data[section])
        if count:
            reasons.append(f'{section} with a single open terminal: {count} records need an open-end equivalent.')
    return reasons


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


def open_stm(path: str | Path, prm_path: str | Path | None = None,
             frequency_hz: float = 60.0, logger: Logger | None = None) -> dev.MultiCircuit:
    """Build native AC devices, refusing unsupported substantive records."""
    logger = Logger() if logger is None else logger
    data, source = read_stm(path)
    parameters, assignments, prm_source = read_prm(prm_path)
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError('STM import frequency must be positive and finite.')
    for section in ('series_caps', 'dclinks', 'vsc'):
        if data[section]:
            raise NotImplementedError(f'STM {section} are not supported in this AC importer; '
                                      'the network has not been imported partially.')
    grid = dev.MultiCircuit(name=Path(path).stem, Sbase=data['Sbase'], fbase=frequency_hz)
    grid.comments = 'Imported ONS EMS STM v5 AC network. PRM commands are not executed automatically.'
    metadata = {'schema': 1, 'format': 'ONS EMS STM v5', 'stm': source,
                'prm': prm_source, 'parameters': parameters, 'assignments': assignments,
                'frequency_hz': frequency_hz, 'diagnostics': [], 'applied_gui_options': {}}
    grid.import_metadata = metadata

    def warning(message: str):
        metadata['diagnostics'].append(message)
        logger.add_warning(message)

    def records(section: str, minimum: int):
        for row, line_number in data[section]:
            context = f'STM {section} line {line_number}'
            if len(row) < minimum:
                raise ValueError(f'{context}: expected at least {minimum} fields, got {len(row)}.')
            yield row, context

    buses, kinds = {}, {}

    def bus(value: str):
        key = bus_id(value)
        if key not in buses:
            raise ValueError(f'STM references missing bus {key}.')
        return buses[key]

    groups = {}
    for section, factory, add in (
            ('areas', dev.Area, grid.add_area), ('zones', dev.Zone, grid.add_zone),
            ('owners', dev.Owner, grid.add_owner), ('substations', dev.Substation, grid.add_substation)):
        groups[section] = {}
        for row, context in records(section, 2):
            key = bus_id(row[0])
            if key in groups[section]:
                raise ValueError(f'{context}: duplicate identifier {key}.')
            obj = factory(name=row[1], code=key)
            add(obj)
            groups[section][key] = obj
    for row, context in records('buses', 17):
        key, kind = bus_id(row[0]), _integer(row, 3, context)
        if key in buses or kind not in (0, 1, 2, 3, 4):
            raise ValueError(f'{context}: duplicate bus or unsupported type for {key}.')
        vnom, vm, va, vmax, vmin = [_number(row, i, context) for i in (2, 9, 10, 11, 12)]
        if vnom <= 0 or vm <= 0 or vmin <= 0 or vmax < vmin:
            raise ValueError(f'{context}: invalid nominal voltage, voltage seed or voltage limits.')
        obj = dev.Bus(name=row[1], code=key, Vnom=vnom, Vm0=vm, Va0=np.deg2rad(va),
                      vmax=vmax, vmin=vmin, active=kind != 4, is_slack=kind == 3)
        for field, section, column in (('area', 'areas', 7), ('zone', 'zones', 8),
                                       ('substation', 'substations', 16)):
            code = bus_id(row[column])
            if code != '0' and code not in groups[section]:
                raise ValueError(f'{context}: undefined {field} {code}.')
            setattr(obj, field, groups[section].get(code))
        grid.add_bus(obj)
        buses[key], kinds[key] = obj, kind
        enabled = _flag(row, 4, context)
        g, b = _number(row, 5, context), _number(row, 6, context)
        if enabled or g or b:
            grid.add_shunt(obj, dev.Shunt(name=f'Bus shunt {key}', code=f'bus_shunt|{key}',
                                         G=g, B=b, active=enabled))
    identifiers = set()

    def device_key(row: list[str], context: str, category: str):
        key = f'{bus_id(row[0])}|{row[1]}'
        if not row[1] or (category, key) in identifiers:
            raise ValueError(f'{context}: missing or duplicate {category} identifier {key}.')
        identifiers.add((category, key))
        return key

    for row, context in records('loads', 9):
        target = bus(row[0])
        key = device_key(row, context, 'load')
        p, q, ip, iq, zp, zq = [_number(row, i, context) for i in range(3, 9)]
        grid.add_load(target, dev.Load(name=row[12] if len(row) > 12 else f'Load {key}', code=key,
                      active=_flag(row, 2, context), P=p, Q=q, Ir=ip / target.Vm0, Ii=-iq / target.Vm0,
                      G=zp / target.Vm0 ** 2, B=-zq / target.Vm0 ** 2))
    for row, context in records('generators', 18):
        target = bus(row[0])
        key = device_key(row, context, 'generator')
        control = bus_id(row[7])
        if control not in ('0', target.code):
            raise NotImplementedError(f'{context}: remote voltage control is not supported ({key}).')
        p, q, qmax, qmin, vset, snom, pmax, pmin = [_number(row, i, context)
                                                 for i in (2, 3, 4, 5, 6, 8, 14, 15)]
        if snom <= 0 or vset <= 0 or qmax < qmin or pmax < pmin:
            raise ValueError(f'{context}: invalid generator ratings, limits or voltage setting.')
        if len(row) >= 26:
            impedances = dict(zip(('r1', 'x1', 'r0', 'x0'),
                                  [_number(row, i, context) for i in range(20, 24)]))
        else:
            impedances = {}
        grid.add_generator(target, dev.Generator(name=row[26] if len(row) > 26 else f'Generator {key}',
                           code=key, P=p, Q=q, Qmax=qmax, Qmin=qmin, vset=vset, Snom=snom,
                           Pmax=pmax, Pmin=pmin, active=_flag(row, 12, context),
                           enabled_dispatch=not _flag(row, 17, context), freq=frequency_hz,
                           control_mode=GeneratorControlMode.V if kinds[target.code] in (2, 3)
                           else GeneratorControlMode.Q, **impedances))
    for row, context in records('shunts', 7):
        target = bus(row[0])
        if _integer(row, 1, context) != 0:
            raise NotImplementedError(f'{context}: switched shunts and SVCs need an additional control adapter.')
        _number(row, 5, context)
        _flag(row, 6, context)
        warning(f'{context}: fixed-shunt declaration retained; the BUS record supplies its electrical G/B injection '
                f'at bus {target.code}, without adding it twice.')

    branch_objects = {}
    for section, minimum in (('lines', 23), ('transformers', 25)):
        for row, context in records(section, minimum):
            f, t = bus(row[0]), bus(row[1])
            key = f'{f.code}|{t.code}|{row[2]}'
            if not row[2] or key in branch_objects or f is t:
                raise ValueError(f'{context}: invalid or duplicate branch {key}.')
            transformer = section == 'transformers'
            r = _number(row, 4 if transformer else 3, context)
            x = _number(row, 5 if transformer else 4, context)
            sf, st = (_flag(row, i, context) for i in ((18, 19) if transformer else (15, 16)))
            if sf != st:
                raise NotImplementedError(f'{context}: a single open terminal needs an open-end equivalent ({key}).')
            rate, rate2, rate3 = [_number(row, i, context) for i in (6, 7, 8)]
            if min(rate, rate2, rate3) < 0 or complex(r, x) == 0:
                raise ValueError(f'{context}: invalid thermal rating or zero impedance ({key}).')
            if transformer:
                if _flag(row, 20, context):
                    raise NotImplementedError(f'{context}: automatic transformer controls are not supported ({key}).')
                tap, phase = _number(row, 9, context), _number(row, 10, context)
                if tap <= 0:
                    raise ValueError(f'{context}: tap module must be positive.')
                obj = dev.Transformer2W(bus_from=f, bus_to=t, name=row[24], code=key,
                                       HV=max(f.Vnom, t.Vnom), LV=min(f.Vnom, t.Vnom),
                                       r=r, x=x, rate=rate, active=sf and st,
                                       nominal_power=rate, tap_module=tap, tap_phase=np.deg2rad(phase))
                grid.add_transformer2w(obj)
            else:
                obj = dev.Line(bus_from=f, bus_to=t, name=row[22], code=key, r=r, x=x,
                               b=_number(row, 5, context) / grid.Sbase, rate=rate, active=sf and st)
                grid.add_line(obj)
                for side, target, offset, connected in (('from', f, 9, sf), ('to', t, 12, st)):
                    enabled = _flag(row, offset, context)
                    g, b = _number(row, offset + 1, context), _number(row, offset + 2, context)
                    if enabled or g or b:
                        grid.add_shunt(target, dev.Shunt(name=f'Terminal {key} {side}',
                                         code=f'terminal|{key}|{side}', G=g * grid.Sbase, B=b * grid.Sbase,
                                         active=enabled and connected))
            if rate > 0:
                obj.contingency_factor = rate3 / rate
                obj.protection_rating_factor = rate3 / rate
            branch_objects[key] = obj
    for row, context in records('line_shunts', 9):
        key = f'{bus_id(row[0])}|{bus_id(row[1])}|{row[2]}'
        if key not in branch_objects:
            raise ValueError(f'{context}: terminal shunt has no matching directed branch {key}.')
        obj = branch_objects[key]
        for side, target, offset in (('from', obj.bus_from, 3), ('to', obj.bus_to, 6)):
            g, b = _number(row, offset + 1, context), _number(row, offset + 2, context)
            grid.add_shunt(target, dev.Shunt(name=f'Additional terminal {key} {side}',
                           code=f'additional_terminal|{key}|{side}', G=g * grid.Sbase, B=b * grid.Sbase,
                           active=_flag(row, offset, context) and obj.active))
    if not buses:
        raise ValueError('STM contains no buses.')
    if prm_source:
        warning('PRM commands have been preserved for consultation, not executed. '
                'Only explicitly selected supported GUI options may be applied after import. '
                'This conversion does not establish equivalence with an ONS EMS study.')
    return grid
