# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Public, synthetic STM examples; no operational network data is used."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from VeraGridEngine import Devices as dev
from VeraGridEngine.IO.file_open import FileOpen, FileOpenOptions, determine_file_type
from VeraGridEngine.IO.file_save import FileSave
from VeraGridEngine.IO.ons_ems.stm_parser import SECTION_NAMES, bus_id, inspect_stm, open_stm, read_prm, read_stm
from VeraGridEngine.enumerations import FileType, GeneratorControlMode
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions


FIXTURES = Path(__file__).parent / 'data' / 'ons_ems'


def _write_stm(tmp_path, transform=None):
    data, _ = read_stm(FIXTURES / 'ac_network_v5.stm')
    sections = {name: [row for row, _ in data[name]] for name in SECTION_NAMES}
    if transform:
        transform(sections)
    import csv
    import io
    text = io.StringIO()
    text.write('5\nBASE 100.0\nSynthetic AC test\n')
    writer = csv.writer(text, quotechar="'", quoting=csv.QUOTE_ALL, lineterminator='\n')
    for name in SECTION_NAMES:
        writer.writerows(sections[name])
        text.write(f'0 / {name}\n')
    path = tmp_path / 'modified.stm'
    path.write_text(text.getvalue(), encoding='utf-8')
    return path


def _reference_grid():
    """Independently construct the same native electrical network."""
    grid = dev.MultiCircuit(Sbase=100, fbase=60)
    first = dev.Bus(name='Source / west', code='1', Vnom=230, is_slack=True, Vm0=1.02, vmax=1.1, vmin=.9)
    second = dev.Bus(name='Load', code='2', Vnom=230, Vm0=.98, Va0=np.deg2rad(-1), vmax=1.1, vmin=.9)
    third = dev.Bus(name='Lower voltage', code='3.1', Vnom=115, Vm0=1.0, Va0=np.deg2rad(-2), vmax=1.1, vmin=.9)
    fourth = dev.Bus(name='Disconnected', code='4', Vnom=115, active=False, Vm0=1, vmax=1.1, vmin=.9)
    for bus in (first, second, third, fourth):
        grid.add_bus(bus)
    grid.add_generator(first, dev.Generator(code='1|01', P=40, Q=3, Qmax=100, Qmin=-100,
                       vset=1.02, Snom=150, Pmax=140, Pmin=0, freq=60))
    grid.add_generator(fourth, dev.Generator(code='4|01', P=2, Q=1, Qmax=10, Qmin=-10,
                       vset=1, Snom=20, Pmax=15, Pmin=0, freq=60, active=False,
                       control_mode=GeneratorControlMode.Q))
    grid.add_load(second, dev.Load(code='2|01', P=20, Q=4, Ir=3/.98, Ii=-.8/.98,
                                  G=2/.98**2, B=-.4/.98**2))
    grid.add_load(third, dev.Load(code='3.1|01', P=8, Q=2))
    grid.add_shunt(second, dev.Shunt(code='bus_shunt|2', B=1.5))
    grid.add_line(dev.Line(bus_from=first, bus_to=second, r=.01, x=.08, b=.03, rate=100))
    grid.add_shunt(first, dev.Shunt(G=.01, B=.02))
    grid.add_shunt(first, dev.Shunt(B=.03))
    grid.add_transformer2w(dev.Transformer2W(bus_from=second, bus_to=third, r=.005, x=.05,
                                           HV=230, LV=115, rate=100, nominal_power=100, tap_module=1.03))
    return grid


def test_stm_native_devices_and_units():
    grid = open_stm(FIXTURES / 'ac_network_v5.stm')
    assert grid.Sbase == 100 and grid.fBase == 60
    assert [b.code for b in grid.buses] == ['1', '2', '3.1', '4']
    assert grid.buses[0].name == 'Source / west'
    assert grid.buses[0].is_slack and not grid.buses[3].active
    assert grid.buses[1].Va0 == np.deg2rad(-1)
    assert grid.buses[0].area.code == '1'
    assert grid.buses[0].zone.code == '1'
    assert grid.buses[0].substation.code == '1'
    assert len(grid.generators) == 2 and not grid.generators[1].active
    assert grid.generators[0].freq == 60
    assert len(grid.loads) == 2
    assert grid.loads[0].Ir == 3/.98
    assert grid.loads[0].Ii == -.8/.98
    assert grid.loads[0].G == 2/.98**2
    assert grid.loads[0].B == -.4/.98**2
    assert len(grid.lines) == 1 and grid.lines[0].B == .03
    assert grid.lines[0].contingency_factor == 1.2
    assert len(grid.transformers2w) == 1
    assert grid.transformers2w[0].tap_module == 1.03
    assert sum(s.B for s in grid.shunts) == pytest.approx(1.55)
    assert sum(s.G for s in grid.shunts) == .01


def test_stm_power_flow_matches_independent_native_network():
    results = []
    for grid in (open_stm(FIXTURES / 'ac_network_v5.stm'), _reference_grid()):
        driver = PowerFlowDriver(grid, PowerFlowOptions(retry_with_other_methods=False, tolerance=1e-9))
        driver.run()
        assert driver.results.converged
        results.append(driver.results)
    np.testing.assert_allclose(results[0].voltage, results[1].voltage, atol=1e-12, rtol=0)
    np.testing.assert_allclose(results[0].Sf, results[1].Sf, atol=1e-10, rtol=0)
    np.testing.assert_allclose(results[0].Sbus, results[1].Sbus, atol=1e-10, rtol=0)


def test_stm_and_prm_file_open_round_trip(tmp_path):
    stm, prm = FIXTURES / 'ac_network_v5.stm', FIXTURES / 'options.prm'
    assert determine_file_type(str(stm)) == FileType.ONS_EMS_STM
    assert determine_file_type([str(prm), str(stm)]) == FileType.ONS_EMS_STM
    assert determine_file_type([str(stm)]) == FileType.ONS_EMS_STM
    grid = FileOpen([str(prm), str(stm)]).open()
    original = deepcopy(grid.import_metadata)
    assert original['parameters']['PFMAXITER'] == 30
    assert original['parameters']['PFFLTSTRT'] is False
    assert original['assignments'][0]['status'] == 'preserved, not applied'
    assert original['stm']['source'] == stm.name
    assert original['prm']['text'] == prm.read_bytes().decode('utf-8')
    assert 'D:\\' not in str(original)
    output = tmp_path / 'network.veragrid'
    log = FileSave(grid, str(output)).save()
    assert log.error_count() == 0
    reopened = FileOpen(str(output)).open()
    assert reopened.import_metadata == original
    assert [b.code for b in reopened.buses] == [b.code for b in grid.buses]
    assert reopened.loads[0].Ir == grid.loads[0].Ir
    assert reopened.loads[0].B == grid.loads[0].B
    assert reopened.transformers2w[0].tap_module == grid.transformers2w[0].tap_module


def test_prm_is_optional_explicit_and_not_evaluated(tmp_path):
    stm = FIXTURES / 'ac_network_v5.stm'
    grid = FileOpen(str(stm)).open()
    assert grid.import_metadata['prm'] is None
    selected = FileOpen(str(stm), options=FileOpenOptions(ems_prm_path=str(FIXTURES / 'options.prm'))).open()
    assert selected.import_metadata['prm'] is not None
    prm = tmp_path / 'unsafe.prm'
    prm.write_text("PFMAXITER = 20\nPFMAXITER = 30\nCUSTOM = __import__('os').abort()\ninclude file.dat\n", encoding='utf-8')
    parameters, history, _ = read_prm(prm)
    assert parameters['PFMAXITER'] == 30
    assert parameters['CUSTOM'] == "__import__('os').abort()"
    assert len(history) == 4
    assert all(row['status'] == 'preserved, not applied' for row in history)


@pytest.mark.parametrize('name', ['dclinks', 'vsc', 'series_caps'])
def test_unsupported_sections_refuse_whole_conversion(tmp_path, name):
    path = _write_stm(tmp_path, lambda data: data[name].append(['1']))
    with pytest.raises(NotImplementedError, match=name):
        open_stm(path)


@pytest.mark.parametrize('change, error', [
    (lambda d: d['buses'].append(d['buses'][0]), 'duplicate bus'),
    (lambda d: d['buses'][0].__setitem__(2, 'nan'), 'non-finite'),
    (lambda d: d['buses'][0].__setitem__(9, '0'), 'voltage seed'),
    (lambda d: d['loads'][0].__setitem__(0, '999'), 'missing bus'),
    (lambda d: d['loads'][0].__setitem__(2, '2'), 'must be 0 or 1'),
    (lambda d: d['generators'][0].__setitem__(7, '2'), 'remote voltage'),
    (lambda d: d['transformers'][0].__setitem__(20, '1'), 'automatic transformer'),
    (lambda d: d['lines'][0].__setitem__(16, '0'), 'single open terminal'),
    (lambda d: d['shunts'][0].__setitem__(1, '1'), 'switched shunts'),
    (lambda d: d['line_shunts'][0].__setitem__(2, '09'), 'no matching directed branch'),
])
def test_invalid_or_unsupported_data_is_not_silently_dropped(tmp_path, change, error):
    with pytest.raises((ValueError, NotImplementedError), match=error):
        open_stm(_write_stm(tmp_path, change))


@pytest.mark.parametrize('text, message', [
    ('4\nBASE 100\nHeader\n', 'version 5'),
    ('5\nBASE nan\nHeader\n', 'positive and finite'),
    ('5\nBASE 100\nHeader\n0 /\n', 'incomplete sections'),
    ("5\nBASE 100\nHeader\n1,'unterminated\n", 'malformed quoted'),
])
def test_malformed_stm(tmp_path, text, message):
    path = tmp_path / 'bad.stm'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        read_stm(path)


@pytest.mark.parametrize('value, expected', [('001', '1'), ('001.02', '1.2'), ('1.0', '1')])
def test_bus_sections_are_stable(value, expected):
    assert bus_id(value) == expected


def test_old_circuit_metadata_remains_readable():
    grid = dev.MultiCircuit()
    grid.parse({'name': 'Legacy', 'Sbase': 100, 'fBase': 50})
    assert grid.import_metadata == {}


def test_provenance_is_copied_and_saved_with_native_multiverse(tmp_path):
    from VeraGridEngine.Devices.multiverse import MultiVerse
    grid = open_stm(FIXTURES / 'ac_network_v5.stm', FIXTURES / 'options.prm')
    copied = grid.copy()
    assert copied.import_metadata == grid.import_metadata
    copied.import_metadata['new'] = 'does not change the original'
    assert 'new' not in grid.import_metadata
    multiverse = MultiVerse(current_model=grid)
    path = tmp_path / 'scenarios.veragrid'
    assert FileSave(grid, str(path), multiverse=multiverse).save().error_count() == 0
    reopened = FileOpen(str(path)).open()
    assert reopened.import_metadata == grid.import_metadata
    data = grid.to_dict()
    data['import_metadata']['example'] = 1
    assert 'example' not in grid.import_metadata


def test_read_only_inventory_preserves_unsupported_records(tmp_path):
    path = _write_stm(tmp_path, lambda data: data['dclinks'].append(['1', '2', 'Raw record']))
    source, _ = read_stm(path)
    metadata = inspect_stm(path, FIXTURES / 'options.prm')
    assert any('dclinks' in reason for reason in metadata['unsupported_features'])
    assert metadata['section_records']['dclinks'][0]['fields'] == ['1', '2', 'Raw record']
    for name in SECTION_NAMES:
        assert [item['fields'] for item in metadata['section_records'][name]] == [row for row, _ in source[name]]
    assert metadata['applied_gui_options'] == {}


@pytest.mark.parametrize('name, buses, dc_records', [
    ('Soulth SIN Brazil Example', 1958, 12), ('SIN Brazil Example', 10614, 42)])
def test_supplied_real_examples_are_explicit_inspection_cases(name, buses, dc_records):
    folder = Path(__file__).resolve().parents[3] / 'examples' / 'ons_ems'
    metadata = inspect_stm(folder / (name + '.stm'), folder / (name + '.prm'))
    assert len(metadata['section_records']['buses']) == buses
    assert len(metadata['section_records']['dclinks']) == dc_records
    assert metadata['unsupported_features']
    assert metadata['applied_gui_options'] == {}
    with pytest.raises(NotImplementedError):
        open_stm(folder / (name + '.stm'), folder / (name + '.prm'))


def test_ambiguous_multi_file_selection_is_rejected():
    stm = str(FIXTURES / 'ac_network_v5.stm')
    with pytest.raises(ValueError, match='exactly one STM'):
        FileOpen([stm, stm]).open()
