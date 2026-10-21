"""Indexed RMS storage tests; desktop navigation is a separate acceptance step."""
from io import BytesIO
import hashlib
import json
import os
import zipfile
import zlib

import numpy as np
import pandas as pd
import pytest

from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Devices.multiverse import MultiVerse
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.Injections.generator import Generator
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.IO.veragrid import zip_interface
from VeraGridEngine.IO.veragrid.rms_archive import ZipRmsValues, would_discard_rms_source
from VeraGridEngine.IO.veragrid.result_store import ChunkedArray, ResultReader, write_array
from VeraGridEngine.Simulations.Rms.rms_results import RmsResults
from VeraGridEngine.Simulations.driver_template import DriverToSave
from VeraGridEngine.Utils.Symbolic.symbolic import Var
from VeraGridEngine.enumerations import SimulationTypes
from VeraGrid.Session.session import SimulationSession


def make_result():
    grid = MultiCircuit(name="Public RMS archive test")
    bus = Bus(name="Bus", Vnom=20.0)
    grid.add_bus(bus)
    generator = Generator(name="Generator")
    grid.add_generator(bus, generator)
    variables = [Var(name=f"Channel {i}", uid=100 + i) for i in range(4)]
    result = RmsResults(
        time_array=pd.date_range("2020-01-01", periods=11, freq="1ms"),
        rms_events_group_names=np.array(["Complete", "Cancelled", "Not started"]),
        rms_events_group_idtags=np.array(["complete", "cancelled", "unstarted"]),
        variables=variables, uid2idx={var.uid: i for i, var in enumerate(variables)},
        vars_glob_name2uid={var.name: var.uid for var in variables},
        devices_vars_info={generator: variables},
        initial_parameter_value_maps=[{generator.idtag + ":P": 1.0}, {}, {}],
        parameter_value_maps=[{generator.idtag + ":P": 1.5}, {}, {}],
        has_event_group_results=np.array([True, True, False]))
    result.values[:] = np.arange(result.values.size).reshape(result.values.shape) / 7
    result.values[0, 0, 0] = -0.0
    result.values[5:, :, 1] = np.nan
    result.values[:, :, 2] = np.nan
    result.well_initialized[:] = [True, True, False]
    result.converged[:] = [True, False, False]
    result.cancelled[:] = [False, True, False]
    return grid, result


def save(path, grid, result):
    saved = DriverToSave("Archive test", SimulationTypes.RmsDynamic_run, result, Logger())
    zip_interface.save_veragrid_data_to_zip(str(path), grid, [saved], {})
    return saved


def load(path):
    return zip_interface.load_session_driver_objects(
        str(path), "Archive test", SimulationTypes.RmsDynamic_run.value)


def assert_bits_equal(left, right):
    np.testing.assert_array_equal(np.asarray(left).view(np.uint64), np.asarray(right).view(np.uint64))


def rewrite_archive(path, change):
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    change(entries)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def test_legacy_default_and_opt_in_reopen_restore_metadata(tmp_path):
    grid, result = make_result()
    path = tmp_path / "result.veragrid"
    save(path, grid, result)
    assert isinstance(load(path)["values"], np.ndarray)
    result.archive_chunked = True
    save(path, grid, result)
    stored = load(path)
    assert isinstance(stored["values"], ChunkedArray)
    assert stored["values"].reader.stats["members"] == 0
    assert "Archive test" in zip_interface.get_session_tree(str(path))
    session = SimulationSession()
    logger = session.register_driver_from_disk_data(grid, SimulationTypes.RmsDynamic_run.value, stored)
    assert not logger.has_logs()
    _, reopened = session.rms_dynamic_simulation
    assert isinstance(reopened.values, ChunkedArray)
    assert reopened.values.reader.stats["members"] == 0
    assert reopened.archive_chunked is True
    assert reopened.initial_parameter_value_maps == result.initial_parameter_value_maps
    assert reopened.parameter_value_maps == result.parameter_value_maps
    assert reopened.vars_glob_name2uid == result.vars_glob_name2uid
    assert len(reopened.devices_vars_info) == 1
    for name in ("has_event_group_results", "well_initialized", "converged", "cancelled"):
        np.testing.assert_array_equal(getattr(reopened, name), getattr(result, name))
    np.testing.assert_array_equal(reopened.time_array.asi8, result.time_array.asi8)
    # The existing channel consultation API also stays lazy.
    assert_bits_equal(reopened.get_vars_data([reopened.variables[1]], 1)[:, 0], result.values[:, 1, 1])
    assert reopened.values.reader.stats["members"] == 1
    assert_bits_equal(reopened.values, result.values)


@pytest.mark.parametrize("key", [
    (slice(3, 9), 2, 0), (slice(None, None, -2), -1, -2),
    (slice(4, 4), slice(None), 0), (5, 2, 0),
    (slice(None), slice(None, None, -1), slice(None, None, -1)),
    (Ellipsis, 1), (slice(0, 11, 5), 0, 0),
])
def test_basic_indexing_matches_dense_exactly(tmp_path, key):
    grid, result = make_result()
    result.archive_chunked = True
    path = tmp_path / "indexed.veragrid"
    save(path, grid, result)
    lazy = load(path)["values"]
    expected = result.values[key]
    actual = lazy[key]
    assert actual.shape == np.shape(expected)
    assert_bits_equal(actual, expected)


def test_channel_request_does_not_read_other_channels_or_groups(tmp_path):
    # Same geometry/dtype/selection used by the developed partial-results test.
    expected = (np.arange(120*32*2).reshape(120,32,2)/19
                + 1j*np.arange(120*32*2).reshape(120,32,2)/29).astype('>c16')
    path = tmp_path / 'selected.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        descriptor = write_array(archive, 'sessions/Test/RMS Dynamic', expected,
                                 str(path), block_bytes=1024, channels=4)
    lazy = ChunkedArray(ResultReader(path), descriptor)
    actual = lazy[7:37, 9, 1]
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected[7:37,9,1].tobytes()
    assert lazy.reader.stats['members'] < len(descriptor['blocks'])
    assert lazy.reader.stats['decoded_bytes'] < expected.nbytes // 4
    reads = lazy.reader.stats['members']
    actual[:] = 123.0
    assert lazy[7:37,9,1].tobytes() == expected[7:37,9,1].tobytes()
    assert lazy.reader.stats['members'] == reads
    assert lazy.reader.stats['cache_hits'] > 0
    assert lazy[:,[9,8],1].tobytes() == expected[:,[9,8],1].tobytes()
    with pytest.raises(IndexError):
        lazy[:, 99, 0]


def test_atomic_in_place_resave_and_explicit_dense_export(tmp_path):
    grid, result = make_result()
    expected = result.values.copy()
    result.archive_chunked = True
    path = tmp_path / "in_place.veragrid"
    save(path, grid, result)
    stored = load(path)
    result.parse_saved_data(grid, stored)
    assert isinstance(result.values, ChunkedArray)
    save(path, grid, result)
    assert isinstance(result.values, ChunkedArray)
    assert_bits_equal(result.values[:, 0, 0], expected[:, 0, 0])
    result.archive_chunked = False
    save(path, grid, result)
    assert isinstance(result.values, np.ndarray)
    assert_bits_equal(result.values, expected)
    assert_bits_equal(load(path)["values"], expected)


def test_failed_chunk_save_preserves_original_archive_and_reader(tmp_path, monkeypatch):
    grid, result = make_result()
    result.archive_chunked = True
    path = tmp_path / "safe.veragrid"
    save(path, grid, result)
    result.parse_saved_data(grid, load(path))
    original = path.read_bytes()
    def fail(*_args):
        raise OSError("Synthetic storage interruption")
    monkeypatch.setattr(zip_interface, "write_array", fail)
    with pytest.raises(OSError, match="storage interruption"):
        save(path, grid, result)
    assert path.read_bytes() == original
    assert_bits_equal(result.values[:, 1, 0], make_result()[1].values[:, 1, 0])
    assert list(tmp_path.glob("*.tmp")) == []


def test_current_source_omission_is_detected_before_gui_save(tmp_path):
    grid, result = make_result()
    result.archive_chunked = True
    path = tmp_path / "source.veragrid"
    saved = save(path, grid, result)
    result.parse_saved_data(grid, load(path))
    assert would_discard_rms_source([saved], [], str(path))
    if os.name == "nt":
        assert would_discard_rms_source([saved], [], str(path).swapcase())
    assert not would_discard_rms_source([saved], [saved], str(path))
    assert not would_discard_rms_source([saved], [], str(tmp_path / "other.veragrid"))


def test_source_replacement_is_not_silently_used(tmp_path):
    grid, result = make_result()
    result.archive_chunked = True
    path = tmp_path / "replaced.veragrid"
    save(path, grid, result)
    lazy = load(path)["values"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="mudou"):
        lazy[:, 0, 0]


@pytest.mark.parametrize('fault', ['version', 'missing', 'dtype', 'shape', 'path', 'duplicate'])
def test_invalid_index_is_rejected_before_session_registration(tmp_path, fault):
    grid, result = make_result()
    result.archive_chunked = True
    path = tmp_path / 'invalid.veragrid'
    save(path, grid, result)
    def change(entries):
        index_name = next(name for name in entries if name.endswith('/result_index.json'))
        index = json.loads(entries[index_name])
        descriptor = index['arrays']['values']
        if fault == 'version': index['version'] = 999
        elif fault == 'missing': descriptor['blocks'].pop()
        elif fault == 'dtype': descriptor['dtype'] = 'O'
        elif fault == 'shape': descriptor['shape'] = [-1, 4, 3]
        elif fault == 'path': descriptor['blocks'][0]['member'] = '../../payload.npy'
        elif fault == 'duplicate': descriptor['blocks'][1] = descriptor['blocks'][0]
        entries[index_name] = json.dumps(index).encode()
    rewrite_archive(path, change)
    with pytest.raises(ValueError):
        load(path)


def test_corrupt_chunk_fails_on_access_without_loading_other_groups(tmp_path):
    grid, result = make_result()
    result.archive_chunked = True
    path = tmp_path / 'corrupt.veragrid'
    save(path, grid, result)
    def change(entries):
        name = next(name for name in entries if '/values/00000000.npy' in name)
        entries[name] = entries[name][:-1] + bytes([entries[name][-1] ^ 1])
    rewrite_archive(path, change)
    # CRC in the persisted descriptor differs, so discovery refuses the source.
    with pytest.raises(ValueError, match='Integridade'):
        load(path)


def test_invalid_npy_header_shape_is_rejected_before_allocation(tmp_path):
    grid, result = make_result()
    result.archive_chunked = True
    path = tmp_path / 'header.veragrid'
    save(path, grid, result)
    def change(entries):
        index_name = next(name for name in entries if name.endswith('/result_index.json'))
        index = json.loads(entries[index_name])
        block = index['arrays']['values']['blocks'][0]
        stream = BytesIO()
        np.lib.format.write_array_header_1_0(
            stream, {'descr': '<f8', 'fortran_order': False, 'shape': (10**12, 4, 1)})
        stream.write(np.zeros((11,4,1), dtype=float).tobytes())
        payload = stream.getvalue()
        entries[block['member']] = payload
        block['nbytes'] = len(payload)
        block['crc32'] = zlib.crc32(payload)
        entries[index_name] = json.dumps(index).encode()
    rewrite_archive(path, change)
    lazy = load(path)['values']
    with pytest.raises(ValueError, match='Cabeçalho'):
        lazy[:, 0, 0]


def test_array_like_writer_does_not_materialize_full_tensor(tmp_path):
    expected = np.arange(60, dtype=float).reshape(10, 3, 2)
    class SlicedValues:
        shape, dtype = expected.shape, expected.dtype
        def __array__(self, *_args, **_kwargs):
            raise AssertionError('Full RMS materialization is forbidden')
        def __getitem__(self, key):
            assert all(isinstance(selector, slice) for selector in key)
            assert key[0].stop - key[0].start <= 4
            return expected[key]
    path = tmp_path / 'protocol.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        descriptor = write_array(archive, 'sessions/Test/RMS Dynamic', SlicedValues(),
                                 str(path), block_bytes=64, channels=2)
    assert_bits_equal(ChunkedArray(ResultReader(path), descriptor), expected)


def test_existing_archive_grid_open_does_not_unpack_chunk_payloads(tmp_path, monkeypatch):
    grid, result = make_result()
    result.archive_chunked = True
    path = tmp_path / "grid_open.veragrid"
    save(path, grid, result)
    original = zipfile.ZipFile.open
    def checked(archive, name, *args, **kwargs):
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert not filename.startswith("rms_results/")
        assert not filename.endswith("/values.rms.json")
        assert "/RMS Dynamic/values/" not in filename
        assert not filename.endswith("/RMS Dynamic/result_index.json")
        return original(archive, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", checked)
    data, _, has_multiverse = zip_interface.get_frames_from_zip(str(path))
    assert "model_data" in data and not has_multiverse


def test_flat_to_multiverse_and_in_place_save_rebinds_active_results(tmp_path):
    grid, result = make_result()
    expected = result.values.copy()
    result.archive_chunked = True
    path = tmp_path / "scenarios.veragrid"
    saved = save(path, grid, result)
    result.parse_saved_data(grid, load(path))
    multiverse = MultiVerse(grid)
    for _ in range(2):
        zip_interface.save_veragrid_multiverse_data_to_zip(
            str(path), {}, multiverse, active_grid_idtag=grid.idtag,
            active_sessions_data=[saved])
        assert result.values.descriptor["blocks"][0]["member"].startswith("multiverse/")
        assert_bits_equal(result.values[:, 0, 0], expected[:, 0, 0])
        stored = load(path)
        assert isinstance(stored["values"], ChunkedArray)
        assert_bits_equal(stored["values"], expected)


def test_real_rms_run_survives_chunked_session_roundtrip(tmp_path):
    from Base.test_dynamic_results_save_load import _build_rms_driver
    driver = _build_rms_driver()
    assert driver.results.converged.all()
    expected = driver.results.values.copy()
    driver.results.archive_chunked = True
    path = tmp_path / "real_run.veragrid"
    zip_interface.save_veragrid_data_to_zip(str(path), driver.grid, [driver.get_save_data()], {})
    stored = zip_interface.load_session_driver_objects(str(path), driver.name, driver.tpe.value)
    session = SimulationSession()
    logger = session.register_driver_from_disk_data(driver.grid, driver.tpe.value, stored)
    assert not logger.has_logs()
    _, result = session.rms_dynamic_simulation
    assert result.values.reader.stats["members"] == 0
    assert result.initial_parameter_value_maps == driver.results.initial_parameter_value_maps
    assert result.parameter_value_maps == driver.results.parameter_value_maps
    assert_bits_equal(result.values, expected)


def test_developed_reader_lru_close_and_replacement_contract(tmp_path):
    path = tmp_path / 'reader.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('a.json', 'a' * 1000)
        archive.writestr('b.json', 'b' * 1000)
    reader = ResultReader(path, cache_bytes=1500)
    reader.read('a.json'); reader.read('b.json')
    assert reader.stats['cache_bytes'] <= 1500
    assert reader.stats['evictions'] == 1
    reader.close()
    with pytest.raises(ValueError, match='encerrada'):
        reader.read('b.json')
    reader = ResultReader(path)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('a.json', 'changed')
    with pytest.raises(ValueError, match='mudou'):
        reader.read('a.json')


def test_previously_published_prototype_remains_readable_and_exports_native_tiles(tmp_path):
    grid, result = make_result()
    expected = result.values.copy()
    legacy = tmp_path / 'previous-prototype.zip'
    blocks = []
    with zipfile.ZipFile(legacy, 'w') as archive:
        for group in range(expected.shape[2]):
            for variable in range(expected.shape[1]):
                stream = BytesIO()
                np.save(stream, expected[:,variable,group], allow_pickle=False)
                payload = stream.getvalue()
                member = f'rms_results/{"0"*32}/g{group}/v{variable}/t0.npy'
                archive.writestr(member, payload)
                blocks.append(dict(member=member,sha256=hashlib.sha256(payload).hexdigest()))
        archive.writestr('values.rms.json', json.dumps(dict(
            format='VeraGrid.RmsColumns',version=1,shape=expected.shape,dtype='<f8',
            chunk_rows=expected.shape[0],blocks=blocks)))
    result.values = ZipRmsValues(legacy, 'values.rms.json')
    assert_bits_equal(result.values, expected)
    result.archive_chunked = True
    target = tmp_path / 'native-tiles.veragrid'
    save(target, grid, result)
    assert isinstance(result.values, ChunkedArray)
    assert_bits_equal(result.values, expected)
    assert_bits_equal(load(target)['values'], expected)
