"""Qualify the existing development lossless store against upstream BE output."""
from copy import deepcopy
from contextlib import closing
import json
import shutil
import sqlite3
import zipfile
import numpy as np
import pytest
import VeraGridEngine.api as vg
from Base.test_dynamic_results_save_load import _build_rms_driver
from VeraGridEngine.Simulations.Rms.rms_driver import RmsSimulationDriver
from VeraGridEngine.Simulations.Rms import trajectory_store
from VeraGridEngine.Simulations.Rms.recovery import export_recovery
from VeraGridEngine.Simulations.Rms.trajectory_store import (
    RmsTrajectoryStore, RmsStoredRows, BUFFER_BYTES, _encode_tile, _decode_tile)
from VeraGridEngine.Simulations.Rms.scenario_values import RmsScenarioValues
from VeraGridEngine.enumerations import RmsProblemTypes
from VeraGridEngine.IO.veragrid.zip_interface import load_session_driver_objects
from VeraGridEngine.IO.veragrid.result_store import ChunkedArray
from VeraGrid.Session.session import SimulationSession


@pytest.fixture(scope='module')
def public_case():
    return _build_rms_driver()


def driver(public_case, tmp_path, mode='lossless_disk', problem=RmsProblemTypes.PowerBalance):
    grid, options, flow = deepcopy((public_case.grid, public_case.options, public_case.pf_results))
    options.trajectory_storage = mode
    options.trajectory_directory = str(tmp_path)
    options.problem_type = problem
    return RmsSimulationDriver(grid, options, flow)


def reopen_recovery(location, output):
    info = export_recovery(location, output)
    values = load_session_driver_objects(str(output), 'Recuperação RMS', 'RMS Dynamic')
    return info, values


@pytest.mark.parametrize('problem', [RmsProblemTypes.PowerBalance,
    RmsProblemTypes.PowerBalanceVectorized, RmsProblemTypes.PowerBalanceFullVectorized])
def test_disk_integrators_equal_memory_and_native_archive(public_case, tmp_path, problem):
    memory = driver(public_case, tmp_path, 'memory', problem)
    memory.run()
    stored = driver(public_case, tmp_path, problem=problem)
    stored.run()
    assert isinstance(stored.results.values, RmsScenarioValues)
    np.testing.assert_array_equal(stored.results.values, memory.results.values)
    assert stored.results.converged.all()
    assert stored.trajectory_store._writer is None
    info, recovered = reopen_recovery(stored.results.trajectory_directory, tmp_path / 'recovered.veragrid')
    assert info['complete_persisted_metadata']
    assert isinstance(recovered['values'], ChunkedArray)
    np.testing.assert_array_equal(recovered['values'], stored.results.values)
    assert np.asarray(recovered['recovered']).all()
    archive = tmp_path / 'disk.veragrid'
    vg.save_file(stored.grid, str(archive), drivers_to_save=[stored.get_save_data()])
    with zipfile.ZipFile(archive) as zipped:
        assert any(n.endswith('/result_index.json') for n in zipped.namelist())
        assert not any(n.endswith('/values.npy') for n in zipped.namelist())
    loaded = load_session_driver_objects(str(archive), stored.name, stored.tpe.value)
    np.testing.assert_array_equal(loaded['values'], memory.results.values)
    assert not np.asarray(loaded['recovered']).any()


def test_disk_cancel_preserves_committed_clock_and_prefix(public_case, tmp_path):
    memory = driver(public_case, tmp_path, 'memory')
    memory.run()
    stored = driver(public_case, tmp_path)
    class Cancel:
        def emit(self, percent):
            if stored.results is not None and 25 <= percent < 90:
                stored.cancel()
    stored.progress_signal = Cancel()
    stored.run()
    accepted = int(stored.results.valid_samples[0])
    assert 1 < accepted < stored.results.nt
    np.testing.assert_array_equal(stored.results.values[:accepted], memory.results.values[:accepted])
    assert np.isnan(stored.results.values[accepted:]).all()
    assert stored.results.cancelled[0] and not stored.results.converged[0]
    info, recovered = reopen_recovery(stored.results.trajectory_directory, tmp_path / 'cancelled.veragrid')
    assert info['counts'] == [accepted]
    np.testing.assert_array_equal(recovered['values'], stored.results.values)
    assert np.asarray(recovered['cancelled']).all() and not np.asarray(recovered['converged']).any()


def test_native_session_restore_and_resave_do_not_materialize_tensor(public_case, tmp_path, monkeypatch):
    stored = driver(public_case, tmp_path)
    stored.run()
    archive = tmp_path / 'first.veragrid'
    vg.save_file(stored.grid, str(archive), drivers_to_save=[stored.get_save_data()])
    data = load_session_driver_objects(str(archive), stored.name, stored.tpe.value)
    def forbid(*args, **kwargs):
        raise AssertionError('The native session must keep saved RMS values tiled')
    monkeypatch.setattr(ChunkedArray, '__array__', forbid)
    session = SimulationSession()
    log = session.register_driver_from_disk_data(stored.grid, stored.tpe.value, data)
    assert not log.has_errors()
    restored_driver, restored = session.rms_dynamic_simulation
    assert isinstance(restored.values, ChunkedArray) and restored.archive_chunked
    assert restored.parameter_value_maps == stored.results.parameter_value_maps
    assert restored.initial_parameter_value_maps == stored.results.initial_parameter_value_maps
    assert restored.valid_samples.tolist() == stored.results.valid_samples.tolist()
    saved_again = tmp_path / 'second.veragrid'
    vg.save_file(stored.grid, str(saved_again), drivers_to_save=[restored_driver.get_save_data()])
    np.testing.assert_array_equal(restored.values[:, :, :], stored.results.values[:, :, :])


@pytest.mark.parametrize('problem', [RmsProblemTypes.PowerBalance, RmsProblemTypes.PowerBalanceVectorized,
                                    RmsProblemTypes.PowerBalanceFullVectorized])
def test_early_numerical_failure_keeps_accepted_prefix(public_case, tmp_path, problem):
    stored = driver(public_case, tmp_path, problem=problem)
    stored.grid.rms_events[0].value = -1000.0
    stored.options.max_iter = 25
    stored.run()
    count = int(stored.results.valid_samples[0])
    assert 1 < count < stored.results.nt
    assert stored.results.has_event_group_results[0]
    assert not stored.results.converged[0] and not stored.results.cancelled[0]
    assert np.isfinite(stored.results.values[:count, :, 0]).all()
    assert np.isnan(stored.results.values[count:, :, 0]).all()
    info, recovered = reopen_recovery(stored.results.trajectory_directory, tmp_path / 'failed.veragrid')
    assert info['counts'] == [count]
    assert not np.asarray(recovered['converged']).any()
    np.testing.assert_array_equal(recovered['values'], stored.results.values)


def test_actual_xor_byte_plane_codec_is_bit_exact():
    bits = np.array([0, 1 << 63, 0x7ff8000000000001, 0xfff8000000000017,
                     0x7ff0000000000000, 0xfff0000000000000, 1, 0x0010000000000000], dtype='<u8')
    values = np.tile(bits.view('<f8'), (19, 2))
    key, payload = _encode_tile(values)
    decoded = _decode_tile(key, payload, *values.shape)
    np.testing.assert_array_equal(decoded.view('<u8'), values.view('<u8'))
    assert not decoded.flags.writeable
    with pytest.raises(ValueError, match='checksum'):
        _decode_tile('0' * 64, payload, *values.shape)
    with pytest.raises(ValueError, match='compressed size'):
        _decode_tile(key, payload + b'junk', *values.shape)


def test_bounded_staging_and_atomic_clock_commit(tmp_path):
    store = RmsTrajectoryStore(tmp_path / 'bounded', create=True)
    rows = store.create_rows('one', (10_000_001, 30_010))
    assert rows.resident_bytes <= BUFFER_BYTES + 30_010 * 8
    times = np.arange(rows.rows_per_block + 3) * 0.001
    for i in range(len(times)):
        rows[i, :] = i + np.arange(30_010) / 16
        rows.accept_until(i + 1, times[:i + 1])
    prefix = rows.accepted_prefix(len(times))
    with store.reader() as connection:
        committed = connection.execute('SELECT committed FROM rms_groups').fetchone()[0]
        clocks = connection.execute('SELECT sum(n_rows) FROM clocks').fetchone()[0]
    assert committed == clocks == len(times)
    np.testing.assert_array_equal(prefix[-1, [0, 16, 30_009]], times.size - 1 + np.array([0,16,30_009]) / 16)
    assert prefix.resident_bytes <= BUFFER_BYTES + 30_010 * 8
    store.close()


def test_sparse_scenarios_preserve_layout_and_nan_tail(tmp_path):
    store = RmsTrajectoryStore(tmp_path / 'scenarios', create=True)
    tensor = RmsScenarioValues((7, 3, 3))
    for scenario, count in ((0, 3), (2, 7)):
        rows = store.create_rows(str(scenario), (7, 3))
        for i in range(count):
            rows[i, :] = scenario * 100 + np.arange(3, dtype=float) + i
            rows.accept_until(i + 1, np.arange(i + 1) * 0.01)
        tensor.attach(scenario, rows.accepted_prefix(count), count)
    assert np.isnan(tensor[:, :, 1]).all() and np.isnan(tensor[3:, :, 0]).all()
    np.testing.assert_array_equal(tensor[:, 1, 2], 201 + np.arange(7))
    store.close()


@pytest.fixture(scope='module')
def completed_store(public_case, tmp_path_factory):
    stored = driver(public_case, tmp_path_factory.mktemp('completed_store'))
    stored.run()
    return stored.trajectory_store.directory


def copy_store(completed_store, tmp_path):
    return shutil.copytree(completed_store, tmp_path / 'recovery')


def test_recovery_rejects_changed_inputs(completed_store, tmp_path):
    location = copy_store(completed_store, tmp_path)
    with (location / 'inputs.veragrid').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='corrompidas'):
        export_recovery(location, tmp_path / 'recovered.veragrid')


@pytest.mark.parametrize('damage', ['tile', 'clock', 'dimensions', 'identity', 'schema'])
def test_recovery_refuses_corruption_without_publishing(completed_store, tmp_path, damage):
    directory = copy_store(completed_store, tmp_path)
    with closing(sqlite3.connect(directory / 'trajectory.sqlite3')) as conn:
        if damage == 'tile': conn.execute("UPDATE blobs SET payload=x'00'")
        if damage == 'clock': conn.execute("UPDATE clocks SET hash=?", ('0' * 64,))
        if damage == 'dimensions': conn.execute('UPDATE rms_groups SET n_cols=n_cols+1')
        if damage == 'identity': conn.execute("UPDATE rms_groups SET id='unknown-group'")
        if damage == 'schema': conn.execute("UPDATE metadata SET value='999' WHERE key='schema_version'")
        conn.commit()
    output = tmp_path / 'recovered.veragrid'
    with pytest.raises((ValueError, RuntimeError)):
        export_recovery(directory, output)
    assert not output.exists()


def test_interrupted_store_discloses_missing_final_state(completed_store, tmp_path):
    directory = copy_store(completed_store, tmp_path)
    with closing(sqlite3.connect(directory / 'trajectory.sqlite3')) as conn:
        conn.execute("DELETE FROM large_metadata WHERE key='result_snapshot'")
        conn.commit()
    info, recovered = reopen_recovery(directory, tmp_path / 'prefix.veragrid')
    assert not info['complete_persisted_metadata']
    assert not np.asarray(recovered['well_initialized']).any()
    assert not np.asarray(recovered['converged']).any()
    assert np.asarray(recovered['recovered']).all()


@pytest.mark.parametrize('damage', ['duplicate_uid', 'missing_variable', 'invalid_index', 'missing_label'])
def test_recovery_rejects_contradictory_channel_maps(completed_store, tmp_path, damage):
    directory = copy_store(completed_store, tmp_path)
    # Use the original streaming metadata API; do not substitute a second format.
    store = RmsTrajectoryStore(directory)
    saved = store.read_json_stream('result_snapshot')
    data = saved['metadata']
    if damage == 'duplicate_uid': data['variables'][1] = data['variables'][0]
    if damage == 'missing_variable': data['variables'].pop()
    if damage == 'invalid_index': data['uid2idx'][next(iter(data['uid2idx']))] = True
    if damage == 'missing_label': data['vars_glob_name2uid'] = {}
    store.close()
    import zlib
    with closing(sqlite3.connect(directory / 'trajectory.sqlite3')) as connection:
        connection.execute('UPDATE large_metadata SET payload=? WHERE key=?',
                           (zlib.compress(json.dumps(saved).encode()), 'result_snapshot'))
        connection.commit()
    output = tmp_path / 'invalid.veragrid'
    with pytest.raises(ValueError, match='RMS recovery'):
        export_recovery(directory, output)
    assert not output.exists()


@pytest.mark.parametrize('failure', ['capture', 'flush'])
def test_store_failure_closes_resources_and_preserves_evidence(public_case, tmp_path, monkeypatch, failure):
    stored = driver(public_case, tmp_path)
    if failure == 'capture':
        def fail(self, results): raise trajectory_store.TrajectoryStorageError('capture failed')
        monkeypatch.setattr(RmsTrajectoryStore, 'capture_results', fail)
    else:
        def fail(self): raise trajectory_store.TrajectoryStorageError('flush failed')
        monkeypatch.setattr(RmsStoredRows, '_flush', fail)
    with pytest.raises(trajectory_store.TrajectoryStorageError):
        stored.run()
    assert stored.trajectory_store._writer is None
    assert stored.trajectory_store.database.is_file()
    assert (stored.trajectory_store.directory / 'inputs.veragrid').is_file()
