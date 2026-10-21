"""Internal native Qt settings/worker checks, not desktop acceptance tests."""
from copy import deepcopy
import hashlib
import json
import sqlite3
from contextlib import closing
from threading import Event

import pytest
from PySide6 import QtCore, QtWidgets

from Base.test_dynamic_results_save_load import _build_rms_driver
from VeraGrid.Gui.Main.SubClasses.Settings.configuration import ConfigurationMain
from VeraGrid.Gui.Main.SubClasses.Settings import rms_storage
from VeraGridEngine.Simulations.Rms.rms_driver import RmsSimulationDriver
from VeraGridEngine.Simulations.Rms import recovery


def test_native_rms_settings_produce_disk_options(qt_app, tmp_path):
    main = ConfigurationMain()
    assert not main.ui.rms_disk_storage.isChecked()
    main.ui.rms_disk_storage.setChecked(True)
    main.ui.rms_recovery_directory.setText(str(tmp_path))
    options = main.get_selected_rms_simulation_options()
    assert options.trajectory_storage == 'lossless_disk'
    assert options.trajectory_directory == str(tmp_path)
    assert main.ui.rms_disk_storage.parentWidget().isEnabled()


@pytest.mark.parametrize('custom_directory', [False, True])
def test_native_storage_config_get_apply_save_load_roundtrip(qt_app, tmp_path, monkeypatch, custom_directory):
    config_path = tmp_path / 'gui_config.json'
    monkeypatch.setattr(ConfigurationMain, 'config_file_path', staticmethod(lambda: str(config_path)))
    expected = str(tmp_path / 'RMS results Ω') if custom_directory else ''
    main = ConfigurationMain()
    main.ui.rms_disk_storage.setChecked(custom_directory)
    main.ui.rms_recovery_directory.setText(expected)
    data = main.get_gui_config_data()
    assert data['dyn']['rms_recovery_directory'] == expected
    assert data['dyn']['rms_disk_storage'] is custom_directory
    main.ui.rms_recovery_directory.setText('temporary value')
    main.ui.rms_disk_storage.setChecked(not custom_directory)
    main.apply_gui_config(data)
    assert main.ui.rms_recovery_directory.text() == expected
    assert main.ui.rms_disk_storage.isChecked() is custom_directory
    main.save_gui_config()
    with config_path.open() as saved:
        assert json.load(saved)['dyn']['rms_recovery_directory'] == expected
    reopened = ConfigurationMain()
    reopened.load_gui_config()
    assert reopened.ui.rms_recovery_directory.text() == expected
    assert reopened.ui.rms_disk_storage.isChecked() is custom_directory
    assert reopened.get_selected_rms_simulation_options().trajectory_directory == expected


def test_storage_config_legacy_and_invalid_path_keep_native_defaults(qt_app):
    main = ConfigurationMain()
    main.apply_gui_config({'dyn': {}})
    assert main.ui.rms_recovery_directory.text() == ''
    assert not main.ui.rms_disk_storage.isChecked()
    main.ui.rms_recovery_directory.setText('keep this path')
    for invalid in (None, 123, [], {}):
        main.apply_gui_config({'dyn': {'rms_recovery_directory': invalid}})
        assert main.ui.rms_recovery_directory.text() == 'keep this path'


def test_recovery_worker_participates_in_main_shutdown(qt_app, monkeypatch):
    main = ConfigurationMain()
    started = Event()
    def wait_for_stop(worker):
        started.set()
        while not worker.isInterruptionRequested():
            worker.msleep(2)
    monkeypatch.setattr(rms_storage._RecoveryWorker, 'run', wait_for_stop)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'information', lambda *args: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *args: None)
    dialog = rms_storage.RecoveryProgress('unused', 'unused', main)
    assert started.wait(2)
    worker = dialog.worker
    assert worker in main.get_all_threads()
    assert main.any_thread_running()
    assert main.stop_all_threads()
    assert not worker.isRunning()
    qt_app.processEvents()
    assert main.rms_recovery_worker is None


@pytest.fixture(scope='module')
def recovery_directory(tmp_path_factory):
    directory = tmp_path_factory.mktemp('gui_recovery')
    base = _build_rms_driver()
    grid, options, flow = deepcopy((base.grid, base.options, base.pf_results))
    options.trajectory_storage = 'lossless_disk'
    options.trajectory_directory = str(directory)
    stored = RmsSimulationDriver(grid, options, flow)
    stored.run()
    return stored.trajectory_store.directory


def _source_hashes(directory):
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('inputs.veragrid', 'trajectory.sqlite3')}


def test_recovery_export_cancel_removes_only_own_output(qt_app, tmp_path, recovery_directory, monkeypatch):
    before = _source_hashes(recovery_directory)
    calls = []
    writing = []
    real_export = recovery.export_recovery
    def export(directory, destination, *, checkpoint, progress):
        def stage(message):
            if 'transacional' in message:
                writing.append(True)
            progress(message)
        def stop_during_stream():
            checkpoint()
            if not writing:
                return
            calls.append(True)
            if len(calls) == 2:
                raise InterruptedError('cancelled between native result tiles')
        return real_export(directory, destination, checkpoint=stop_during_stream, progress=stage)
    monkeypatch.setattr(recovery, 'export_recovery', export)
    parent = QtWidgets.QWidget()
    output = tmp_path / 'recovered.veragrid'
    worker = rms_storage._RecoveryWorker(str(recovery_directory), str(output), parent)
    worker.run()
    assert len(calls) == 2
    assert 'cancelled between native result tiles' in worker.error
    assert not output.exists()
    assert not list(tmp_path.glob('.rms-recovery-*'))
    assert _source_hashes(recovery_directory) == before


def test_recovery_export_does_not_overwrite_existing_study(qt_app, tmp_path, recovery_directory):
    output = tmp_path / 'existing.veragrid'
    output.write_bytes(b'existing user study')
    before = _source_hashes(recovery_directory)
    parent = QtWidgets.QWidget()
    worker = rms_storage._RecoveryWorker(str(recovery_directory), str(output), parent)
    worker.run()
    assert 'existing studies are preserved' in worker.error
    assert output.read_bytes() == b'existing user study'
    assert _source_hashes(recovery_directory) == before
