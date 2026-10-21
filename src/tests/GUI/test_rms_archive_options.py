"""Qt widget/save-option coverage, not physical desktop navigation."""
from types import SimpleNamespace
import zipfile
import numpy as np
from PySide6 import QtWidgets

from VeraGrid.Gui.Main.SubClasses.Settings.configuration import ConfigurationMain
from VeraGrid.Gui.Main.SubClasses.io import IoMain
from VeraGridEngine.enumerations import SimulationTypes


def test_archive_checkbox_is_optional_and_tracks_result_saving(qt_app):
    frame = QtWidgets.QFrame()
    save = QtWidgets.QCheckBox(frame)
    ui = SimpleNamespace(frame_77=frame, saveResultsCheckBox=save,
                         label_155=QtWidgets.QLabel(frame), gridLayout_11=QtWidgets.QGridLayout(frame))
    window = SimpleNamespace(ui=ui, tr=lambda text: text)
    ConfigurationMain.create_rms_archive_controls(window)
    checkbox = window.rms_chunked_results_checkBox
    assert not checkbox.isChecked() and not checkbox.isEnabled()
    save.setChecked(True)
    assert checkbox.isEnabled()
    checkbox.setChecked(True)
    rms = SimpleNamespace(tpe=SimulationTypes.RmsDynamic_run, results=SimpleNamespace(archive_chunk_rows=0))
    pf = SimpleNamespace(tpe=SimulationTypes.PowerFlow_run, results=SimpleNamespace())
    window.session = SimpleNamespace(get_save_data=lambda: [rms, pf])
    window.get_gui_config_data = lambda: {}
    options = IoMain.get_file_save_options(window)
    assert options.sessions_data == [rms, pf]
    assert rms.results.archive_chunk_rows == 131072
    assert not hasattr(pf.results, "archive_chunk_rows")
    checkbox.setChecked(False)
    IoMain.get_file_save_options(window)
    assert rms.results.archive_chunk_rows == 0
    save.setChecked(False)
    assert not checkbox.isEnabled()
    assert IoMain.get_file_save_options(window).sessions_data == []
    frame.deleteLater()
    qt_app.processEvents()


def test_native_curve_reads_one_indexed_channel(qt_app, tmp_path, monkeypatch):
    from GUI.test_dynamic_native_plots import build_dynamic_plot_results
    from VeraGrid.Gui.DynamicModelEditor.Plots.dynamic_plots_handler import DynamicsResultsHandler
    from VeraGridEngine.IO.veragrid.rms_archive import ZipRmsValues, write_rms_values
    results = build_dynamic_plot_results()
    results.values[2:, :, 0] = np.nan
    path = tmp_path / "curve.zip"
    with zipfile.ZipFile(path, "w") as archive:
        write_rms_values(archive, "values.rms.json", results.values, 2)
    results.values = ZipRmsValues(path, "values.rms.json")
    def forbid_full_tensor(*_args, **_kwargs):
        raise AssertionError("A native curve must not materialize all RMS channels")
    monkeypatch.setattr(ZipRmsValues, "__array__", forbid_full_tensor)
    handler = DynamicsResultsHandler(results=results)
    assert results.values.members_read == 0
    handler.plot_series(series=handler.series_by_var_uid[1][0])
    qt_app.processEvents()
    plot = handler._open_plot_dialogues[-1]
    assert plot.isVisible() and plot.chart._has_points
    assert plot.chart._x_min == 0.0 and plot.chart._x_max == 0.5
    assert results.values.members_read == 2  # Selected omega only, not efd.
    handler.close_plot_dialogs()
    qt_app.processEvents()
    assert plot.chart._disposed
