# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Native Qt widget and option tests, separate from desktop acceptance checks."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6 import QtWidgets

from VeraGrid.Gui.FileDialogues.ons_ems_import import (
    EmsImportDialog, EmsImportDataDialog, apply_prm_pf_settings)
from VeraGridEngine.IO.file_open import FileOpenOptions
from VeraGridEngine.IO.ons_ems.stm_parser import inspect_stm, open_stm


FIXTURES = Path(__file__).parents[1] / 'FileFormats' / 'data' / 'ons_ems'


def test_selected_prm_can_be_removed_without_fallback(qt_app):
    options = FileOpenOptions()
    dialog = EmsImportDialog([str(FIXTURES / 'ac_network_v5.stm'), str(FIXTURES / 'options.prm')], options)
    assert not dialog.apply_pf.isChecked()
    assert dialog.prm.text().endswith('options.prm')
    dialog.prm.clear()
    dialog.frequency.setValue(50)
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert options.ems_prm_path is None
    assert not options.ems_apply_prm_pf
    assert options.ems_frequency_hz == 50


def test_selected_prm_is_retained_without_implicit_application(qt_app):
    options = FileOpenOptions()
    dialog = EmsImportDialog([str(FIXTURES / 'ac_network_v5.stm')], options)
    assert dialog.prm.text() == ''
    dialog.prm.setText(str(FIXTURES / 'options.prm'))
    dialog.accept()
    assert options.ems_prm_path.endswith('options.prm')
    assert not options.ems_apply_prm_pf


def test_explicit_supported_prm_settings_and_native_consultation(qt_app):
    options = FileOpenOptions()
    dialog = EmsImportDialog([str(FIXTURES / 'ac_network_v5.stm'), str(FIXTURES / 'options.prm')], options)
    dialog.apply_pf.setChecked(True)
    dialog.accept()
    assert options.ems_apply_prm_pf
    grid = open_stm(FIXTURES / 'ac_network_v5.stm', FIXTURES / 'options.prm')
    ui = SimpleNamespace(max_iterations_spinBox=QtWidgets.QSpinBox(),
                         use_voltage_guess_checkBox=QtWidgets.QCheckBox())
    ui.max_iterations_spinBox.setRange(1, 1000)
    ui.max_iterations_spinBox.setValue(50)
    ui.use_voltage_guess_checkBox.setChecked(False)
    apply_prm_pf_settings(ui, grid.import_metadata)
    assert ui.max_iterations_spinBox.value() == 30
    assert ui.use_voltage_guess_checkBox.isChecked()
    assert set(grid.import_metadata['applied_gui_options']) == {'PFMAXITER', 'PFFLTSTRT'}
    assert grid.import_metadata['assignments'][2]['status'] == 'preserved, not applied'
    assert grid.import_metadata['assignments'][3]['status'] == 'preserved, not applied'
    consultation = EmsImportDataDialog(grid.import_metadata)
    table = consultation.findChild(QtWidgets.QTableWidget)
    assert table.rowCount() == 4
    assert table.item(3, 1).text() == 'CTGBLOCKQG'
    assert table.item(3, 3).text() == 'preserved, not applied'
    text = [v.toPlainText() for v in consultation.findChildren(QtWidgets.QPlainTextEdit)]
    assert grid.import_metadata['stm']['text'].replace('\r\n', '\n') in text
    assert grid.import_metadata['prm']['text'].replace('\r\n', '\n') in text


def test_unrepresentable_prm_does_not_partially_change_gui(qt_app):
    ui = SimpleNamespace(max_iterations_spinBox=QtWidgets.QSpinBox(),
                         use_voltage_guess_checkBox=QtWidgets.QCheckBox())
    ui.max_iterations_spinBox.setRange(1, 100)
    ui.max_iterations_spinBox.setValue(50)
    ui.use_voltage_guess_checkBox.setChecked(True)
    with pytest.raises(ValueError, match='outside the GUI'):
        apply_prm_pf_settings(ui, {'parameters': {'PFFLTSTRT': True, 'PFMAXITER': 1000}})
    assert ui.max_iterations_spinBox.value() == 50
    assert ui.use_voltage_guess_checkBox.isChecked()


def test_unsupported_case_opens_read_only_inventory_then_rejects_import(qt_app, monkeypatch):
    folder = Path(__file__).resolve().parents[3] / 'examples' / 'ons_ems'
    options = FileOpenOptions()
    seen = []

    def inspect_dialog(dialog):
        assert dialog.findChild(QtWidgets.QLabel, 'ons_emsReadOnlyNotice') is not None
        view = dialog.findChild(QtWidgets.QTableView, 'ons_emsSourceRecords')
        assert view.model().rowCount() == 1958
        selector = dialog.findChild(QtWidgets.QComboBox, 'ons_emsSourceSection')
        selector.setCurrentIndex(selector.findData('dclinks'))
        assert view.model().rowCount() == 12
        seen.append(True)
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(EmsImportDataDialog, 'exec', inspect_dialog)
    dialog = EmsImportDialog([str(folder / 'Soulth SIN Brazil Example.stm'),
                                  str(folder / 'Soulth SIN Brazil Example.prm')], options)
    dialog.apply_pf.setChecked(True)
    dialog.accept()
    assert seen == [True]
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert options.ems_prm_path is None
    assert not options.ems_apply_prm_pf


@pytest.mark.parametrize('parameters', [{'PFMAXITER': True}, {'PFMAXITER': 2.5}, {'PFFLTSTRT': 1}])
def test_invalid_prm_options_leave_gui_unchanged(qt_app, parameters):
    ui = SimpleNamespace(max_iterations_spinBox=QtWidgets.QSpinBox(),
                         use_voltage_guess_checkBox=QtWidgets.QCheckBox())
    ui.max_iterations_spinBox.setValue(50)
    with pytest.raises(ValueError):
        apply_prm_pf_settings(ui, {'parameters': parameters})
    assert ui.max_iterations_spinBox.value() == 50
