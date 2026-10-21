# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Qt glue for the already-developed PRM/default/permission workflow."""
from pathlib import Path
from types import SimpleNamespace
from PySide6 import QtWidgets
from VeraGrid.Gui.FileDialogues import ons_ems_import as dialogs
from VeraGridEngine.IO.file_open import FileOpenOptions
from VeraGridEngine.IO.ons_ems.stm_parser import open_stm

FIXTURES = Path(__file__).parents[1] / 'FileFormats' / 'data' / 'ons_ems'


def pf_ui():
    ui=SimpleNamespace(max_iterations_spinBox=QtWidgets.QSpinBox(),
        tolerance_spinBox=QtWidgets.QSpinBox(),apply_pf_prm_button=QtWidgets.QPushButton())
    ui.max_iterations_spinBox.setRange(1,1000)
    ui.max_iterations_spinBox.setValue(50)
    ui.tolerance_spinBox.setRange(1,12)
    ui.tolerance_spinBox.setValue(6)
    for name in ('control_tap_modules_checkBox','control_tap_phase_checkBox',
            'control_discrete_shunts_checkBox','control_q_checkBox',
            'control_remote_voltage_checkBox','use_voltage_guess_checkBox'):
        setattr(ui,name,QtWidgets.QCheckBox())
    return ui


def test_selected_prm_can_be_removed_without_fallback(qt_app):
    options=FileOpenOptions()
    dialog=dialogs.EmsImportDialog([str(FIXTURES/'ac_network_v5.stm'),str(FIXTURES/'options.prm')],options)
    assert not dialog.apply_pf.isChecked()
    dialog.prm.clear();dialog.frequency.setValue(50);dialog.accept()
    assert dialog.result()==QtWidgets.QDialog.DialogCode.Accepted
    assert options.ems_prm_path is None and not options.ems_apply_prm_pf
    assert options.ems_frequency_hz==50


def test_prm_is_explicit_and_full_case_is_not_replaced_by_preview(qt_app):
    folder=Path(__file__).resolve().parents[3]/'examples'/'ons_ems'
    options=FileOpenOptions()
    dialog=dialogs.EmsImportDialog([str(folder/'Soulth SIN Brazil Example.stm'),
                                   str(folder/'Soulth SIN Brazil Example.prm')],options)
    dialog.accept()
    assert dialog.result()==QtWidgets.QDialog.DialogCode.Accepted
    assert options.ems_prm_path.endswith('Soulth SIN Brazil Example.prm')
    assert not options.ems_apply_prm_pf


def test_imported_defaults_block_automatic_actuators_and_restore_saved_choices(qt_app):
    grid=open_stm(FIXTURES/'ac_network_v5.stm',FIXTURES/'options.prm')
    ui=pf_ui()
    for _,widget,_,_,_ in dialogs.PF_CONTROL_SWITCHES:getattr(ui,widget).setChecked(True)
    dialogs.apply_imported_pf_settings(ui,grid.import_metadata,fresh_import=True)
    assert all(not getattr(ui,widget).isChecked() for _,widget,_,_,_ in dialogs.PF_CONTROL_SWITCHES)
    assert ui.max_iterations_spinBox.value()==30
    assert ui.tolerance_spinBox.value()==3
    assert ui.use_voltage_guess_checkBox.isChecked()
    assert ui.apply_pf_prm_button.isEnabled()
    saved={'power_flow':{'transformer_taps_module_control':True,
        'transformer_taps_phase_control':False,'discrete_shunts_control':True}}
    dialogs.apply_imported_pf_settings(ui,grid.import_metadata,gui_config=saved,fresh_import=False)
    assert ui.control_tap_modules_checkBox.isChecked()
    assert not ui.control_tap_phase_checkBox.isChecked()
    assert ui.control_discrete_shunts_checkBox.isChecked()


def test_explicit_prm_preview_applies_qualified_options_and_keeps_other_commands(qt_app,monkeypatch):
    grid=open_stm(FIXTURES/'ac_network_v5.stm',FIXTURES/'options.prm')
    grid.import_metadata['parameters'].update(PFBLKTAP=False,PFBLKSHFT=False,PFBLKSHT=False,PFGENVARIT=0)
    parent=SimpleNamespace(ui=pf_ui(),circuit=grid)
    seen=[]
    monkeypatch.setattr(dialogs,'confirm_prm_changes',lambda *args,**kwargs:seen.append(args) or True)
    assert dialogs.apply_prm_pf_settings_dialog(parent)
    assert seen
    assert parent.ui.max_iterations_spinBox.value()==30
    assert parent.ui.tolerance_spinBox.value()==3
    assert parent.ui.use_voltage_guess_checkBox.isChecked()
    assert parent.ui.control_q_checkBox.isChecked()
    assert all(getattr(parent.ui,widget).isChecked() for _,widget,_,_,_ in dialogs.PF_CONTROL_SWITCHES)
    assert set(grid.import_metadata['pf_control_origins'])=={'control_taps_modules','control_taps_phase','control_discrete_shunts'}
    consultation=dialogs.EmsImportDataDialog(grid.import_metadata)
    table=consultation.findChild(QtWidgets.QTableWidget)
    assert table.rowCount()==4
    assert any(table.item(row,1).text()=='CTGBLOCKQG' for row in range(table.rowCount()))
    text=[view.toPlainText() for view in consultation.findChildren(QtWidgets.QPlainTextEdit)]
    assert grid.import_metadata['stm']['text'].replace('\r\n','\n') in text
    assert grid.import_metadata['prm']['text'].replace('\r\n','\n') in text


def test_cancelled_prm_preview_does_not_change_gui(qt_app,monkeypatch):
    grid=open_stm(FIXTURES/'ac_network_v5.stm',FIXTURES/'options.prm')
    parent=SimpleNamespace(ui=pf_ui(),circuit=grid)
    monkeypatch.setattr(dialogs,'confirm_prm_changes',lambda *args,**kwargs:False)
    assert not dialogs.apply_prm_pf_settings_dialog(parent)
    assert parent.ui.max_iterations_spinBox.value()==50
    assert parent.ui.tolerance_spinBox.value()==6
    assert not parent.ui.use_voltage_guess_checkBox.isChecked()
