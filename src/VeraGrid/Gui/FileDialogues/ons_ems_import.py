# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Explicit STM/PRM selection and read-only source consultation."""
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from VeraGridEngine.IO.ons_ems.stm_parser import inspect_stm, read_prm


class _SourceRecordModel(QtCore.QAbstractTableModel):
    """Render source rows lazily, including large cases, without 100k widgets."""

    def __init__(self, records, parent=None):
        super().__init__(parent)
        self.records = records
        self.columns = max((len(record['fields']) for record in records), default=0) + 1

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.records)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.columns

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        record = self.records[index.row()]
        if index.column() == 0:
            return record['line']
        fields = record['fields']
        return fields[index.column() - 1] if index.column() <= len(fields) else ''

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return 'Source line' if section == 0 else f'Field {section}'
        return super().headerData(section, orientation, role)


class EmsImportDialog(QtWidgets.QDialog):
    """The source folder is never scanned for an unselected PRM."""

    def __init__(self, files, options, parent=None):
        super().__init__(parent)
        self.options = options
        self.setWindowTitle(self.tr('Import ONS EMS STM'))
        self.resize(690, 310)
        files = files if isinstance(files, list) else [files]
        stm = [Path(f) for f in files if Path(f).suffix.lower() == '.stm']
        prm = [f for f in files if Path(f).suffix.lower() == '.prm']
        self.folder = str(stm[0].parent) if stm else ''
        self.stm = stm[0] if len(stm) == 1 else None
        self.selection_valid = len(stm) == 1 and len(prm) <= 1 and len(stm) + len(prm) == len(files)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(self.tr('Import the STM v5 AC network with fixed controls. '
                                       'Unsupported electrical devices or controls stop the import with an explanation. '
                                       'DYN, events and security regions are outside this importer.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.prm = QtWidgets.QLineEdit(prm[0] if len(prm) == 1 else '')
        self.prm.setObjectName('ons_emsPrmPath')
        self.prm.setPlaceholderText(self.tr('Optional PRM; no neighbouring files are chosen automatically'))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.prm)
        browse = QtWidgets.QPushButton(self.tr('Choose PRM...'))
        browse.setObjectName('ons_emsChoosePrm')
        browse.clicked.connect(self.choose_prm)
        row.addWidget(browse)
        clear = QtWidgets.QPushButton(self.tr('Remove'))
        clear.clicked.connect(self.prm.clear)
        row.addWidget(clear)
        layout.addLayout(row)
        self.frequency = QtWidgets.QDoubleSpinBox()
        self.frequency.setObjectName('ons_emsFrequencyHz')
        self.frequency.setRange(1, 400)
        self.frequency.setSuffix(' Hz')
        self.frequency.setValue(options.ems_frequency_hz)
        form = QtWidgets.QFormLayout()
        form.addRow(self.tr('System frequency (STM does not declare it)'), self.frequency)
        layout.addLayout(form)
        self.apply_pf = QtWidgets.QCheckBox(self.tr('Apply supported PRM flow settings: PFMAXITER and PFFLTSTRT'))
        self.apply_pf.setObjectName('ons_emsApplyPrmPf')
        self.apply_pf.setChecked(False)
        layout.addWidget(self.apply_pf)
        note = QtWidgets.QLabel(self.tr('The current GUI settings remain in effect unless you select the option above. '
                                       'All other PRM commands are retained for consultation and are not executed. '
                                       'Use File > ONS EMS import data to inspect the original inputs after importing.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(self.tr('Import'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choose_prm(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.tr('Choose ONS EMS parameters'),
                                                      self.folder, 'ONS EMS PRM (*.prm *.PRM)')
        if path:
            self.prm.setText(path)

    def accept(self):
        if not self.selection_valid:
            QtWidgets.QMessageBox.warning(self, self.tr('ONS EMS STM'),
                                          self.tr('Select exactly one STM and at most one optional PRM.'))
            return
        prm = self.prm.text().strip()
        if prm and (not Path(prm).is_file() or Path(prm).suffix.lower() != '.prm'):
            QtWidgets.QMessageBox.warning(self, self.tr('ONS EMS PRM'),
                                          self.tr('Select an existing PRM or clear the field.'))
            return
        try:
            parameters, _, _ = read_prm(prm or None)
            if self.apply_pf.isChecked():
                if not prm:
                    raise ValueError('Select a PRM before applying its flow settings.')
                validate_pf_parameters(parameters)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, self.tr('ONS EMS PRM'), str(error))
            return
        try:
            metadata = inspect_stm(self.stm, prm or None, self.frequency.value())
        except (OSError, ValueError, NotImplementedError) as error:
            QtWidgets.QMessageBox.warning(self, self.tr('ONS EMS STM'), str(error))
            return
        if metadata['unsupported_features']:
            # Keep the existing project intact. Previewing records is not an
            # electrical import, and never creates a misleading partial grid.
            preview = EmsImportDataDialog(metadata, parent=self)
            try:
                preview.exec()
            finally:
                preview.deleteLater()
            self.reject()
            return
        self.options.ems_prm_path = prm or None
        self.options.ems_frequency_hz = self.frequency.value()
        self.options.ems_apply_prm_pf = self.apply_pf.isChecked()
        super().accept()


def validate_pf_parameters(parameters):
    """No tolerance-unit conversion or delayed-control approximation is made."""
    if 'PFMAXITER' in parameters:
        value = parameters['PFMAXITER']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 1 or int(value) != value:
            raise ValueError('PFMAXITER must be a positive integer.')
    if 'PFFLTSTRT' in parameters and not isinstance(parameters['PFFLTSTRT'], bool):
        raise ValueError('PFFLTSTRT must be T or F.')


def apply_prm_pf_settings(ui, metadata):
    """Called only following the user's explicit import-checkbox selection."""
    parameters = metadata.get('parameters', {})
    validate_pf_parameters(parameters)
    applied = metadata.setdefault('applied_gui_options', {})
    if 'PFMAXITER' in parameters:
        value = int(parameters['PFMAXITER'])
        widget = ui.max_iterations_spinBox
        if not widget.minimum() <= value <= widget.maximum():
            raise ValueError('PFMAXITER is outside the GUI iteration range; it was not rounded or applied.')
    for command, value in parameters.items():
        if command == 'PFMAXITER':
            ui.max_iterations_spinBox.setValue(int(value))
            applied[command] = {'value': int(value), 'origin': 'Explicit PRM selection in import dialog'}
        elif command == 'PFFLTSTRT':
            ui.use_voltage_guess_checkBox.setChecked(not value)
            applied[command] = {'value': value, 'origin': 'Explicit PRM selection in import dialog'}
    for assignment in metadata.get('assignments', []):
        if assignment.get('command') in applied:
            assignment['status'] = 'applied to GUI by explicit user selection'


class EmsImportDataDialog(QtWidgets.QDialog):
    """Embedded sources stay inspectable after saving/reopening the project."""

    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('ONS EMS import data'))
        self.resize(860, 570)
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName('ons_emsImportDataTabs')
        unsupported = metadata.get('unsupported_features', [])
        if unsupported:
            notice = QtWidgets.QLabel(self.tr('Source inspection only. This case needs additional electrical '
                                               'adapters. The current circuit is unchanged; no partial network '
                                               'can be simulated from this preview.'))
            notice.setWordWrap(True)
            notice.setObjectName('ons_emsReadOnlyNotice')
            layout.addWidget(notice)
            self.setWindowTitle(self.tr('ONS EMS source inspection — not imported'))
        sections = metadata.get('section_records', {})
        if sections:
            page = QtWidgets.QWidget()
            section_layout = QtWidgets.QVBoxLayout(page)
            selector = QtWidgets.QComboBox()
            selector.setObjectName('ons_emsSourceSection')
            for name, records in sections.items():
                selector.addItem(f'{name} ({len(records)})', name)
            record_view = QtWidgets.QTableView()
            record_view.setObjectName('ons_emsSourceRecords')
            record_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            record_view.horizontalHeader().setDefaultSectionSize(115)

            def select_section():
                model = _SourceRecordModel(sections[selector.currentData()], parent=record_view)
                old_model = record_view.model()
                record_view.setModel(model)
                if old_model:
                    old_model.deleteLater()

            selector.currentIndexChanged.connect(select_section)
            select_section()
            section_layout.addWidget(selector)
            section_layout.addWidget(record_view)
            tabs.addTab(page, self.tr('All source records'))
        table = QtWidgets.QTableWidget(len(metadata.get('assignments', [])), 4)
        table.setHorizontalHeaderLabels([self.tr('Line'), self.tr('Command'), self.tr('Value'), self.tr('Treatment')])
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for index, assignment in enumerate(metadata.get('assignments', [])):
            for column, value in enumerate((assignment['line'], assignment.get('command', ''),
                                             assignment.get('value', assignment.get('text', '')),
                                             assignment['status'])):
                table.setItem(index, column, QtWidgets.QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(table, self.tr('PRM commands'))
        summary = QtWidgets.QPlainTextEdit()
        summary.setReadOnly(True)
        lines = [f"{metadata.get('format', '')}; frequency: {metadata.get('frequency_hz', '')} Hz"]
        for name in ('stm', 'prm'):
            source = metadata.get(name)
            if source:
                lines.append(f"{name.upper()}: {source['source']}\nSHA256: {source['sha256']}")
                view = QtWidgets.QPlainTextEdit(source['text'])
                view.setReadOnly(True)
                tabs.addTab(view, self.tr('Original ') + name.upper())
        lines.extend(metadata.get('diagnostics', []))
        lines.extend(unsupported)
        summary.setPlainText('\n\n'.join(lines))
        tabs.addTab(summary, self.tr('Sources and diagnostics'))
        if unsupported:
            tabs.setCurrentWidget(summary)
        layout.addWidget(tabs)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
