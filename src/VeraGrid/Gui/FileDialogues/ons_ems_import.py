# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# SPDX-License-Identifier: MPL-2.0
"""Explicit STM/PRM selection and read-only source consultation."""
from pathlib import Path
import math
import json

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
        note = QtWidgets.QLabel(self.tr('Import the complete STM v5 network using the developed EMS power-flow adapters. '
                                       'Equipment and control records are retained, with execution limits explained. '
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
        self.apply_pf = QtWidgets.QCheckBox(self.tr('Review and apply supported PRM control permissions after importing'))
        self.apply_pf.setObjectName('ons_emsApplyPrmPf')
        self.apply_pf.setChecked(False)
        layout.addWidget(self.apply_pf)
        note = QtWidgets.QLabel(self.tr('PF numeric defaults come from the selected PRM. Automatic taps, phase shifters '
                                       'and discrete banks start blocked; the GUI choices take precedence. '
                                       'The option above opens the existing PRM preview before applying permissions. '
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
        from VeraGridEngine.IO.ons_ems.policy import parameter_inventory
        records = parameter_inventory(metadata.get('parameters', {}), metadata.get('parameter_assignments'))
        effective_records = metadata.get('effective_pf', {}).get('commands', [])
        for record in records:
            match = next((item for item in effective_records if item['command'] == record['command']
                          and item.get('line') == record.get('line')), None)
            if match and match.get('overridden_by') and record['effective_assignment']:
                record['status'] = 'overridden by explicit option'
                record['detail'] = f"{match['overridden_by']} = {match['effective_value']}; source request retained"
        table = QtWidgets.QTableWidget(len(records), 5)
        table.setHorizontalHeaderLabels([self.tr('Line'), self.tr('Command'), self.tr('Requested'),
                                         self.tr('Status'), self.tr('Treatment')])
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for index, assignment in enumerate(records):
            for column, value in enumerate((assignment.get('line', ''), assignment['command'],
                                             assignment['requested'], assignment['status'], assignment['detail'])):
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
        if metadata.get('effective_pf'):
            effective = QtWidgets.QPlainTextEdit(json.dumps(metadata['effective_pf'], ensure_ascii=False,
                                                           indent=2, default=str))
            effective.setReadOnly(True)
            tabs.addTab(effective, self.tr('Effective PF options'))
        if unsupported:
            tabs.setCurrentWidget(summary)
        layout.addWidget(tabs)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


PF_CONTROL_SWITCHES = (
    ('control_taps_modules', 'control_tap_modules_checkBox', 'transformer_taps_module_control', 'BLKTAP', 'Módulo dos taps'),
    ('control_taps_phase', 'control_tap_phase_checkBox', 'transformer_taps_phase_control', 'BLKSHFT', 'Defasadores'),
    ('control_discrete_shunts', 'control_discrete_shunts_checkBox', 'discrete_shunts_control', 'BLKSHT', 'Bancos de shunts discretos'),
)


def apply_imported_pf_settings(ui, metadata, *, gui_config=None, fresh_import=True):
    """Restore saved choices; a raw import never grants automatic permissions."""
    parameters = metadata.get('parameters', {})
    effective=metadata.get('effective_pf',{}).get('options',{})
    saved = (gui_config or {}).get('power_flow', {})
    origins = metadata.setdefault('pf_control_origins', {})
    for field, widget, config_key, _command, _label in PF_CONTROL_SWITCHES:
        if not fresh_import and config_key in saved:
            value, origin = bool(saved[config_key]), 'Escolha conservada do estudo salvo'
        elif not fresh_import and field in effective:
            value, origin = bool(effective[field]), 'Última opção explícita incorporada ao estudo'
        elif not fresh_import:
            value, origin = False, 'Migração: permissão geral ausente no estudo legado; iniciada bloqueada'
        else:
            value, origin = False, 'Padrão bloqueado de novo caso importado'
        control = getattr(ui, widget)
        blocker = QtCore.QSignalBlocker(control)
        control.setChecked(value)
        del blocker
        # Opening a native study is not a new control choice. Preserve its
        # provenance as well as its value, otherwise merely reopening changes
        # the RMS/DSA input fingerprint and makes current curves historical.
        if fresh_import or field not in origins:
            origins[field] = origin
    ui.apply_pf_prm_button.setEnabled(bool(parameters))
    # The GUI configuration has already restored precision, iterations and
    # controls. Runtime metadata must not overwrite that deliberate snapshot.
    if not fresh_import or metadata.get('format') != 'ONS EMS STM v5':
        return
    tolerance = effective.get('tolerance',parameters.get('PFMVATOL'))
    if (isinstance(tolerance, (int, float)) and tolerance > 0 and
            math.isclose(-math.log10(tolerance), round(-math.log10(tolerance)), abs_tol=1e-10)):
        ui.tolerance_spinBox.setValue(round(-math.log10(tolerance)))
    maximum = effective.get('max_iter',parameters.get('PFMAXITER'))
    if isinstance(maximum, (int, float)) and maximum > 0:
        ui.max_iterations_spinBox.setValue(int(maximum))
    if 'PFGENVARIT' in parameters and maximum:
        ui.control_q_checkBox.setChecked(parameters['PFGENVARIT'] < maximum)
    if 'control_Q' in effective:ui.control_q_checkBox.setChecked(effective['control_Q'])
    ui.control_remote_voltage_checkBox.setChecked(effective.get('control_remote_voltage',bool(metadata.get('remote_voltage_controls'))))
    ui.use_voltage_guess_checkBox.setChecked(effective.get('use_stored_guess',not parameters.get('PFFLTSTRT', False)))


def confirm_prm_changes(parent, title, rows, notes=()):
    """Preview UI changes; reading a PRM alone never applies them."""
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(800, 470)
    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel('Confira as alterações abaixo. Aplicar muda somente estas opções; '
        'os bloqueios individuais e os comandos ainda sem suporte permanecem. Nenhum cálculo será iniciado.')
    label.setWordWrap(True)
    layout.addWidget(label)
    table = QtWidgets.QTableWidget(len(rows), 4)
    table.setHorizontalHeaderLabels(['Opção', 'Valor atual', 'Valor do PRM', 'Origem'])
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            text = ('Permitido' if value else 'Bloqueado') if isinstance(value, bool) else str(value)
            table.setItem(i, j, QtWidgets.QTableWidgetItem(text))
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table, 1)
    if notes:
        note = QtWidgets.QPlainTextEdit('\n'.join(notes))
        note.setReadOnly(True)
        note.setMaximumHeight(115)
        layout.addWidget(note)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Apply |
                                         QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).setText('Aplicar opções')
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted


def apply_prm_pf_settings_dialog(parent):
    parameters = parent.circuit.import_metadata.get('parameters', {})
    if not parameters:
        QtWidgets.QMessageBox.information(parent, 'Padrões do PRM', 'Este estudo não possui PRM incorporado.')
        return False
    changes, rows, notes = [], [], []
    for field, widget, _config_key, suffix, label in PF_CONTROL_SWITCHES:
        command = 'PF'+suffix
        if command in parameters:
            if not isinstance(parameters[command],bool):
                notes.append(command+': valor de bloqueio inválido; não será aplicado.')
                continue
            value = not bool(parameters[command])
            control = getattr(parent.ui, widget)
            changes.append((control, value, field))
            rows.append((label, control.isChecked(), value, command+' (permissão inversa ao bloqueio)'))
    value = parameters.get('PFMVATOL')
    if isinstance(value, (int, float)) and value > 0 and math.isfinite(value):
        exponent = -math.log10(value)
        if (math.isclose(exponent, round(exponent), abs_tol=1e-10)
                and parent.ui.tolerance_spinBox.minimum() <= round(exponent) <= parent.ui.tolerance_spinBox.maximum()):
            changes.append((parent.ui.tolerance_spinBox, int(round(exponent)), None))
            rows.append(('Tolerância do fluxo [pu]', 10.**-parent.ui.tolerance_spinBox.value(), value, 'PFMVATOL'))
        else:
            notes.append('PFMVATOL não é uma potência de 10 representável neste campo; não foi arredondado nem aplicado.')
    maximum = parameters.get('PFMAXITER')
    if (not isinstance(maximum,bool) and isinstance(maximum, (int, float)) and math.isfinite(maximum)
            and parent.ui.max_iterations_spinBox.minimum() <= maximum <= parent.ui.max_iterations_spinBox.maximum()
            and float(maximum).is_integer()):
        changes.append((parent.ui.max_iterations_spinBox, int(maximum), None))
        rows.append(('Máximo de iterações', parent.ui.max_iterations_spinBox.value(), int(maximum), 'PFMAXITER'))
        delay = parameters.get('PFGENVARIT', 0)
        if delay == 0 or delay >= maximum:
            value = delay < maximum
            changes.append((parent.ui.control_q_checkBox, value, None))
            rows.append(('Limites de Q', parent.ui.control_q_checkBox.isChecked(), value, 'PFGENVARIT'))
        else:
            notes.append('PFGENVARIT intermediário sem execução qualificada; a escolha de limites de Q não será alterada.')
    if 'PFFLTSTRT' in parameters:
        value = not bool(parameters['PFFLTSTRT'])
        changes.append((parent.ui.use_voltage_guess_checkBox, value, None))
        rows.append(('Usar tensão cadastrada', parent.ui.use_voltage_guess_checkBox.isChecked(), value, 'PFFLTSTRT'))
    notes.append('Ajuste de bancos discretos não bloqueia SVC contínuo. Eventos/proteções RMS não são desligados por estas permissões do fluxo.')
    notes.append('Busca, contingências e parâmetros próprios de uma região não são alterados por este botão do F5.')
    if not confirm_prm_changes(parent, 'Prévia do PRM — fluxo de potência', rows, notes):
        return False
    origins = parent.circuit.import_metadata.setdefault('pf_control_origins', {})
    for widget, value, field in changes:
        if isinstance(widget, QtWidgets.QCheckBox):
            widget.setChecked(value)
        else:
            widget.setValue(value)
        if field:
            origins[field] = 'PRM aplicado explicitamente pela prévia do fluxo'
    if hasattr(parent, "study_model_changed"): parent.study_model_changed()
    return True
