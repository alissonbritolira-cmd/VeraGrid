# SPDX-License-Identifier: MPL-2.0
"""Optional trajectory storage controls, using the existing RMS settings page."""
from pathlib import Path
import tempfile
from PySide6 import QtCore, QtWidgets


class _RecoveryWorker(QtCore.QThread):
    stage = QtCore.Signal(str)

    def __init__(self, directory, destination, parent):
        super().__init__(parent)
        self.directory, self.destination = directory, destination
        self.error = None

    def checkpoint(self, *_):
        if self.isInterruptionRequested():
            raise InterruptedError('Recovery cancelled; original files were preserved')

    def remove_owned_file(self, path):
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as error:
            self.error = (self.error or 'Recovery cleanup failed') + f'; remove temporary file {path}: {error}'

    def run(self):
        temporary = None
        owns_destination = False
        try:
            from VeraGridEngine.Simulations.Rms.recovery import export_recovery
            self.stage.emit('Checking committed RMS samples...')
            self.checkpoint()
            if Path(self.destination).exists():
                raise ValueError('Choose a new destination; existing studies are preserved')
            self.stage.emit('Saving recovered RMS history...')
            # Export beside the destination, then publish exclusively. Cancellation
            # must leave neither a misleading partial study nor changed originals.
            with tempfile.NamedTemporaryFile(dir=Path(self.destination).parent,
                                             prefix='.rms-recovery-', suffix='.veragrid', delete=False) as target:
                temporary = Path(target.name)
            # The development recovery API requires a new destination.
            temporary.unlink()
            export_recovery(self.directory, str(temporary), checkpoint=self.checkpoint,
                            progress=self.stage.emit)
            if not temporary.is_file():
                raise ValueError('Recovery export did not complete; preserve the recovery directory')
            self.checkpoint()
            with open(self.destination, 'xb') as target:
                owns_destination = True
                with temporary.open('rb') as source:
                    while data := source.read(1024**2):
                        self.checkpoint()
                        target.write(data)
            self.checkpoint()
        except Exception as error:
            self.error = str(error)
            if owns_destination:
                self.remove_owned_file(self.destination)
        finally:
            if temporary is not None:
                self.remove_owned_file(temporary)


class RecoveryProgress(QtWidgets.QProgressDialog):
    def __init__(self, directory, destination, parent):
        super().__init__('Checking RMS recovery...', 'Cancel', 0, 0, parent)
        self.setWindowTitle('Recover RMS results')
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setMinimumDuration(0)
        self.worker = _RecoveryWorker(directory, destination, self)
        parent.rms_recovery_worker = self.worker
        self.worker.stage.connect(self.setLabelText)
        self.canceled.connect(self.worker.requestInterruption)
        self.worker.finished.connect(self.completed)
        self.worker.start()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.requestInterruption()
            event.ignore()
        else:
            event.accept()

    def completed(self):
        if self.parent().rms_recovery_worker is self.worker:
            self.parent().rms_recovery_worker = None
        self.reset()
        if self.worker.error:
            QtWidgets.QMessageBox.warning(self.parent(), 'RMS recovery', self.worker.error)
        else:
            QtWidgets.QMessageBox.information(self.parent(), 'RMS recovery',
                'Recovered historical samples were saved. Open the new study and load the saved RMS results '
                'to inspect curves. Recovery does not resume integration.\n' + self.worker.destination)
        self.deleteLater()


def install_rms_storage_controls(main):
    main.rms_recovery_worker = None
    box = QtWidgets.QGroupBox(main.tr('RMS trajectory storage (experimental)'), main.ui.frame_113)
    layout = QtWidgets.QVBoxLayout(box)
    main.ui.rms_disk_storage = QtWidgets.QCheckBox(main.tr('Write accepted samples to disk in blocks'), box)
    main.ui.rms_disk_storage.setObjectName('rms_disk_storage')
    main.ui.rms_disk_storage.setToolTip(main.tr(
        'Optional lossless storage with bounded trajectory buffers. Original inputs and committed samples '
        'remain available for recovery. The in-memory mode remains the default.'))
    layout.addWidget(main.ui.rms_disk_storage)
    row = QtWidgets.QHBoxLayout()
    main.ui.rms_recovery_directory = QtWidgets.QLineEdit(box)
    main.ui.rms_recovery_directory.setObjectName('rms_recovery_directory')
    main.ui.rms_recovery_directory.setPlaceholderText(main.tr('Default: local VeraGrid/RmsRecovery directory'))
    row.addWidget(main.ui.rms_recovery_directory, 1)
    browse = QtWidgets.QPushButton(main.tr('Folder...'), box)
    browse.setObjectName('rms_recovery_browse')
    def choose():
        directory = QtWidgets.QFileDialog.getExistingDirectory(main, main.tr('Choose RMS recovery folder'))
        if directory:
            main.ui.rms_recovery_directory.setText(directory)
    browse.clicked.connect(choose)
    row.addWidget(browse)
    recover = QtWidgets.QPushButton(main.tr('Recover results...'), box)
    recover.setObjectName('rms_recover_results')
    def open_recovery():
        if main.any_thread_running():
            QtWidgets.QMessageBox.information(main, 'RMS recovery', 'Finish or cancel the active operation first.')
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(main, main.tr('Choose interrupted RMS execution'),
            main.ui.rms_recovery_directory.text())
        if not directory:
            return
        destination, _ = QtWidgets.QFileDialog.getSaveFileName(main, main.tr('Save recovered RMS history'),
            'Recovered RMS.veragrid', 'VeraGrid (*.veragrid)')
        if destination:
            if not destination.lower().endswith('.veragrid'):
                destination += '.veragrid'
            if Path(destination).exists():
                QtWidgets.QMessageBox.warning(main, 'RMS recovery', 'Choose a new filename to preserve existing studies.')
                return
            dialog = RecoveryProgress(directory, destination, main)
            dialog.show()
    recover.clicked.connect(open_recovery)
    row.addWidget(recover)
    layout.addLayout(row)
    grid = main.ui.gridLayout_32
    items = []
    while grid.count():
        position = grid.getItemPosition(0)
        items.append((position, grid.takeAt(0)))
    for (r, c, rs, cs), item in items:
        grid.addItem(item, r + (1 if r >= 6 else 0), c, rs, cs)
    grid.addWidget(box, 6, 0, 1, 4)
