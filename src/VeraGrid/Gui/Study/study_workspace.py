# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Diagram-independent navigation using the native circuit and property editors."""
from __future__ import annotations

from collections import deque
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from VeraGrid.Gui.object_model import ObjectsModel
from VeraGridEngine.Devices.Parents.physical_device import PhysicalDevice
from VeraGridEngine.Devices.Substation.bus import Bus
from VeraGridEngine.Devices.multi_circuit import MultiCircuit


def connected_buses(device: PhysicalDevice) -> list[Bus]:
    """Return distinct native endpoint objects, including three-winding devices."""
    buses: list[Bus] = []
    for attribute in ("bus", "bus_from", "bus_to", "bus1", "bus2", "bus3"):
        bus = getattr(device, attribute, None)
        if isinstance(bus, Bus) and all(bus is not other for other in buses):
            buses.append(bus)
    return buses


class StudyObjectsModel(ObjectsModel):
    """Also guard commits if a worker starts or the circuit changes mid-edit."""

    def __init__(self, can_edit: Callable[[], bool], **kwargs):
        self.can_edit = can_edit
        super().__init__(**kwargs)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if not self.can_edit():
            flags &= ~(QtCore.Qt.ItemFlag.ItemIsEditable | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        return flags

    def setData(self, index: QtCore.QModelIndex, value, role=None) -> bool:
        if not self.can_edit():
            return False
        return super().setData(index, value, role)


class EquipmentFilter(QtCore.QSortFilterProxyModel):
    """Match every search word across type, name, code, ID and endpoints."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.words: list[str] = []
        self.device_type = None
        self.setSortCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)

    def filterAcceptsRow(self, row: int, parent: QtCore.QModelIndex) -> bool:
        item = self.sourceModel().item(row, 0)
        device = item.data(QtCore.Qt.ItemDataRole.UserRole)
        text = item.data(QtCore.Qt.ItemDataRole.UserRole + 1)
        return (self.device_type is None or device.device_type == self.device_type) and all(
            word in text for word in self.words)

    def apply(self, text: str, device_type=None) -> None:
        self.words = text.casefold().split()
        self.device_type = device_type
        self.invalidate()


class StudyWorkspace(QtWidgets.QWidget):
    """Browse and edit snapshot data without a diagram or a duplicate data model.

    Results remain in the existing Results tab. This first workspace deliberately
    does not implement its own solver, result cache or profile editor.
    """

    def __init__(self, circuit: Callable[[], MultiCircuit],
                 busy: Callable[[], bool], parent=None,
                 power_flow_action: QtGui.QAction | None = None,
                 show_results: Callable[[], None] | None = None,
                 error_message: Callable[[str], None] | None = None):
        super().__init__(parent)
        self.setObjectName("studyWorkspace")
        self.get_circuit = circuit
        self.is_busy = busy
        self.error_message = error_message
        self.circuit: MultiCircuit | None = None
        self.devices: list[PhysicalDevice] = []
        self.selected: PhysicalDevice | None = None
        self.history = deque(maxlen=40)
        self.properties_model: StudyObjectsModel | None = None
        self._refreshing = False

        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        layout.addLayout(toolbar)
        self.back_button = QtWidgets.QPushButton(self)
        self.back_button.setObjectName("studyBackButton")
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(self.go_back)
        toolbar.addWidget(self.back_button)
        self.refresh_button = QtWidgets.QPushButton(self)
        self.refresh_button.setObjectName("studyRefreshButton")
        self.refresh_button.clicked.connect(self.refresh)
        toolbar.addWidget(self.refresh_button)
        toolbar.addStretch(1)
        if power_flow_action is not None:
            button = QtWidgets.QToolButton(self)
            button.setDefaultAction(power_flow_action)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            toolbar.addWidget(button)
        self.results_button = QtWidgets.QPushButton(self)
        self.results_button.setObjectName("studyResultsButton")
        self.results_button.setEnabled(show_results is not None)
        if show_results is not None:
            self.results_button.clicked.connect(show_results)
        toolbar.addWidget(self.results_button)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        layout.addWidget(split, 1)
        search_panel = QtWidgets.QWidget(split)
        left = QtWidgets.QVBoxLayout(search_panel)
        self.search = QtWidgets.QLineEdit(search_panel)
        self.search.setObjectName("studyEquipmentSearch")
        self.search.setClearButtonEnabled(True)
        self.category = QtWidgets.QComboBox(search_panel)
        self.category.setObjectName("studyEquipmentType")
        left.addWidget(self.search)
        left.addWidget(self.category)
        self.matches_label = QtWidgets.QLabel(search_panel)
        left.addWidget(self.matches_label)
        self.source = QtGui.QStandardItemModel(self)
        self.proxy = EquipmentFilter(self)
        self.proxy.setSourceModel(self.source)
        self.hits = QtWidgets.QTableView(search_panel)
        self.hits.setObjectName("studyEquipmentMatches")
        self.hits.setModel(self.proxy)
        self.hits.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.hits.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.hits.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hits.setSortingEnabled(True)
        self.hits.horizontalHeader().setStretchLastSection(True)
        self.hits.setColumnWidth(0, 105)
        self.hits.setColumnWidth(1, 160)
        left.addWidget(self.hits, 1)
        self.hits.selectionModel().currentChanged.connect(self._hit_selected)
        self.search.textChanged.connect(self._filter)
        self.category.currentIndexChanged.connect(self._filter)

        details_panel = QtWidgets.QWidget(split)
        right = QtWidgets.QVBoxLayout(details_panel)
        self.heading = QtWidgets.QLabel(details_panel)
        self.heading.setWordWrap(True)
        right.addWidget(self.heading)
        bus_row = QtWidgets.QHBoxLayout()
        self.bus_selector = QtWidgets.QComboBox(details_panel)
        self.bus_selector.setObjectName("studyConnectedBus")
        bus_row.addWidget(self.bus_selector, 1)
        self.bus_button = QtWidgets.QPushButton(details_panel)
        self.bus_button.setObjectName("studyGoToBusButton")
        self.bus_button.clicked.connect(self.go_to_bus)
        bus_row.addWidget(self.bus_button)
        right.addLayout(bus_row)
        self.connections_label = QtWidgets.QLabel(details_panel)
        right.addWidget(self.connections_label)
        self.connections = QtWidgets.QTableWidget(0, 3, details_panel)
        self.connections.setObjectName("studyBusConnections")
        self.connections.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.connections.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.connections.setSortingEnabled(True)
        self.connections.horizontalHeader().setStretchLastSection(True)
        self.connections.doubleClicked.connect(self._connection_selected)
        right.addWidget(self.connections, 1)
        self.property_filter = QtWidgets.QLineEdit(details_panel)
        self.property_filter.setObjectName("studyPropertySearch")
        self.property_filter.setClearButtonEnabled(True)
        self.property_filter.textChanged.connect(self._filter_properties)
        right.addWidget(self.property_filter)
        self.properties = QtWidgets.QTableView(details_panel)
        self.properties.setObjectName("studySnapshotProperties")
        self.properties.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.properties, 3)
        self.note = QtWidgets.QLabel(details_panel)
        self.note.setWordWrap(True)
        right.addWidget(self.note)
        split.setSizes([380, 700])
        self._search_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Find, self)
        self._search_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._search_shortcut.activated.connect(self.search.setFocus)
        self.retranslate_ui()
        self._state_timer = QtCore.QTimer(self)
        self._state_timer.setInterval(500)
        self._state_timer.timeout.connect(self._update_state)
        self._state_timer.start()

    def _update_state(self) -> None:
        """A file/session switch may happen while this tab stays selected."""
        if self.isVisible():
            if self.get_circuit() is not self.circuit and not self.is_busy():
                self.refresh()
            self.properties.setEnabled(self._can_edit())
            self.refresh_button.setEnabled(not self.is_busy())

    def retranslate_ui(self) -> None:
        self.back_button.setText(self.tr("Back"))
        self.refresh_button.setText(self.tr("Refresh equipment"))
        self.results_button.setText(self.tr("Results"))
        self.bus_button.setText(self.tr("Go to bus"))
        self.search.setPlaceholderText(self.tr("Search name, code, ID or connected bus…"))
        self.search.setAccessibleName(self.tr("Study equipment search"))
        self.property_filter.setPlaceholderText(self.tr("Filter snapshot properties…"))
        self.property_filter.setAccessibleName(self.tr("Study property filter"))
        self.source.setHorizontalHeaderLabels([self.tr("Type"), self.tr("Name"), self.tr("Code")])
        self.connections.setHorizontalHeaderLabels([self.tr("Type"), self.tr("Name"), self.tr("Code")])
        self.note.setText(self.tr("Edit snapshot data here; time profiles are unchanged. "
                                  "Run power flow again after electrical changes and consult Results."))
        if self.selected is None:
            self.heading.setText(self.tr("Select equipment to inspect or edit its snapshot data."))

    def refresh(self) -> None:
        """Refresh on entry or explicitly after edits made in another view."""
        if self.is_busy():
            return
        circuit = self.get_circuit()
        same_circuit = circuit is self.circuit
        previous = self.selected if same_circuit else None
        self.circuit = circuit
        if not same_circuit:
            self.history.clear()
        self._refreshing = True
        self.devices = [device for device in circuit.get_all_elements_iter()
                        if isinstance(device, PhysicalDevice)]
        self.source.removeRows(0, self.source.rowCount())
        old_type = self.category.currentData()
        self.category.clear()
        self.category.addItem(self.tr("All equipment"), None)
        types = sorted({device.device_type for device in self.devices}, key=lambda item: item.value)
        for device_type in types:
            self.category.addItem(device_type.value, device_type)
        self.category.setCurrentIndex(max(0, self.category.findData(old_type)))
        for device in self.devices:
            row = [QtGui.QStandardItem(str(text)) for text in
                   (device.device_type.value, device.name, device.code or "")]
            row[0].setData(device, QtCore.Qt.ItemDataRole.UserRole)
            row[0].setData(" ".join([device.device_type.value, device.name, device.code or "", device.idtag]
                                   + [f"{bus.name} {bus.code}" for bus in connected_buses(device)]).casefold(),
                           QtCore.Qt.ItemDataRole.UserRole + 1)
            row[0].setToolTip(device.idtag)
            self.source.appendRow(row)
        self._refreshing = False
        self._filter()
        if previous is not None and any(previous is device for device in self.devices):
            self.open_device(previous, add_history=False)
        else:
            self._clear_selection()
        self.back_button.setEnabled(bool(self.history))

    def _filter(self, *_unused) -> None:
        if self._refreshing:
            return
        self.proxy.apply(self.search.text(), self.category.currentData())
        self.matches_label.setText(self.tr("{count} matches").format(count=self.proxy.rowCount()))

    def _hit_selected(self, current: QtCore.QModelIndex, _previous) -> None:
        if current.isValid() and not self._refreshing:
            source = self.proxy.mapToSource(current.siblingAtColumn(0))
            self.open_device(self.source.item(source.row(), 0).data(QtCore.Qt.ItemDataRole.UserRole))

    def _can_edit(self, device: PhysicalDevice | None = None) -> bool:
        device = self.selected if device is None else device
        return (not self.is_busy() and self.get_circuit() is self.circuit and device is not None
                and device is self.selected and any(other is device for other in
                        self.circuit.get_elements_by_type(device.device_type)))

    def _clear_selection(self) -> None:
        self.selected = None
        self._dispose_property_model()
        self.connections.setRowCount(0)
        self.bus_selector.clear()
        self.bus_button.setEnabled(False)
        self.heading.setText(self.tr("Select equipment to inspect or edit its snapshot data."))
        self.connections_label.setText(self.tr("No equipment selected"))

    def _dispose_property_model(self) -> None:
        """Native delegates are parented to the view, rather than the old model."""
        self.properties.setModel(None)
        if self.properties_model is not None:
            for row in range(len(self.properties_model.attributes)):
                delegate = self.properties.itemDelegateForRow(row)
                self.properties.setItemDelegateForRow(row, None)
                if delegate is not None:
                    delegate.deleteLater()
            self.properties_model.deleteLater()
            self.properties_model = None

    def open_device(self, device: PhysicalDevice, add_history: bool = True) -> None:
        if self.get_circuit() is not self.circuit or not any(device is other for other in self.devices):
            return
        if self.selected is not device and add_history and self.selected is not None:
            self.history.append(self.selected)
        self.selected = device
        self.back_button.setEnabled(bool(self.history))
        self._dispose_property_model()
        # A separate table shares the exact native objects, metadata and delegates.
        _, dictionary = self.circuit.get_dictionary_of_lists(elm_type=device.device_type)
        self.properties_model = StudyObjectsModel(
            can_edit=lambda: self._can_edit(device), objects=[device], property_list=list(device.property_list),
            time_index=None, parent=self.properties, editable=True, transposed=True,
            dictionary_of_lists=dictionary, error_msg_ptr=self.error_message)
        self.properties.setModel(self.properties_model)
        self.properties_model.dataChanged.connect(self._edited)
        self._show_context()
        self._filter_properties()

    def _show_context(self) -> None:
        device = self.selected
        if device is None:
            return
        self.heading.setText(f"{device.device_type.value} · {device.name} · {device.code or device.idtag}")
        self.bus_selector.clear()
        for bus in connected_buses(device):
            self.bus_selector.addItem(f"{bus.name} · {bus.code or bus.idtag}", bus)
        self.bus_button.setEnabled(self.bus_selector.count() > 0)
        self.connections.setSortingEnabled(False)
        self.connections.setRowCount(0)
        if isinstance(device, Bus):
            connected = [other for other in self.devices
                         if any(bus is device for bus in connected_buses(other))]
            self.connections.setRowCount(len(connected))
            for row, other in enumerate(connected):
                for col, text in enumerate((other.device_type.value, other.name, other.code or "")):
                    item = QtWidgets.QTableWidgetItem(str(text))
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, other)
                    self.connections.setItem(row, col, item)
            self.connections_label.setText(self.tr("Connected equipment — double-click to open"))
        else:
            self.connections_label.setText(self.tr("Choose an endpoint and use Go to bus to inspect connections."))
        self.connections.setSortingEnabled(True)

    def _filter_properties(self, *_unused) -> None:
        if self.properties_model is not None:
            words = self.property_filter.text().casefold().split()
            for row, prop in enumerate(self.properties_model.property_list):
                text = f"{prop.name} {prop.units} {prop.definition}".casefold()
                self.properties.setRowHidden(row, not all(word in text for word in words))

    def _edited(self, *_unused) -> None:
        self.note.setText(self.tr("Snapshot data changed. Previous results are historical; "
                                  "run power flow again to update them. Time profiles are unchanged."))
        self._show_context()
        # Update search labels without replacing an editor that is still committing.
        for row in range(self.source.rowCount()):
            item = self.source.item(row, 0)
            device = item.data(QtCore.Qt.ItemDataRole.UserRole)
            self.source.item(row, 1).setText(device.name)
            self.source.item(row, 2).setText(device.code or "")
            item.setData(" ".join([device.device_type.value, device.name, device.code or "", device.idtag]
                                 + [f"{bus.name} {bus.code}" for bus in connected_buses(device)]).casefold(),
                         QtCore.Qt.ItemDataRole.UserRole + 1)
        self._filter()

    def go_to_bus(self) -> None:
        bus = self.bus_selector.currentData()
        if bus is not None:
            self.open_device(bus)

    def _connection_selected(self, index: QtCore.QModelIndex) -> None:
        item = self.connections.item(index.row(), 0)
        if item is not None:
            self.open_device(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def go_back(self) -> None:
        while self.history:
            device = self.history.pop()
            if any(device is other for other in self.devices):
                self.open_device(device, add_history=False)
                break
        self.back_button.setEnabled(bool(self.history))
