"""Focused native Qt checks; these are not a desktop navigation campaign."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from VeraGrid.Gui.Study.study_workspace import StudyWorkspace, connected_buses
from VeraGridEngine.Devices import Bus, Generator, Line, Load, MultiCircuit, Transformer3W


def _grid():
    grid = MultiCircuit(name="Study workspace demo")
    first = Bus(name="North bus", code="101", Vnom=110)
    second = Bus(name="South bus", code="202", Vnom=110)
    grid.add_bus(first)
    grid.add_bus(second)
    line = Line(bus_from=first, bus_to=second, name="North South", code="L-12", r=0.01, x=0.1)
    grid.add_line(line)
    load = Load(name="Workshop", code="LOAD-1", P=10.0, Q=2.0)
    grid.add_load(bus=second, api_obj=load)
    generator = Generator(name="Generator", code="GEN-1", P=10.0)
    grid.add_generator(bus=first, api_obj=generator)
    return grid, first, second, line, load, generator


def _workspace(qt_app, **kwargs):
    grid, *devices = _grid()
    state = {"grid": grid, "busy": False}
    widget = StudyWorkspace(circuit=lambda: state["grid"], busy=lambda: state["busy"], **kwargs)
    widget.refresh()
    return widget, state, devices


def _choose(widget, device):
    for row in range(widget.proxy.rowCount()):
        source = widget.proxy.mapToSource(widget.proxy.index(row, 0))
        candidate = widget.source.item(source.row(), 0).data(QtCore.Qt.ItemDataRole.UserRole)
        if candidate is device:
            widget.hits.setCurrentIndex(widget.proxy.index(row, 1))
            return
    raise AssertionError("Device not present in visible matches")


def test_search_native_id_code_bus_and_type(qt_app):
    widget, _, devices = _workspace(qt_app)
    first, second, line, load, _generator = devices
    assert widget.proxy.rowCount() == 5
    for query, expected in ((line.idtag, line), ("load-1", load), ("Workshop 202", load)):
        widget.search.setText(query)
        assert widget.proxy.rowCount() == 1
        _choose(widget, expected)
        assert widget.selected is expected
    widget.search.clear()
    widget.category.setCurrentIndex(widget.category.findData(first.device_type))
    assert widget.proxy.rowCount() == 2
    widget.search.setText("101")
    assert widget.proxy.rowCount() == 1
    _choose(widget, first)
    assert widget.selected is first
    assert second is not first


def test_filtered_sorted_selection_edits_actual_native_object(qt_app):
    widget, state, devices = _workspace(qt_app)
    _first, _second, _line, load, generator = devices
    widget.proxy.sort(1, QtCore.Qt.SortOrder.DescendingOrder)
    widget.search.setText("Workshop")
    _choose(widget, load)
    model = widget.properties_model
    row = model.attributes.index("P")
    assert model.setData(model.index(row, 0), 15.0, QtCore.Qt.ItemDataRole.EditRole)
    assert load.P == 15.0
    assert generator.P == 10.0
    assert state["grid"].loads[0] is load
    assert "run power flow" in widget.note.text().lower()


def test_bus_connections_navigation_back_and_sorted_identity(qt_app):
    widget, _, devices = _workspace(qt_app)
    first, second, line, load, _generator = devices
    widget.open_device(second)
    assert widget.connections.rowCount() == 2
    widget.connections.sortItems(1, QtCore.Qt.SortOrder.DescendingOrder)
    row = next(row for row in range(widget.connections.rowCount())
               if widget.connections.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole) is load)
    widget._connection_selected(widget.connections.model().index(row, 1))
    assert widget.selected is load
    assert widget.bus_selector.count() == 1
    widget.go_to_bus()
    assert widget.selected is second
    widget.go_back()
    assert widget.selected is load
    widget.open_device(line)
    assert widget.bus_selector.count() == 2
    widget.bus_selector.setCurrentIndex(0)
    widget.go_to_bus()
    assert widget.selected is first


def test_native_snapshot_editor_does_not_modify_time_profiles(qt_app):
    widget, state, devices = _workspace(qt_app)
    load = devices[3]
    state["grid"].create_profiles(steps=3, step_length=1, step_unit="h")
    before = load.P_prof.toarray().copy()
    widget.open_device(load)
    model = widget.properties_model
    assert model.time_index_ is None
    assert model.setData(model.index(model.attributes.index("P"), 0), 22.0)
    assert load.P == 22.0
    np.testing.assert_array_equal(load.P_prof.toarray(), before)


def test_property_filter_keeps_native_rows_and_delegates(qt_app):
    widget, _, devices = _workspace(qt_app)
    widget.open_device(devices[3])
    model = widget.properties_model
    row = model.attributes.index("active")
    delegate = widget.properties.itemDelegateForRow(row)
    widget.property_filter.setText("active")
    assert not widget.properties.isRowHidden(row)
    assert widget.properties.itemDelegateForRow(row) is delegate
    assert model.setData(model.index(row, 0), QtCore.Qt.CheckState.Unchecked,
                         QtCore.Qt.ItemDataRole.CheckStateRole)
    assert devices[3].active is False


def test_busy_or_new_grid_rejects_an_editor_commit(qt_app):
    widget, state, devices = _workspace(qt_app)
    load = devices[3]
    widget.open_device(load)
    model = widget.properties_model
    index = model.index(model.attributes.index("P"), 0)
    state["busy"] = True
    assert not model.setData(index, 99.0)
    assert not (model.flags(index) & QtCore.Qt.ItemFlag.ItemIsEditable)
    state["busy"] = False
    state["grid"] = MultiCircuit()
    assert not model.setData(index, 99.0)
    assert load.P == 10.0
    widget.refresh()
    assert widget.selected is None
    assert widget.properties.model() is None
    assert widget.source.rowCount() == 0
    assert not widget.history


def test_removed_equipment_rejects_pending_commit(qt_app):
    widget, state, devices = _workspace(qt_app)
    load = devices[3]
    widget.open_device(load)
    model = widget.properties_model
    state["grid"].loads.remove(load)
    assert not model.setData(model.index(model.attributes.index("P"), 0), 99.0)
    widget.refresh()
    assert widget.selected is None


def test_selection_change_rejects_previous_editor_commit(qt_app):
    widget, _, devices = _workspace(qt_app)
    widget.open_device(devices[3])
    old_model = widget.properties_model
    old_index = old_model.index(old_model.attributes.index("P"), 0)
    widget.open_device(devices[4])
    assert not old_model.setData(old_index, 99.0)
    assert devices[3].P == 10.0
    assert devices[4].P == 10.0


def test_refresh_preserves_object_identity_and_renamed_search(qt_app):
    widget, _, devices = _workspace(qt_app)
    load = devices[3]
    widget.open_device(load)
    load.name = "Updated workshop"
    widget.refresh()
    assert widget.selected is load
    widget.search.setText("Updated")
    assert widget.proxy.rowCount() == 1


def test_rename_under_active_filter_preserves_selected_object(qt_app):
    widget, _, devices = _workspace(qt_app)
    load = devices[3]
    widget.search.setText("Workshop")
    _choose(widget, load)
    model = widget.properties_model
    assert model.setData(model.index(model.attributes.index("name"), 0), "Renamed load")
    assert load.name == "Renamed load"
    assert widget.selected is load
    widget.search.setText("Renamed")
    assert widget.proxy.rowCount() == 1


def test_repeated_selection_disposes_view_owned_delegates(qt_app):
    widget, _, devices = _workspace(qt_app)
    for _ in range(5):
        for device in devices:
            widget.open_device(device)
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            qt_app.processEvents()
    delegates = widget.properties.findChildren(QtWidgets.QItemDelegate,
        options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)
    assert len(delegates) <= len(widget.properties_model.attributes)


def test_three_winding_endpoints_are_distinct_native_buses(qt_app):
    grid, first, second, *_ = _grid()
    third = Bus(name="Third bus")
    grid.add_bus(third)
    transformer = Transformer3W(bus1=first, bus2=second, bus3=third)
    assert connected_buses(transformer) == [first, second, third]


def test_actions_reuse_native_callbacks(qt_app):
    invoked = []
    action = QtGui.QAction("Power flow")
    action.triggered.connect(lambda: invoked.append("power flow"))
    widget, _, _ = _workspace(qt_app, power_flow_action=action,
                               show_results=lambda: invoked.append("results"))
    button = next(button for button in widget.findChildren(QtWidgets.QToolButton)
                  if button.defaultAction() is action)
    button.click()
    widget.results_button.click()
    assert invoked == ["power flow", "results"]


def test_main_tab_integration_refreshes_shared_circuit(qt_app):
    from VeraGrid.Gui.Main.SubClasses.Model.data_base import DataBaseTableMain

    gui = DataBaseTableMain()
    grid, _first, _second, _line, load, _generator = _grid()
    gui.circuit = grid
    tabs = gui.ui.mainTabWidget
    assert tabs.indexOf(gui.study_workspace) == tabs.indexOf(gui.ui.ResultsTab) + 1
    tabs.setCurrentWidget(gui.study_workspace)
    assert gui.study_workspace.circuit is grid
    gui.study_workspace.search.setText(load.idtag)
    _choose(gui.study_workspace, load)
    assert gui.study_workspace.selected is load
    gui.study_workspace.results_button.click()
    assert tabs.currentWidget() is gui.ui.ResultsTab
