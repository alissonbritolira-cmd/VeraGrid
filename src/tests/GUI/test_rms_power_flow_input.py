"""Regression coverage for refusing stale PF inputs before GUI RMS dispatch.

The dispatcher tests isolate its message/session callbacks. They are unit tests,
not a replacement for navigating the real Windows application.
"""
from types import SimpleNamespace
import numpy as np
import pytest

import VeraGridEngine.api as vg
from VeraGrid.Gui.Main.SubClasses import simulations
from VeraGridEngine.Simulations.PowerFlow import power_flow_driver
from VeraGridEngine.Simulations.PowerFlow.power_flow_input import power_flow_input_fingerprint
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
from VeraGridEngine.IO.veragrid.zip_interface import load_session_driver_objects


def _grid():
    grid = vg.MultiCircuit()
    first = vg.Bus(name="Source", Vnom=110, is_slack=True)
    second = vg.Bus(name="Load bus", Vnom=110)
    grid.add_bus(first)
    grid.add_bus(second)
    grid.add_line(vg.Line(bus_from=first, bus_to=second, r=0.01, x=0.1))
    grid.add_generator(first, vg.Generator(P=10, vset=1.0))
    grid.add_load(second, vg.Load(P=10, Q=1))
    return grid


def _flow(grid):
    driver = vg.PowerFlowDriver(grid, vg.PowerFlowOptions(
        control_q=False, control_taps_modules=False, control_taps_phase=False,
        control_remote_voltage=False, retry_with_other_methods=False,
    ))
    driver.run()
    assert driver.results.converged
    return driver


def _dispatch(grid, pf_driver, monkeypatch):
    messages, calls = [], []
    monkeypatch.setattr(simulations, "info_msg", messages.append)
    window = SimpleNamespace(
        circuit=grid,
        session=SimpleNamespace(
            power_flow=(pf_driver, pf_driver.results),
            run=lambda driver, **kwargs: calls.append(("run", driver)),
        ),
        get_selected_rms_simulation_options=lambda: RmsOptions(simulation_time=0.02),
        remove_simulation=lambda study: calls.append(("remove", study)),
        add_simulation=lambda study: calls.append(("add", study)),
        LOCK=lambda: calls.append(("lock", None)),
        tr=lambda text: text,
        post_rms=lambda: None,
        ui=SimpleNamespace(progress_label=SimpleNamespace(setText=lambda text: None),
                           progressBar=SimpleNamespace(setValue=lambda value: None)),
    )
    simulations.SimulationsMain.run_rms(window)
    return messages, calls


def test_rms_dispatch_refuses_edited_load_before_starting_another_study(monkeypatch):
    grid = _grid()
    driver = _flow(grid)
    grid.get_loads()[0].P = 15
    messages, calls = _dispatch(grid, driver, monkeypatch)
    assert len(messages) == 1
    assert "Run a power flow again" in messages[0]
    assert calls == []


def test_rms_dispatch_accepts_unchanged_and_recomputed_power_flow(monkeypatch):
    grid = _grid()
    for power in (10, 15):
        grid.get_loads()[0].P = power
        driver = _flow(grid)
        messages, calls = _dispatch(grid, driver, monkeypatch)
        assert messages == []
        rms_driver = [value for action, value in calls if action == "run"]
        assert len(rms_driver) == 1
        assert rms_driver[0].pf_results is driver.results


def test_rms_dispatch_requests_new_pf_for_legacy_results(monkeypatch):
    grid = _grid()
    driver = _flow(grid)
    driver.results.input_fingerprint = np.array([""])
    messages, calls = _dispatch(grid, driver, monkeypatch)
    assert "Run a power flow again" in messages[0]
    assert calls == []


@pytest.mark.parametrize("change", [
    lambda grid: setattr(grid.get_loads()[0], "P", 15),
    lambda grid: setattr(grid.get_loads()[0], "Q", 3),
    lambda grid: setattr(grid.get_loads()[0], "active", False),
    lambda grid: setattr(grid.get_loads()[0], "bus", grid.buses[0]),
    lambda grid: setattr(grid.lines[0], "R", 0.02),
    lambda grid: setattr(grid.lines[0], "bus_to", grid.buses[0]),
    lambda grid: setattr(grid.generators[0], "Vset", 1.02),
    lambda grid: setattr(grid, "Sbase", 200),
    lambda grid: setattr(grid, "fBase", 50),
    lambda grid: grid.buses.reverse(),
])
def test_electrical_edits_change_input_identity(change):
    grid = _grid()
    before = power_flow_input_fingerprint(grid)
    change(grid)
    assert before != power_flow_input_fingerprint(grid)


def test_names_comments_and_diagram_coordinates_do_not_invalidate_pf():
    grid = _grid()
    before = power_flow_input_fingerprint(grid)
    grid.buses[0].name = "Renamed"
    grid.buses[0].x = 100
    grid.buses[0].y = 200
    grid.lines[0].comment = "A note"
    assert power_flow_input_fingerprint(grid) == before


def test_shunt_arrays_and_tap_changer_settings_are_included():
    grid = _grid()
    shunt = vg.ControllableShunt()
    grid.add_controllable_shunt(grid.buses[1], shunt)
    shunt.b_steps = np.arange(2000, dtype=float)
    before = power_flow_input_fingerprint(grid)
    shunt.b_steps[1000] += 0.01
    assert power_flow_input_fingerprint(grid) != before
    transformer = vg.Transformer2W(bus_from=grid.buses[0], bus_to=grid.buses[1])
    grid.add_transformer2w(transformer)
    before = power_flow_input_fingerprint(grid)
    transformer.tap_changer.dV += 0.01
    assert power_flow_input_fingerprint(grid) != before


def test_power_flow_input_identity_survives_result_archive(tmp_path):
    grid = _grid()
    driver = _flow(grid)
    path = tmp_path / "pf_input.veragrid"
    vg.save_file(grid, str(path), drivers_to_save=[driver.get_save_data()])
    stored = load_session_driver_objects(str(path), driver.name, driver.tpe.value)
    reopened = vg.open_file(str(path))
    restored = vg.PowerFlowDriver(reopened)
    restored.results.parse_saved_data(grid=reopened, data_dict=stored)
    np.testing.assert_array_equal(restored.results.input_fingerprint, driver.results.input_fingerprint)
    assert restored.results.input_fingerprint[0] == power_flow_input_fingerprint(reopened)


def test_profiles_use_the_solved_time_index():
    grid = _grid()
    grid.create_profiles(steps=2, step_length=1, step_unit="h")
    grid.get_loads()[0].P_prof.set(np.array([10.0, 15.0]))
    first = vg.PowerFlowDriver(grid, t_idx=0)
    second = vg.PowerFlowDriver(grid, t_idx=1)
    first.run()
    second.run()
    assert first.results.input_fingerprint[0] == power_flow_input_fingerprint(grid, t_idx=0)
    assert second.results.input_fingerprint[0] == power_flow_input_fingerprint(grid, t_idx=1)
    assert first.results.input_fingerprint[0] != second.results.input_fingerprint[0]


def test_inputs_edited_during_power_flow_do_not_receive_a_verified_identity(monkeypatch):
    grid = _grid()
    real_solver = power_flow_driver.multi_island_pf

    def solve_then_edit(**kwargs):
        results = real_solver(**kwargs)
        grid.get_loads()[0].P = 15.0
        return results

    monkeypatch.setattr(power_flow_driver, "multi_island_pf", solve_then_edit)
    driver = _flow(grid)
    assert driver.results.input_fingerprint[0] == ""
