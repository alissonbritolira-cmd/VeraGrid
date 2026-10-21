"""Cancellation must retain samples returned by the real RMS integrators."""
from copy import deepcopy

import numpy as np
import pytest

import VeraGridEngine.api as vg
from Base.test_dynamic_results_save_load import _build_rms_driver
from VeraGrid.Session.session import SimulationSession
from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
from VeraGridEngine.IO.veragrid.zip_interface import load_session_driver_objects
from VeraGridEngine.Simulations.Rms import rms_driver
from VeraGridEngine.Simulations.Rms.rms_driver import RmsSimulationDriver
from VeraGridEngine.enumerations import RmsProblemTypes, SimulationTypes


@pytest.fixture(scope="module")
def public_rms_case():
    return _build_rms_driver()


def _driver(public_case, problem_type=RmsProblemTypes.PowerBalance):
    grid, options, flow = deepcopy((public_case.grid, public_case.options, public_case.pf_results))
    options.problem_type = problem_type
    return RmsSimulationDriver(grid=grid, options=options, pf_results=flow)


class _CancelDuringIntegration:
    def __init__(self, driver, after_completed_group=False):
        self.driver = driver
        self.after_completed_group = after_completed_group

    def emit(self, percent):
        results = self.driver.results
        if results is not None and 25 <= float(percent) < 90:
            if not self.after_completed_group or results.has_event_group_results[0]:
                self.driver.cancel()


@pytest.mark.parametrize("problem_type, solver_name", [
    (RmsProblemTypes.PowerBalance, "BackEulerImplicitIntegration"),
    (RmsProblemTypes.PowerBalanceVectorized, "BackEulerImplicitIntegrationVec"),
    (RmsProblemTypes.PowerBalanceFullVectorized, "BackEulerImplicitIntegrationFullVec"),
])
def test_cancel_retains_real_integrator_prefix(public_rms_case, monkeypatch, problem_type, solver_name):
    complete = _driver(public_rms_case, problem_type)
    complete.run()
    assert complete.results.converged[0]
    assert not complete.results.cancelled[0]

    returned = []
    solver_type = getattr(rms_driver, solver_name)
    real_simulate = solver_type.simulate

    def observe_return(solver):
        result = real_simulate(solver)
        returned.append(result)
        return result

    monkeypatch.setattr(solver_type, "simulate", observe_return)
    cancelled = _driver(public_rms_case, problem_type)
    cancelled.progress_signal = _CancelDuringIntegration(cancelled)
    cancelled.run()
    times, values, initialized, _ = returned[0]
    results = cancelled.results
    assert 1 < len(times) < len(results.time_array)
    assert results.has_event_group_results[0]
    assert results.cancelled[0]
    assert not results.converged[0]
    assert results.well_initialized[0] == initialized
    np.testing.assert_array_equal(results.values[:len(times), :, 0], values)
    np.testing.assert_array_equal(values, complete.results.values[:len(times), :, 0])
    assert np.isnan(results.values[len(times):, :, 0]).all()
    assert "cancel" in str(cancelled.logger).lower()


def test_cancel_preserves_completed_groups_and_leaves_unstarted_groups_unavailable(public_rms_case):
    driver = _driver(public_rms_case)
    driver.grid.add_rms_events_group(RmsEventsGroup(name="Inactive", active=False))
    driver.grid.add_rms_events_group(RmsEventsGroup(name="Interrupted"))
    driver.grid.add_rms_events_group(RmsEventsGroup(name="Not started"))
    driver.progress_signal = _CancelDuringIntegration(driver, after_completed_group=True)
    driver.run()
    results = driver.results
    np.testing.assert_array_equal(results.has_event_group_results, [True, False, True, False])
    np.testing.assert_array_equal(results.converged, [True, False, False, False])
    np.testing.assert_array_equal(results.cancelled, [False, False, True, False])
    assert np.isfinite(results.values[:, :, 0]).all()
    assert np.isfinite(results.values[:, :, 2]).all(axis=1).sum() < results.nt
    assert np.isnan(results.values[-1, :, 2]).all()


def test_cancel_before_preparation_does_not_fabricate_results(public_rms_case):
    driver = _driver(public_rms_case)
    driver.cancel()
    driver.run()
    assert driver.results is None


def test_cancel_before_integrating_group_does_not_publish_placeholder_rows(public_rms_case):
    driver = _driver(public_rms_case)

    class CancelAtGroupStart:
        def emit(self, percent):
            if driver.results is not None and percent == 5:
                driver.cancel()

    driver.progress_signal = CancelAtGroupStart()
    driver.run()
    assert not driver.results.has_event_group_results.any()
    assert not driver.results.converged.any()
    np.testing.assert_array_equal(driver.results.cancelled, [True])


def test_numerical_failure_is_partial_without_being_marked_cancelled(public_rms_case):
    driver = _driver(public_rms_case)
    driver.grid.rms_events[0].value = -1000.0
    driver.options.max_iter = 25
    driver.run()
    results = driver.results
    assert results.has_event_group_results[0]
    assert not results.converged[0]
    assert not results.cancelled[0]
    accepted = np.isfinite(results.values[:, :, 0]).all(axis=1).sum()
    assert 1 < accepted < results.nt
    assert np.isnan(results.values[accepted:, :, 0]).all()


@pytest.mark.parametrize("legacy", [False, True])
def test_cancelled_result_archive_restores_samples_status_and_metadata(public_rms_case, tmp_path, legacy):
    driver = _driver(public_rms_case)
    driver.progress_signal = _CancelDuringIntegration(driver)
    driver.run()
    path = tmp_path / "cancelled.veragrid"
    vg.save_file(driver.grid, str(path), drivers_to_save=[driver.get_save_data()])
    stored = load_session_driver_objects(str(path), driver.name, driver.tpe.value)
    if legacy:
        stored.pop("cancelled", None)
    session = SimulationSession()
    logger = session.register_driver_from_disk_data(
        grid=driver.grid, study_name=SimulationTypes.RmsDynamic_run.value, data_dict=stored,
    )
    assert not logger.has_logs(), str(logger)
    restored = session.get_results(SimulationTypes.RmsDynamic_run)
    np.testing.assert_array_equal(restored.values, driver.results.values)
    np.testing.assert_array_equal(restored.has_event_group_results, driver.results.has_event_group_results)
    np.testing.assert_array_equal(restored.converged, driver.results.converged)
    np.testing.assert_array_equal(restored.cancelled, [not legacy])
    assert restored.devices_vars_info
    assert restored.parameter_value_maps == driver.results.parameter_value_maps
    assert restored.initial_parameter_value_maps == driver.results.initial_parameter_value_maps
