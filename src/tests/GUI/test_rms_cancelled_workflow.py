"""Unit coverage for RMS completion messages, separate from desktop acceptance."""
from types import SimpleNamespace

import numpy as np
import pytest

from VeraGrid.Gui.Main.SubClasses.simulations import SimulationsMain
from VeraGridEngine.basic_structures import Logger


@pytest.mark.parametrize("cancelled, available, converged, expected", [
    (True, None, False, "RMS cancelled before producing results"),
    (True, False, False, "RMS cancelled before producing results"),
    (True, True, False, "RMS cancelled with partial results"),
    # Cancellation between groups must not report the whole batch as complete.
    (True, True, True, "RMS cancelled with partial results"),
    (False, True, False, "RMS stopped with partial results"),
    (False, True, True, None),
])
def test_rms_completion_distinguishes_cancellation(cancelled, available, converged, expected):
    labels, warnings, errors, infos, actions = [], [], [], [], []
    driver = SimpleNamespace(is_cancel=lambda: cancelled, logger=Logger())
    results = None if available is None else SimpleNamespace(
        rms_events_group_names=np.array(["Group"]),
        has_event_group_results=np.array([available]),
        well_initialized=np.array([True]),
        converged=np.array([converged]),
    )
    window = SimpleNamespace(
        session=SimpleNamespace(rms_dynamic_simulation=(driver, results), is_anything_running=lambda: False),
        circuit=SimpleNamespace(rms_events_groups=[SimpleNamespace(active=True)]),
        remove_simulation=lambda study: actions.append("remove"),
        update_available_results=lambda: actions.append("refresh"),
        UNLOCK=lambda: actions.append("unlock"),
        ui=SimpleNamespace(progress_label=SimpleNamespace(setText=labels.append)),
        show_warning_toast=warnings.append,
        show_error_toast=lambda text, **kwargs: errors.append(text),
        show_info_toast=infos.append,
        tr=lambda text: text,
    )
    SimulationsMain.post_rms(window)
    assert actions == ["remove", "refresh", "unlock"]
    if expected is not None:
        assert labels == [expected]
    if cancelled:
        assert warnings == [expected]
        assert infos == []
        assert errors == []
    elif converged:
        assert any("converged" in message for message in infos)
    else:
        assert len(errors) == 1
