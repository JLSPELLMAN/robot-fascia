import numpy as np

from fascia_robot.phase4 import METRICS, _compare_value
from fascia_robot.trajectory import squat_phase


def test_squat_phase_is_symmetric_and_bounded():
    cfg = {"descend_start": 1.0, "descend_end": 2.0, "hold_end": 3.0, "ascend_end": 4.0}
    values = np.array([squat_phase(t, cfg) for t in np.linspace(0, 5, 501)])
    assert np.min(values) == 0.0
    assert np.max(values) == 1.0
    assert np.isclose(squat_phase(1.5, cfg), squat_phase(3.5, cfg))


def test_comparison_classifies_lower_values_as_improvements():
    assert _compare_value(10.0, 9.0)["classification"] == "improvement"
    assert _compare_value(10.0, 11.0)["classification"] == "regression"
    assert _compare_value(None, None)["classification"] == "unchanged_not_recovered"
    assert len(METRICS) == 8
