from __future__ import annotations

import numpy as np


def minimum_jerk_phase(t: float, start: float, end: float) -> float:
    """C2-continuous scalar transition from zero to one."""
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    u = (t - start) / (end - start)
    return float(10 * u**3 - 15 * u**4 + 6 * u**5)


def arm_raise_offset(t: float, cfg: dict) -> float:
    amplitude = np.deg2rad(float(cfg["amplitude_deg"]))
    if t < cfg["raise_end"]:
        return amplitude * minimum_jerk_phase(t, cfg["raise_start"], cfg["raise_end"])
    if t < cfg["hold_end"]:
        return float(amplitude)
    return amplitude * (1.0 - minimum_jerk_phase(t, cfg["hold_end"], cfg["lower_end"]))


def squat_phase(t: float, cfg: dict) -> float:
    if t < cfg["descend_end"]:
        return minimum_jerk_phase(t, cfg["descend_start"], cfg["descend_end"])
    if t < cfg["hold_end"]:
        return 1.0
    return 1.0 - minimum_jerk_phase(t, cfg["hold_end"], cfg["ascend_end"])
