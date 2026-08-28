from __future__ import annotations

import numpy as np


def quaternion_angle(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(q @ reference), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def _first_held_time(time: np.ndarray, ok: np.ndarray, start: float, hold: float) -> float | None:
    indices = np.flatnonzero((time >= start) & ok)
    for idx in indices:
        end = np.searchsorted(time, time[idx] + hold)
        if end < len(time) and np.all(ok[idx:end + 1]):
            return float(time[idx] - start)
    return None


def common_metrics(a: dict[str, np.ndarray], analysis_start: float) -> dict[str, float]:
    dt = np.gradient(a["time"])
    window = a["time"] >= analysis_start
    return {
        "analysis_start_s": float(analysis_start),
        "peak_joint_jerk_rad_s3": float(np.max(np.abs(a["jerk"][window, 6:]))),
        "mean_joint_jerk_rad_s3": float(np.mean(np.abs(a["jerk"][window, 6:]))),
        "peak_actuator_torque_nm": float(np.max(np.abs(a["torque"][window]))),
        "absolute_mechanical_actuator_energy_j": float(np.sum(np.abs(a["power"][window]) * dt[window, None])),
    }


def arm_metrics(a: dict[str, np.ndarray], actuator_id: int, dof_id: int, cfg: dict) -> dict[str, float | None]:
    result = common_metrics(a, float(cfg["warmup"]))
    q = a["qpos"][:, dof_id + 1]  # floating base nq is nv+1 before hinge coordinates
    target = a["target"][:, actuator_id]
    peak_target = float(np.max(target))
    hold = (a["time"] >= cfg["raise_end"]) & (a["time"] <= cfg["hold_end"])
    overshoot = max(0.0, float(np.max(q[hold]) - peak_target))
    tol = np.deg2rad(float(cfg["settling_tolerance_deg"]))
    post = a["time"] >= cfg["lower_end"]
    final_target = float(target[-1])
    result.update({
        "commanded_amplitude_deg": float(cfg["amplitude_deg"]),
        "peak_tracking_error_deg": float(np.rad2deg(np.max(np.abs(q - target)))),
        "overshoot_deg": float(np.rad2deg(overshoot)),
        "settling_time_s": _first_held_time(a["time"], np.abs(q - final_target) <= tol, cfg["lower_end"], cfg["settling_hold"]),
        "peak_commanded_joint_jerk_rad_s3": float(np.max(np.abs(a["jerk"][:, dof_id]))),
        "commanded_joint_mean_jerk_rad_s3": float(np.mean(np.abs(a["jerk"][:, dof_id]))),
    })
    return result


def perturbation_metrics(a: dict[str, np.ndarray], cfg: dict) -> dict[str, float | int | None]:
    result = common_metrics(a, float(cfg["warmup"]))
    start = float(cfg["force_start"])
    baseline_mask = (a["time"] >= start - 0.2) & (a["time"] < start)
    base_com = np.mean(a["com"][baseline_mask], axis=0)
    displacement = a["com"] - base_com
    lateral = displacement[:, 1]
    after = a["time"] >= start
    peak = float(np.max(np.abs(lateral[after])))
    force_end = start + float(cfg["force_duration"])
    threshold = max(float(cfg["recovery_absolute_m"]), peak * float(cfg["recovery_fraction"]))
    reference_quat = np.mean(a["torso_quat"][baseline_mask], axis=0)
    reference_quat /= np.linalg.norm(reference_quat)
    torso_angle = quaternion_angle(a["torso_quat"], reference_quat)
    centered = lateral[after] - np.mean(lateral[-100:])
    crossings = int(np.count_nonzero(np.diff(np.signbit(centered))))
    dt = np.gradient(a["time"])
    recovery = a["time"] >= start
    result.update({
        "impulse_ns": float(cfg["force_newtons"] * cfg["force_duration"]),
        "max_com_displacement_m": float(np.max(np.linalg.norm(displacement[after], axis=1))),
        "max_lateral_com_displacement_m": peak,
        "max_torso_rotation_deg": float(np.rad2deg(np.max(torso_angle[after]))),
        "recovery_time_s": _first_held_time(a["time"], np.abs(lateral) <= threshold, force_end, cfg["recovery_hold"]),
        "post_impact_lateral_zero_crossings": crossings,
        "recovery_absolute_mechanical_energy_j": float(np.sum(np.abs(a["power"][recovery]) * dt[recovery, None])),
    })
    return result

