from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from .fascia.coupling import RobotFasciaCoupling
from .fascia.geometry import build_torso_topology
from .metrics import arm_metrics, common_metrics, perturbation_metrics, quaternion_angle
from .model import actuator_id, load_model
from .runner import PROJECT_ROOT, load_config
from .telemetry import Recorder, write_csv
from .trajectory import arm_raise_offset, squat_phase


METRICS = (
    "peak_joint_jerk_rad_s3",
    "mean_joint_jerk_rad_s3",
    "absolute_mechanical_actuator_energy_j",
    "peak_actuator_torque_nm",
    "max_com_displacement_m",
    "max_torso_rotation_deg",
    "oscillation_count",
    "recovery_time_s",
)

LABELS = {
    "peak_joint_jerk_rad_s3": "Peak jerk (rad/s³)",
    "mean_joint_jerk_rad_s3": "Mean jerk (rad/s³)",
    "absolute_mechanical_actuator_energy_j": "Actuator work (J)",
    "peak_actuator_torque_nm": "Peak torque (N·m)",
    "max_com_displacement_m": "COM displacement (m)",
    "max_torso_rotation_deg": "Torso rotation (deg)",
    "oscillation_count": "Oscillation count",
    "recovery_time_s": "Recovery time (s)",
}


def _first_held(time: np.ndarray, ok: np.ndarray, start: float, hold: float) -> float | None:
    for index in np.flatnonzero((time >= start) & ok):
        end = np.searchsorted(time, time[index] + hold)
        if end < len(time) and np.all(ok[index:end + 1]):
            return float(time[index] - start)
    return None


def _motion_metrics(arrays: dict[str, np.ndarray], cfg: dict, event_start: float, event_end: float, axis: int) -> dict:
    result = common_metrics(arrays, float(cfg["warmup"]))
    pre = (arrays["time"] >= event_start - 0.2) & (arrays["time"] < event_start)
    reference_com = np.mean(arrays["com"][pre], axis=0)
    com_delta = arrays["com"] - reference_com
    active = arrays["time"] >= event_start
    qref = np.mean(arrays["torso_quat"][pre], axis=0); qref /= np.linalg.norm(qref)
    torso_angle = quaternion_angle(arrays["torso_quat"], qref)
    post = arrays["time"] >= event_end
    scalar = com_delta[:, axis]
    centered = scalar[post] - np.mean(scalar[-100:])
    tolerance = float(cfg.get("recovery_absolute_m", 0.005))
    result.update({
        "max_com_displacement_m": float(np.max(np.linalg.norm(com_delta[active], axis=1))),
        "max_torso_rotation_deg": float(np.rad2deg(np.max(torso_angle[active]))),
        "oscillation_count": int(np.count_nonzero(np.diff(np.signbit(centered)))),
        "recovery_time_s": _first_held(arrays["time"], np.linalg.norm(com_delta, axis=1) <= tolerance, event_end, float(cfg.get("recovery_hold", 0.25))),
    })
    return result


def _make_context(config: dict):
    mcfg = config["model"]
    np.random.seed(int(mcfg["seed"]))
    return load_model(PROJECT_ROOT / mcfg["path"], mcfg["keyframe"], float(mcfg["timestep"]))


def _run_trial(experiment: str, fascia_enabled: bool, blind_label: str, config: dict, fconfig: dict, output: Path):
    ctx = _make_context(config)
    cfg = config[experiment]
    coupling = RobotFasciaCoupling(ctx, build_torso_topology(fconfig), fconfig) if fascia_enabled else None
    recorder = Recorder(ctx)
    arm_aid = arm_dof = None
    squat_ids = {}
    if experiment == "arm_raise":
        arm_aid = actuator_id(ctx.model, cfg["joint"]); arm_dof = int(ctx.actuator_dof_ids[arm_aid])
    elif experiment == "squat":
        squat_ids = {actuator_id(ctx.model, name): np.deg2rad(value) for name, value in cfg["joint_offsets_deg"].items()}
    elif experiment == "perturbation":
        perturb_body = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"])
        force_axis = np.asarray(cfg["axis_world"], dtype=float); force_axis /= np.linalg.norm(force_axis)

    fascia_peak_tension = 0.0
    fascia_min_tension = 0.0
    fascia_peak_residual = 0.0
    steps = round(float(cfg["duration"]) / ctx.model.opt.timestep)
    for _ in range(steps):
        time = float(ctx.data.time)
        target = ctx.home_ctrl.copy()
        external = np.zeros(3)
        if experiment == "arm_raise":
            target[arm_aid] += arm_raise_offset(time, cfg)
        elif experiment == "squat":
            phase = squat_phase(time, cfg)
            for aid, offset in squat_ids.items(): target[aid] += phase * offset
        else:
            active = cfg["force_start"] <= time < cfg["force_start"] + cfg["force_duration"]
            external = force_axis * float(cfg["force_newtons"]) if active else np.zeros(3)
        ctx.data.ctrl[:] = target; ctx.data.xfrc_applied[:] = 0.0
        if experiment == "perturbation": ctx.data.xfrc_applied[perturb_body, :3] += external
        if coupling:
            snapshot = coupling.compute_and_apply()
            fascia_peak_tension = max(fascia_peak_tension, float(np.max(snapshot.tensions)))
            fascia_min_tension = min(fascia_min_tension, float(np.min(snapshot.tensions)))
            fascia_peak_residual = max(fascia_peak_residual, snapshot.free_force_residual)
        mujoco.mj_step(ctx.model, ctx.data)
        recorder.sample(target, external)

    arrays = recorder.arrays()
    if experiment == "arm_raise":
        metrics = arm_metrics(arrays, arm_aid, arm_dof, cfg)
        metrics.update(_motion_metrics(arrays, cfg, cfg["raise_start"], cfg["lower_end"], 1))
    elif experiment == "squat":
        metrics = _motion_metrics(arrays, cfg, cfg["descend_start"], cfg["ascend_end"], 2)
    else:
        raw = perturbation_metrics(arrays, cfg)
        metrics = {
            **raw,
            "oscillation_count": raw["post_impact_lateral_zero_crossings"],
        }
    trial_dir = output / experiment / blind_label
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_csv(trial_dir / "telemetry.csv", arrays, ctx)
    payload = {
        "blind_label": blind_label,
        "experiment": experiment,
        "metrics": metrics,
    }
    with (trial_dir / "summary.json").open("w") as handle: json.dump(payload, handle, indent=2)
    diagnostics = None if not coupling else {
        "minimum_edge_tension_n": fascia_min_tension,
        "peak_edge_tension_n": fascia_peak_tension,
        "peak_free_node_residual_n": fascia_peak_residual,
    }
    return metrics, arrays, diagnostics


def _compare_value(baseline, fascia) -> dict:
    if baseline is None and fascia is None:
        return {"percent_change": None, "classification": "unchanged_not_recovered"}
    if baseline is None:
        return {"percent_change": None, "classification": "improvement_recovered_only_with_fascia"}
    if fascia is None:
        return {"percent_change": None, "classification": "regression_recovered_only_without_fascia"}
    if baseline == 0:
        if fascia == 0: return {"percent_change": 0.0, "classification": "unchanged"}
        return {"percent_change": None, "classification": "regression_from_zero"}
    change = 100.0 * (float(fascia) - float(baseline)) / abs(float(baseline))
    classification = "improvement" if change < -0.1 else "regression" if change > 0.1 else "unchanged"
    return {"percent_change": change, "classification": classification}


def _write_comparison(output: Path, comparisons: dict, arrays: dict) -> None:
    rows = []
    for experiment, metrics in comparisons.items():
        for metric, values in metrics.items():
            rows.append([experiment, metric, values["baseline"], values["fascia"], values["percent_change"], values["classification"]])
    with (output / "paired_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["experiment", "metric", "baseline", "fascia", "percent_change", "classification"]); writer.writerows(rows)
    with (output / "paired_metrics.md").open("w") as handle:
        handle.write("| Experiment | Metric | Baseline | Exofascia | Change | Classification |\n|---|---|---:|---:|---:|---|\n")
        for experiment, metric, baseline, fascia, change, classification in rows:
            change_text = "n/a" if change is None else f"{change:+.2f}%"
            handle.write(f"| {experiment} | {LABELS[metric]} | {baseline} | {fascia} | {change_text} | {classification} |\n")

    for experiment, metric_values in comparisons.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.ravel()
        for ax, metric in zip(axes, METRICS):
            values = metric_values[metric]; numeric = [values["baseline"], values["fascia"]]
            if any(v is None for v in numeric):
                ax.text(0.5, 0.5, "not recovered", ha="center", va="center"); ax.set_xticks([])
            else:
                ax.bar(["Baseline", "Exofascia"], numeric, color=["#777777", "#7b2cbf"])
            ax.set_title(LABELS[metric]); ax.tick_params(axis="x", labelrotation=15)
        fig.suptitle(f"Blinded paired comparison — {experiment}"); fig.tight_layout(); fig.savefig(output / f"{experiment}_paired_metrics.png", dpi=150); plt.close(fig)

        base, fascia = arrays[experiment]["baseline"], arrays[experiment]["fascia"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        pre = base["time"] < 1.25; ref_b = np.mean(base["com"][pre], axis=0); ref_f = np.mean(fascia["com"][pre], axis=0)
        axes[0,0].plot(base["time"], np.linalg.norm(base["com"]-ref_b,axis=1), label="baseline"); axes[0,0].plot(fascia["time"], np.linalg.norm(fascia["com"]-ref_f,axis=1), label="exofascia"); axes[0,0].set_ylabel("COM displacement (m)")
        axes[0,1].plot(base["time"], np.max(np.abs(base["jerk"][:,6:]),axis=1)); axes[0,1].plot(fascia["time"], np.max(np.abs(fascia["jerk"][:,6:]),axis=1)); axes[0,1].set_ylabel("Peak joint jerk (rad/s³)")
        axes[1,0].plot(base["time"], np.sum(np.abs(base["power"]),axis=1)); axes[1,0].plot(fascia["time"], np.sum(np.abs(fascia["power"]),axis=1)); axes[1,0].set_ylabel("Σ |actuator power| (W)")
        axes[1,1].plot(base["time"], np.max(np.abs(base["torque"]),axis=1)); axes[1,1].plot(fascia["time"], np.max(np.abs(fascia["torque"]),axis=1)); axes[1,1].set_ylabel("Peak actuator torque (N·m)")
        for ax in axes.ravel(): ax.set_xlabel("Time (s)"); ax.grid(alpha=.2)
        axes[0,0].legend(); fig.suptitle(f"Paired time histories — {experiment}"); fig.tight_layout(); fig.savefig(output / f"{experiment}_paired_timeseries.png", dpi=150); plt.close(fig)


def run_phase4(output: Path | None = None) -> dict:
    output = output or PROJECT_ROOT / "results/phase4"
    output.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT_ROOT / "configs/experiments.yaml"; fascia_path = PROJECT_ROOT / "configs/fascia_phase3.yaml"
    config = load_config(config_path); fconfig = yaml.safe_load(fascia_path.read_text())
    config_hash = hashlib.sha256(config_path.read_bytes() + fascia_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(int(config["model"]["seed"])); conditions = [False, True]; rng.shuffle(conditions)
    mapping = {f"condition_{chr(65+i)}": enabled for i, enabled in enumerate(conditions)}
    blinded_results, decoded_arrays, decoded_diagnostics = {}, {}, {}
    for experiment in ("arm_raise", "squat", "perturbation"):
        blinded_results[experiment] = {}; decoded_arrays[experiment] = {}; decoded_diagnostics[experiment] = {}
        for blind_label, enabled in mapping.items():
            metrics, trial_arrays, diagnostics = _run_trial(experiment, enabled, blind_label, config, fconfig, output)
            blinded_results[experiment][blind_label] = metrics
            decoded_arrays[experiment]["fascia" if enabled else "baseline"] = trial_arrays
            decoded_diagnostics[experiment]["fascia" if enabled else "baseline"] = diagnostics

    comparisons = {}
    baseline_label = next(label for label, enabled in mapping.items() if not enabled)
    fascia_label = next(label for label, enabled in mapping.items() if enabled)
    for experiment in blinded_results:
        comparisons[experiment] = {}
        for metric in METRICS:
            baseline = blinded_results[experiment][baseline_label].get(metric)
            fascia = blinded_results[experiment][fascia_label].get(metric)
            comparisons[experiment][metric] = {"baseline": baseline, "fascia": fascia, **_compare_value(baseline, fascia)}
    counts = {"improvement": 0, "regression": 0, "unchanged": 0}
    for values in comparisons.values():
        for item in values.values():
            category = item["classification"]
            if category.startswith("improvement"): counts["improvement"] += 1
            elif category.startswith("regression"): counts["regression"] += 1
            else: counts["unchanged"] += 1
    result = {
        "protocol": "single deterministic paired simulation; conditions analyzed under labels A/B and decoded only after all six trials completed; descriptive, not statistical inference",
        "configuration_hash": config_hash,
        "blinding_seed": int(config["model"]["seed"]),
        "decoded_mapping": {label: "exofascia" if enabled else "baseline" for label, enabled in mapping.items()},
        "fascia_parameters_optimized": False,
        "edge_behavior": "tension_only",
        "pretension": fconfig["pretension"],
        "classification_threshold_percent": 0.1,
        "decoded_fascia_diagnostics": {experiment: values["fascia"] for experiment, values in decoded_diagnostics.items()},
        "classification_counts": counts,
        "overall_interpretation": "mixed: the fixed exofascia produced both improvements and regressions" if counts["improvement"] and counts["regression"] else "uniform directional result",
        "comparisons": comparisons,
    }
    with (output / "comparison.json").open("w") as handle: json.dump(result, handle, indent=2)
    _write_comparison(output, comparisons, decoded_arrays)
    return result


if __name__ == "__main__":
    print(json.dumps(run_phase4(), indent=2))
