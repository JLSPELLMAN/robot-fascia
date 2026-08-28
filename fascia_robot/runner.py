from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .metrics import arm_metrics, perturbation_metrics
from .model import actuator_id, load_model
from .plotting import plot_arm, plot_perturbation
from .telemetry import Recorder, write_csv
from .trajectory import arm_raise_offset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _context(config: dict):
    model_cfg = config["model"]
    np.random.seed(int(model_cfg["seed"]))
    return load_model(PROJECT_ROOT / model_cfg["path"], model_cfg["keyframe"], float(model_cfg["timestep"]))


def _save(experiment: str, recorder: Recorder, metrics: dict, config: dict, output_root: Path) -> dict:
    out = output_root / experiment
    out.mkdir(parents=True, exist_ok=True)
    arrays = recorder.arrays()
    write_csv(out / "telemetry.csv", arrays, recorder.context)
    payload = {
        "experiment": experiment,
        "variant": "baseline",
        "fascia_present": False,
        "model_source_commit": "da76818e269b82289eba39808e2fb91d679d6994",
        "mujoco_version": mujoco.__version__,
        "timestep_s": recorder.context.model.opt.timestep,
        "seed": int(config["model"]["seed"]),
        "energy_definition": "time integral of sum(abs(actuator_torque * joint_velocity)); mechanical, not electrical",
        "metrics": metrics,
        "experiment_config": config[experiment],
    }
    with (out / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    return arrays


def run_arm_raise(config: dict, output_root: Path) -> dict:
    ctx = _context(config)
    cfg = config["arm_raise"]
    aid = actuator_id(ctx.model, cfg["joint"])
    dof = int(ctx.actuator_dof_ids[aid])
    recorder = Recorder(ctx)
    steps = round(float(cfg["duration"]) / ctx.model.opt.timestep)
    for _ in range(steps):
        target = ctx.home_ctrl.copy()
        target[aid] += arm_raise_offset(ctx.data.time, cfg)
        ctx.data.ctrl[:] = target
        ctx.data.xfrc_applied[:] = 0.0
        mujoco.mj_step(ctx.model, ctx.data)
        recorder.sample(target, np.zeros(3))
    arrays = recorder.arrays()
    metrics = arm_metrics(arrays, aid, dof, cfg)
    arrays = _save("arm_raise", recorder, metrics, config, output_root)
    plot_arm(output_root / "arm_raise" / "baseline.png", arrays, aid, dof)
    return metrics


def run_perturbation(config: dict, output_root: Path) -> dict:
    ctx = _context(config)
    cfg = config["perturbation"]
    body_id = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"])
    if body_id < 0:
        raise ValueError(f"Body not found: {cfg['body']}")
    axis = np.asarray(cfg["axis_world"], dtype=float)
    axis /= np.linalg.norm(axis)
    recorder = Recorder(ctx)
    steps = round(float(cfg["duration"]) / ctx.model.opt.timestep)
    for _ in range(steps):
        ctx.data.ctrl[:] = ctx.home_ctrl
        ctx.data.xfrc_applied[:] = 0.0
        active = cfg["force_start"] <= ctx.data.time < cfg["force_start"] + cfg["force_duration"]
        force = axis * float(cfg["force_newtons"]) if active else np.zeros(3)
        ctx.data.xfrc_applied[body_id, :3] = force
        mujoco.mj_step(ctx.model, ctx.data)
        recorder.sample(ctx.home_ctrl, force)
    arrays = recorder.arrays()
    metrics = perturbation_metrics(arrays, cfg)
    arrays = _save("perturbation", recorder, metrics, config, output_root)
    plot_perturbation(output_root / "perturbation" / "baseline.png", arrays, cfg)
    return metrics

