from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from fascia_robot.fascia.coupling import RobotFasciaCoupling
from fascia_robot.fascia.geometry import build_torso_topology
from fascia_robot.fascia.telemetry import FasciaRecorder
from fascia_robot.fascia.visualization import plot_network
from fascia_robot.metrics import perturbation_metrics
from fascia_robot.runner import PROJECT_ROOT, load_config
from fascia_robot.model import load_model
from fascia_robot.telemetry import Recorder, write_csv


def run_perturbation_fascia(
    experiment_config: Path | None = None,
    fascia_config: Path | None = None,
    output: Path | None = None,
) -> dict:
    experiment_config = experiment_config or PROJECT_ROOT / "configs/experiments.yaml"
    fascia_config = fascia_config or PROJECT_ROOT / "configs/fascia_phase3.yaml"
    output = output or PROJECT_ROOT / "results/phase3/perturbation_fascia"
    config = load_config(experiment_config)
    with fascia_config.open() as handle:
        fconfig = yaml.safe_load(handle)
    mcfg, cfg = config["model"], config["perturbation"]
    np.random.seed(int(mcfg["seed"]))
    ctx = load_model(PROJECT_ROOT / mcfg["path"], mcfg["keyframe"], float(mcfg["timestep"]))
    body_id = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"])
    axis = np.asarray(cfg["axis_world"], dtype=float); axis /= np.linalg.norm(axis)
    topology = build_torso_topology(fconfig)
    coupling = RobotFasciaCoupling(ctx, topology, fconfig)
    fascia_recorder = FasciaRecorder(coupling.network.topology)
    robot_recorder = Recorder(ctx)
    initial_positions = coupling.network.positions.copy()
    maximum_snapshot = None
    maximum_tension = -1.0
    steps = round(float(cfg["duration"]) / ctx.model.opt.timestep)
    for _ in range(steps):
        ctx.data.ctrl[:] = ctx.home_ctrl
        ctx.data.xfrc_applied[:] = 0.0
        active = cfg["force_start"] <= ctx.data.time < cfg["force_start"] + cfg["force_duration"]
        external = axis * float(cfg["force_newtons"]) if active else np.zeros(3)
        ctx.data.xfrc_applied[body_id, :3] += external
        fsnapshot = coupling.compute_and_apply()
        if np.max(np.abs(fsnapshot.tensions)) > maximum_tension:
            maximum_tension = float(np.max(np.abs(fsnapshot.tensions)))
            maximum_snapshot = fsnapshot
        mujoco.mj_step(ctx.model, ctx.data)
        robot_recorder.sample(ctx.home_ctrl, external)
        fascia_recorder.sample(float(ctx.data.time), fsnapshot)

    output.mkdir(parents=True, exist_ok=True)
    arrays = robot_recorder.arrays()
    write_csv(output / "telemetry.csv", arrays, ctx)
    robot_metrics = perturbation_metrics(arrays, cfg)
    fascia_summary = fascia_recorder.write(output, {
        "experiment": "perturbation_fascia",
        "fascia_enabled": True,
        "mass_model": fconfig["mass_model"],
        "pretension": fconfig["pretension"],
        "performance_claim": False,
    })
    plot_network(output / "fascia_diagnostic.png", coupling.network.topology, initial_positions, maximum_snapshot)
    summary = {"robot_metrics_for_diagnostics_only": robot_metrics, "fascia_metrics": fascia_summary}
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_perturbation_fascia(), indent=2))
