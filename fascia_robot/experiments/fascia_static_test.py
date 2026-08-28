from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fascia_robot.fascia.geometry import build_torso_topology
from fascia_robot.fascia.network import FasciaNetwork
from fascia_robot.fascia.telemetry import FasciaRecorder
from fascia_robot.fascia.visualization import plot_network
from fascia_robot.trajectory import minimum_jerk_phase


ROOT = Path(__file__).resolve().parents[2]


def run_static_test(config_path: Path | None = None, output: Path | None = None, damping: bool = True) -> dict:
    config_path = config_path or ROOT / "configs/fascia_phase3.yaml"
    output = output or ROOT / "results/phase3/static_deformation"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not damping:
        config["damping_ns_per_m"] = {key: 0.0 for key in config["damping_ns_per_m"]}
    topology = build_torso_topology(config)
    samples = int(config["static_test"]["samples"])
    duration = float(config["static_test"]["duration_s"])
    hold = float(config["static_test"]["hold_s"])
    dt = (duration + hold) / (samples - 1)
    network = FasciaNetwork(topology, dt, config["solver"])
    recorder = FasciaRecorder(topology)
    initial = network.positions.copy()
    attachments = {n.id: n.initial_position.copy() for n in topology.nodes if n.attachment}
    network.equilibrate(attachments)
    left = next(n.id for n in topology.nodes if n.attachment and n.attachment.region == "top_left")
    maximum_snapshot = None
    for time in np.linspace(0.0, duration + hold, samples):
        phase = minimum_jerk_phase(float(time), 0.0, duration)
        current = {nid: pos.copy() for nid, pos in attachments.items()}
        current[left] += np.array([0.0, config["static_test"]["displacement_m"] * phase, 0.0])
        snapshot = network.step(current)
        recorder.sample(float(time), snapshot)
        maximum_snapshot = snapshot
    metadata = {
        "experiment": "static_deformation",
        "fascia_enabled": True,
        "mass_model": config["mass_model"],
        "pretension": config["pretension"],
        "controlled_displacement_m": config["static_test"]["displacement_m"],
        "damping_enabled": damping,
    }
    summary = recorder.write(output, metadata)
    plot_network(output / "fascia_diagnostic.png", topology, initial, maximum_snapshot)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_static_test(), indent=2))
