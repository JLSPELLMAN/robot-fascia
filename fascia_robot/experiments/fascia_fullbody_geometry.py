from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from fascia_robot.fascia.full_body_geometry import FullBodyTopology, build_full_body_superficial_mesh
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


EDGE_COLORS = {
    "superficial_circumferential": "#219ebc",
    "superficial_longitudinal": "#2a9d8f",
    "superficial_diagonal": "#f4a261",
    "superficial_interface": "#9b5de5",
}


def _body_position(context, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return context.data.xpos[body_id]


def plot_topology(path: Path, topology: FullBodyTopology, context, config: dict) -> None:
    fig = plt.figure(figsize=(11, 12)); ax = fig.add_subplot(111, projection="3d")
    positions = np.asarray([node.position for node in topology.nodes])
    for edge in topology.edges:
        segment = positions[[edge.node_a, edge.node_b]]
        ax.plot(*segment.T, color=EDGE_COLORS[edge.edge_class], alpha=config["visualization"]["edge_alpha"], linewidth=0.7 if edge.edge_class != "superficial_interface" else 1.8)
    internal = np.array([node.id for node in topology.nodes if node.attachment is None])
    attached = np.array([node.id for node in topology.nodes if node.attachment is not None])
    ax.scatter(*positions[internal].T, color="#263238", s=config["visualization"]["node_size"], alpha=.7, label="internal superficial node")
    ax.scatter(*positions[attached].T, color="#d00000", marker="s", s=config["visualization"]["attachment_size"], label="selected attachment node")

    chains = [
        ["pelvis", "torso_link"],
        ["torso_link", "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_link"],
        ["torso_link", "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link"],
        ["pelvis", "left_hip_pitch_link", "left_knee_link", "left_ankle_pitch_link"],
        ["pelvis", "right_hip_pitch_link", "right_knee_link", "right_ankle_pitch_link"],
    ]
    for chain in chains:
        points = np.asarray([_body_position(context, name) for name in chain]); ax.plot(*points.T, color="black", linewidth=3, alpha=.55)
    ax.set_xlabel("anterior x (m)"); ax.set_ylabel("left y (m)"); ax.set_zlabel("superior z (m)")
    ax.set_box_aspect((.7, .8, 1.8)); ax.view_init(elev=12, azim=-70); ax.legend(loc="upper left")
    ax.set_title("Anatomical Exofascia v1 — Phase A coarse superficial whole-body mesh")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def run_phase_a(config_path: Path | None = None, output: Path | None = None) -> dict:
    config_path = config_path or PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml"
    output = output or PROJECT_ROOT / "results/anatomical_v1/phase_a"
    config = yaml.safe_load(config_path.read_text())
    context = load_model(PROJECT_ROOT / config["model_path"], config["keyframe"], 0.002)
    topology = build_full_body_superficial_mesh(context, config)
    output.mkdir(parents=True, exist_ok=True)
    plot_topology(output / "full_body_superficial_mesh.png", topology, context, config)
    regions = sorted({node.region for node in topology.nodes})
    edge_classes = {name: sum(edge.edge_class == name for edge in topology.edges) for name in EDGE_COLORS}
    attachment_zones = sorted({node.attachment.name for node in topology.nodes if node.attachment})
    summary = {
        "architecture": "Anatomical Exofascia v1",
        "phase": "A — full-body geometry only",
        "mechanics_enabled": False,
        "optimization_performed": False,
        "node_count": len(topology.nodes),
        "internal_node_count": sum(node.attachment is None for node in topology.nodes),
        "attachment_node_count": sum(node.attachment is not None for node in topology.nodes),
        "edge_count": len(topology.edges),
        "edge_class_counts": edge_classes,
        "component_count": len(topology.component_rings),
        "components": sorted(topology.component_rings),
        "regions": regions,
        "attachment_zones": attachment_zones,
        "topology_hash": topology.topology_hash,
        "deferred_until_later_phases": ["deep regional sleeves", "reinforced anatomical pathways", "layer sliding/shear", "anisotropic materials", "nonlinear material law", "propagation testing", "performance testing", "optimization"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_phase_a(), indent=2))

