from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from fascia_robot.fascia.deep_regional_geometry import add_deep_regional_sleeves
from fascia_robot.fascia.full_body_geometry import FullBodyTopology, build_full_body_superficial_mesh
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


def _body_position(context, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return context.data.xpos[body_id]


def plot_phase_b(path: Path, topology: FullBodyTopology, context, config: dict) -> None:
    fig = plt.figure(figsize=(11, 12))
    ax = fig.add_subplot(111, projection="3d")
    positions = np.asarray([node.position for node in topology.nodes])
    colors = {
        "deep_circumferential": "#0077b6", "deep_longitudinal": "#00b4d8",
        "deep_diagonal": "#ff8500", "deep_plantar_longitudinal": "#2a9d8f",
        "deep_plantar_transverse": "#9b5de5", "deep_plantar_diagonal": "#f4a261",
    }
    for edge in topology.edges:
        segment = positions[[edge.node_a, edge.node_b]]
        if edge.layer == "superficial":
            ax.plot(*segment.T, color="#adb5bd", alpha=.18, linewidth=.45)
        else:
            ax.plot(*segment.T, color=colors[edge.edge_class], alpha=.78, linewidth=1.0)
    deep_ids = [node.id for node in topology.nodes if node.layer == "deep"]
    attachment_ids = [node.id for node in topology.nodes if node.attachment is not None]
    ax.scatter(*positions[deep_ids].T, color="#023047", s=config["visualization"]["node_size"], alpha=.8, label="deep regional node")
    ax.scatter(*positions[attachment_ids].T, color="#d00000", marker="s", s=config["visualization"]["attachment_size"], label="Phase A attachment node")
    chains = [
        ["pelvis", "torso_link"],
        ["torso_link", "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_link"],
        ["torso_link", "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link"],
        ["pelvis", "left_hip_pitch_link", "left_knee_link", "left_ankle_pitch_link"],
        ["pelvis", "right_hip_pitch_link", "right_knee_link", "right_ankle_pitch_link"],
    ]
    for chain in chains:
        points = np.asarray([_body_position(context, name) for name in chain])
        ax.plot(*points.T, color="black", linewidth=3, alpha=.55)
    ax.set_xlabel("anterior x (m)"); ax.set_ylabel("left y (m)"); ax.set_zlabel("superior z (m)")
    ax.set_box_aspect((.7, .8, 1.8)); ax.view_init(elev=12, azim=-70); ax.legend(loc="upper left")
    ax.set_title("Anatomical Exofascia v1 — Phase B deep regional sleeves")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def run_phase_b(config_path: Path | None = None, output: Path | None = None) -> dict:
    config_path = config_path or PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml"
    output = output or PROJECT_ROOT / "results/anatomical_v1/phase_b"
    config = yaml.safe_load(config_path.read_text())
    context = load_model(PROJECT_ROOT / config["model_path"], config["keyframe"], 0.002)
    superficial = build_full_body_superficial_mesh(context, config)
    topology = add_deep_regional_sleeves(superficial, context, config)
    output.mkdir(parents=True, exist_ok=True)
    plot_phase_b(output / "deep_regional_sleeves.png", topology, context, config)
    deep_nodes = [node for node in topology.nodes if node.layer == "deep"]
    deep_edges = [edge for edge in topology.edges if edge.layer == "deep"]
    summary = {
        "architecture": "Anatomical Exofascia v1",
        "phase": "B — deep regional geometry only",
        "mechanics_enabled": False,
        "optimization_performed": False,
        "node_count": len(topology.nodes),
        "superficial_node_count": len(topology.nodes) - len(deep_nodes),
        "deep_node_count": len(deep_nodes),
        "attachment_node_count": sum(node.attachment is not None for node in topology.nodes),
        "edge_count": len(topology.edges),
        "superficial_edge_count": len(topology.edges) - len(deep_edges),
        "deep_edge_count": len(deep_edges),
        "deep_edge_class_counts": {name: sum(edge.edge_class == name for edge in deep_edges) for name in sorted({edge.edge_class for edge in deep_edges})},
        "deep_components": sorted({node.component for node in deep_nodes}),
        "deep_regions": sorted({node.region for node in deep_nodes}),
        "topology_hash": topology.topology_hash,
        "assumption": "Deep regional structures are geometry-only and intentionally uncoupled from the superficial layer until the specified interface phase.",
        "deferred_until_later_phases": ["reinforced anatomical pathways", "layer sliding/shear", "anisotropic materials", "nonlinear material law", "propagation testing", "performance testing", "optimization"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_phase_b(), indent=2))
