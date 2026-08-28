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
from fascia_robot.fascia.full_body_geometry import build_full_body_superficial_mesh
from fascia_robot.fascia.reinforced_pathways import add_reinforced_pathways
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


PATH_COLORS = {
    "reinforced_posterior_cross_body": "#d00000",
    "reinforced_lateral_leg": "#7209b7",
    "reinforced_calf_achilles_plantar": "#2a9d8f",
    "reinforced_upper_limb_continuity": "#ff8500",
}


def _body_position(context, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return context.data.xpos[body_id]


def plot_phase_c(path: Path, topology, context) -> None:
    fig = plt.figure(figsize=(11, 12)); ax = fig.add_subplot(111, projection="3d")
    positions = np.asarray([node.position for node in topology.nodes])
    for edge in topology.edges:
        segment = positions[[edge.node_a, edge.node_b]]
        if edge.layer == "superficial":
            ax.plot(*segment.T, color="#ced4da", alpha=.10, linewidth=.35)
        elif edge.layer == "deep":
            ax.plot(*segment.T, color="#4ea8de", alpha=.16, linewidth=.45)
        else:
            ax.plot(*segment.T, color=PATH_COLORS[edge.edge_class], alpha=.95, linewidth=2.5)
    for edge_class, color in PATH_COLORS.items():
        ax.plot([], [], [], color=color, linewidth=3, label=edge_class.removeprefix("reinforced_").replace("_", " "))
    chains = [
        ["pelvis", "torso_link"],
        ["torso_link", "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_link"],
        ["torso_link", "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link"],
        ["pelvis", "left_hip_pitch_link", "left_knee_link", "left_ankle_pitch_link"],
        ["pelvis", "right_hip_pitch_link", "right_knee_link", "right_ankle_pitch_link"],
    ]
    for chain in chains:
        points = np.asarray([_body_position(context, name) for name in chain]); ax.plot(*points.T, color="black", linewidth=3, alpha=.5)
    reinforced_nodes = sorted({nid for edge in topology.edges if edge.layer == "reinforced" for nid in (edge.node_a, edge.node_b)})
    ax.scatter(*positions[reinforced_nodes].T, color="#212529", s=12, alpha=.85)
    ax.set_xlabel("anterior x (m)"); ax.set_ylabel("left y (m)"); ax.set_zlabel("superior z (m)")
    ax.set_box_aspect((.7, .8, 1.8)); ax.view_init(elev=12, azim=-70); ax.legend(loc="upper left")
    ax.set_title("Anatomical Exofascia v1 — Phase C reinforced pathways")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def run_phase_c(config_path: Path | None = None, output: Path | None = None) -> dict:
    config_path = config_path or PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml"
    output = output or PROJECT_ROOT / "results/anatomical_v1/phase_c"
    config = yaml.safe_load(config_path.read_text())
    context = load_model(PROJECT_ROOT / config["model_path"], config["keyframe"], .002)
    phase_a = build_full_body_superficial_mesh(context, config)
    phase_b = add_deep_regional_sleeves(phase_a, context, config)
    topology = add_reinforced_pathways(phase_b, config)
    output.mkdir(parents=True, exist_ok=True)
    plot_phase_c(output / "reinforced_pathways.png", topology, context)
    reinforced = [edge for edge in topology.edges if edge.layer == "reinforced"]
    summary = {
        "architecture": "Anatomical Exofascia v1",
        "phase": "C — reinforced pathway geometry only",
        "mechanics_enabled": False,
        "optimization_performed": False,
        "node_count": len(topology.nodes),
        "edge_count": len(topology.edges),
        "reinforced_edge_count": len(reinforced),
        "pathway_edge_counts": {name: sum(edge.edge_class == name for edge in reinforced) for name in PATH_COLORS},
        "pathway_evidence": {name: sorted({edge.evidence_level for edge in reinforced if edge.edge_class == name}) for name in PATH_COLORS},
        "topology_hash": topology.topology_hash,
        "scientific_scope": "Routes are biomimetic engineering abstractions; Phase C does not assign stiffness or claim physiological functional significance.",
        "deferred_until_later_phases": ["anisotropic directional properties", "sliding/shear interfaces", "nonlinear tension-only material behavior", "passivity revalidation", "propagation testing", "performance testing", "optimization"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_phase_c(), indent=2))
