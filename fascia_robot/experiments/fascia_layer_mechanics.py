from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from fascia_robot.fascia.deep_regional_geometry import add_deep_regional_sleeves
from fascia_robot.fascia.deep_attachments import add_selected_deep_attachments
from fascia_robot.fascia.full_body_geometry import build_full_body_superficial_mesh
from fascia_robot.fascia.material_models import AnatomicalMechanics, assign_axial_material, nonlinear_elastic_force
from fascia_robot.fascia.reinforced_pathways import add_reinforced_pathways
from fascia_robot.fascia.shear_interfaces import add_shear_interfaces
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


def run_phase_d(output: Path | None = None) -> dict:
    output = output or PROJECT_ROOT / "results/anatomical_v1/phase_d"; output.mkdir(parents=True, exist_ok=True)
    geometry = yaml.safe_load((PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml").read_text())
    materials = yaml.safe_load((PROJECT_ROOT / "configs/fascia_materials.yaml").read_text())
    context = load_model(PROJECT_ROOT / geometry["model_path"], geometry["keyframe"], .002)
    topology = add_shear_interfaces(add_selected_deep_attachments(add_reinforced_pathways(add_deep_regional_sleeves(build_full_body_superficial_mesh(context, geometry), context, geometry), geometry)), materials)
    mechanics = AnatomicalMechanics(topology, materials); zero = mechanics.evaluate(mechanics.initial)
    strains = np.linspace(-.02, .12, 300)
    representative = {}
    for key, edge in {
        "superficial": next(e for e in topology.edges if e.layer == "superficial"),
        "deep_trunk": next(e for e in topology.edges if e.layer == "deep" and "trunk" in topology.nodes[e.node_a].component),
        "lateral_leg": next(e for e in topology.edges if e.edge_class == "reinforced_lateral_leg"),
        "plantar_chain": next(e for e in topology.edges if e.edge_class == "reinforced_calf_achilles_plantar"),
    }.items():
        representative[key] = assign_axial_material(edge, materials)
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, material in representative.items():
        ax.plot(strains*100, [nonlinear_elastic_force(float(s), material) for s in strains], label=name.replace("_", " "))
    ax.axvline(0, color="black", linewidth=.8); ax.set_xlabel("strain (%)"); ax.set_ylabel("elastic tension (N)")
    ax.set_title("Phase D tension-only nonlinear material laws"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "material_laws.png", dpi=170); plt.close(fig)
    summary = {
        "architecture": "Anatomical Exofascia v1", "phase": "D — layer mechanics",
        "optimization_performed": False, "performance_testing_performed": False,
        "node_count": len(topology.nodes), "edge_count": len(topology.edges),
        "axial_edge_count": sum(e.layer != "interface" for e in topology.edges),
        "shear_interface_count": sum(e.layer == "interface" for e in topology.edges),
        "pretension": materials["pretension"], "zero_state_max_force_n": float(np.max(np.abs(zero.nodal_forces))),
        "zero_state_energy_j": zero.stored_energy, "net_internal_force_n": zero.net_internal_force.tolist(),
        "net_internal_torque_nm": zero.net_internal_torque.tolist(), "topology_hash": topology.topology_hash,
        "material_model": "passive tension-only nonlinear toe-region elasticity plus projected Kelvin-Voigt damping",
        "interface_model": "weak passive central-force layer-pair spring-damper permitting relative glide",
        "deferred_until_phase_e": ["shoulder propagation", "pelvic rotation", "ankle/foot propagation", "torso twist", "distant-region force validation"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


if __name__ == "__main__": print(json.dumps(run_phase_d(), indent=2))
