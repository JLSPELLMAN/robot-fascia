from __future__ import annotations

import json
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from fascia_robot.fascia.deep_regional_geometry import add_deep_regional_sleeves
from fascia_robot.fascia.deep_attachments import add_selected_deep_attachments
from fascia_robot.fascia.full_body_geometry import build_full_body_superficial_mesh
from fascia_robot.fascia.material_models import AnatomicalMechanics
from fascia_robot.fascia.quasistatic import QuasiStaticAnatomicalSolver
from fascia_robot.fascia.reinforced_pathways import add_reinforced_pathways
from fascia_robot.fascia.shear_interfaces import add_shear_interfaces
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


def build_phase_e():
    geometry = yaml.safe_load((PROJECT_ROOT/"configs/fascia_anatomical_v1.yaml").read_text())
    materials = yaml.safe_load((PROJECT_ROOT/"configs/fascia_materials.yaml").read_text())
    context = load_model(PROJECT_ROOT/geometry["model_path"], geometry["keyframe"], .002)
    topology = add_shear_interfaces(add_selected_deep_attachments(add_reinforced_pathways(add_deep_regional_sleeves(build_full_body_superficial_mesh(context, geometry), context, geometry), geometry)), materials)
    return topology, materials, AnatomicalMechanics(topology, materials)


def _attachments(topology, positions):
    return {n.id: positions[n.id].copy() for n in topology.nodes if n.attachment is not None}


def _rotate(points: dict[int, np.ndarray], ids: list[int], center: np.ndarray, axis: np.ndarray, angle: float) -> None:
    axis = axis/np.linalg.norm(axis); c=np.cos(angle); s=np.sin(angle)
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    rotation=c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*cross
    for nid in ids: points[nid] = center + rotation@(points[nid]-center)


def _region_metrics(topology, snapshot, reference, threshold: float) -> dict[str, dict[str, float]]:
    result = {}
    for region in sorted({n.region for n in topology.nodes}):
        node_ids = [n.id for n in topology.nodes if n.region == region]
        node_set = set(node_ids)
        edge_ids = [e.id for e in topology.edges if e.region == region or e.node_a in node_set or e.node_b in node_set]
        axial = [i for i in edge_ids if topology.edges[i].layer != "interface"]
        axial_deltas = [abs(snapshot.edge_states[i].tension-reference.edge_states[i].tension) for i in axial]
        deltas = [abs(snapshot.edge_states[i].tension-reference.edge_states[i].tension) for i in edge_ids]
        result[region] = {
            "max_nodal_force_n": float(max(np.linalg.norm(snapshot.nodal_forces[i]) for i in node_ids)),
            "max_tension_n": float(max((abs(snapshot.edge_states[i].tension) for i in axial), default=0.0)),
            "stored_energy_j": float(sum(snapshot.edge_states[i].stored_energy for i in edge_ids)),
            "max_delta_tension_n": float(max(deltas, default=0.0)),
            "engaged_edge_count": int(sum(delta > threshold for delta in deltas)),
            "engaged_axial_edge_count": int(sum(delta > threshold for delta in axial_deltas)),
        }
    return result


def _cases(topology, materials, initial):
    cfg=materials["phase_e_displacements"]; cases={}
    attached=[n for n in topology.nodes if n.attachment]
    # 16 — left shoulder lateral displacement.
    boundary=_attachments(topology,initial); ids=[n.id for n in attached if n.attachment.name=="left_shoulder_frame"]
    for nid in ids: boundary[nid][1]+=float(cfg["shoulder_lateral_m"])
    cases["left_shoulder_displacement"]=(boundary,ids)
    # 17 — pelvis axial rotation.
    boundary=_attachments(topology,initial); ids=[n.id for n in attached if n.attachment.name=="pelvic_frame"]
    center=np.mean(initial[ids],axis=0); _rotate(boundary,ids,center,np.array([0,0,1.]),np.deg2rad(cfg["pelvic_rotation_deg"]))
    cases["pelvic_rotation"]=(boundary,ids)
    # 18 — left ankle/foot dorsiflexion about the lateral axis.
    boundary=_attachments(topology,initial)
    ids=[n.id for n in attached if n.component in {"left_foot","deep_left_plantar"}]
    ankle_ids=[n.id for n in attached if (n.component=="left_leg" and n.ring==5) or (n.component=="deep_left_crural" and n.ring==2)]
    center=np.mean(initial[ankle_ids],axis=0); _rotate(boundary,ids,center,np.array([0,1.,0]),-np.deg2rad(cfg["ankle_dorsiflexion_deg"]))
    heel_ids=[n.id for n in attached if n.component in {"left_foot","deep_left_plantar"} and n.ring==0]
    for nid in heel_ids: boundary[nid][2]-=float(cfg["heel_lever_displacement_m"])
    cases["heel_ankle_dorsiflexion"]=(boundary,ids)
    # Required diagnostic arm-elevation chain.
    boundary=_attachments(topology,initial); ids=[n.id for n in attached if n.attachment.name in {"left_shoulder_frame","left_forearm_origin","left_distal_forearm"}]
    shoulder=np.mean(initial[[n.id for n in attached if n.attachment.name=="left_shoulder_frame"]],axis=0)
    _rotate(boundary,ids,shoulder,np.array([1.,0,0]),np.deg2rad(cfg["arm_elevation_deg"]))
    cases["arm_elevation"]=(boundary,ids)
    # 19 — opposing upper/lower torso twist.
    boundary=_attachments(topology,initial); ids=[n.id for n in attached if n.attachment.name=="thoracic_frame"]
    center=np.mean(initial[ids],axis=0); _rotate(boundary,ids,center,np.array([0,0,1.]),np.deg2rad(cfg["torso_twist_deg"]))
    cases["torso_twist"]=(boundary,ids)
    return cases


EXPECTED = {
    "left_shoulder_displacement": ["deep_thoracic", "deep_pelvic_fascia", "left_proximal_thigh", "right_proximal_thigh"],
    "pelvic_rotation": ["deep_thoracic", "deep_right_shoulder_girdle", "deep_left_fascia_lata", "deep_right_fascia_lata"],
    "heel_ankle_dorsiflexion": ["left_plantar", "deep_left_crural", "deep_left_distal_crural"],
    "arm_elevation": ["deep_left_forearm", "deep_left_upper_arm", "deep_left_shoulder_girdle", "deep_scapular_chest", "neck"],
    "torso_twist": ["deep_thoracic", "deep_lumbar", "deep_pelvic_fascia"],
}


def run_phase_e(output: Path | None=None) -> dict:
    output=output or PROJECT_ROOT/"results/anatomical_v1/phase_e"; output.mkdir(parents=True,exist_ok=True)
    topology,materials,_=build_phase_e(); results={}
    phase_e_materials=copy.deepcopy(materials)
    phase_e_materials["pretension"]=float(materials["phase_e_validation"]["assembly_pretension"])
    phase_e_materials["pretension_by_layer"]={
        "superficial":phase_e_materials["pretension"], "deep":phase_e_materials["pretension"],
        "reinforced":float(materials["phase_e_validation"]["reinforced_pathway_pretension"]),
    }
    mechanics=AnatomicalMechanics(topology,phase_e_materials); initial=mechanics.initial
    reference_solver=QuasiStaticAnatomicalSolver(mechanics,phase_e_materials)
    reference=reference_solver.solve(_attachments(topology,initial))
    for _ in range(int(materials["phase_e_validation"]["maximum_solver_restarts"])):
        if reference.success: break
        reference=reference_solver.solve(_attachments(topology,initial),initial=reference.positions)
    if not reference.success: raise RuntimeError(f"Phase E reference equilibrium failed: {reference.message}, residual={reference.free_force_residual_n}")
    threshold=float(materials["phase_e_validation"]["engagement_delta_tension_n"])
    for name,(boundary,moved) in _cases(topology,materials,initial).items():
        solver=QuasiStaticAnatomicalSolver(mechanics,phase_e_materials); solved=solver.solve(boundary,initial=reference.positions)
        for _ in range(int(materials["phase_e_validation"]["maximum_solver_restarts"])):
            if solved.success: break
            solved=solver.solve(boundary,initial=solved.positions)
        regional=_region_metrics(topology,solved.snapshot,reference.snapshot,threshold)
        expected={region: regional[region] for region in EXPECTED[name]}
        results[name]={"success":solved.success,"iterations":solved.iterations,"free_force_residual_n":solved.free_force_residual_n,
                       "moved_attachment_nodes":moved,"stored_energy_j":solved.snapshot.stored_energy,
                       "net_internal_force_n":solved.snapshot.net_internal_force.tolist(),"net_internal_torque_nm":solved.snapshot.net_internal_torque.tolist(),
                       "expected_region_metrics":expected,
                       "all_expected_regions_engaged":all(v["engaged_edge_count"]>0 for v in expected.values())}
        (output/f"{name}.json").write_text(json.dumps({**results[name],"regional_metrics":regional},indent=2)+"\n")
    labels=list(results); engagement=[sum(v["engaged_edge_count"] for v in results[n]["expected_region_metrics"].values()) for n in labels]
    peak_delta=[max(v["max_delta_tension_n"] for v in results[n]["expected_region_metrics"].values()) for n in labels]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8),sharex=True); x=np.arange(len(labels))
    ax1.bar(x,engagement,color="#2a9d8f"); ax1.set_ylabel("expected-region\nedges above threshold")
    ax2.bar(x,np.asarray(peak_delta)*1000,color="#e76f51"); ax2.axhline(threshold*1000,color="black",linestyle="--",label="acceptance threshold")
    ax2.set_ylabel("largest expected-region\nΔ tension (mN)"); ax2.set_xticks(x,[s.replace("_","\n") for s in labels]); ax2.legend()
    fig.suptitle("Phase E quasi-static propagation diagnostics"); fig.tight_layout(); fig.savefig(output/"propagation_summary.png",dpi=170); plt.close(fig)
    summary={"architecture":"Anatomical Exofascia v1","phase":"E — propagation validation","performance_testing_performed":False,"optimization_performed":False,
             "case_count":len(results),"all_solvers_converged":all(v["success"] for v in results.values()),
             "all_expected_regions_engaged":all(v["all_expected_regions_engaged"] for v in results.values()),"cases":results,"topology_hash":topology.topology_hash,
             "assembly_pretension":phase_e_materials["pretension"],"engagement_delta_tension_n":threshold,
             "reinforced_pathway_pretension":phase_e_materials["pretension_by_layer"]["reinforced"],
             "interpretation":"These are prescribed quasi-static network propagation diagnostics, not humanoid performance results.",
             "validation_status":"passed" if all(v["all_expected_regions_engaged"] for v in results.values()) else "failed_expected_distant_region_engagement"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); return summary


if __name__=="__main__": print(json.dumps(run_phase_e(),indent=2))
