from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import mujoco
import numpy as np
import yaml

from fascia_robot.fascia.deep_attachments import add_selected_deep_attachments
from fascia_robot.fascia.deep_regional_geometry import add_deep_regional_sleeves
from fascia_robot.fascia.full_body_geometry import FullBodyTopology, build_full_body_superficial_mesh
from fascia_robot.fascia.material_models import assign_axial_material
from fascia_robot.fascia.reinforced_pathways import add_reinforced_pathways
from fascia_robot.fascia.shear_interfaces import add_shear_interfaces
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


OUTPUT = PROJECT_ROOT / "results/anatomical_v1/topology_inspection"

ANATOMICAL_COLORS = {
    "superficial_whole_body": "#8e9aaf",
    "deep_fascia": "#4361ee",
    "thoracolumbar_fascia": "#7b2cbf",
    "shoulder_scapular_fascia": "#ef476f",
    "pectoral_anterior_torso_fascia": "#ff9f1c",
    "brachial_forearm_sleeves": "#06d6a0",
    "pelvic_gluteal_fascia": "#9c6644",
    "fascia_lata_it_band": "#e63946",
    "thigh_crural_fascia": "#2a9d8f",
    "ankle_achilles_pathways": "#118ab2",
    "plantar_fascia": "#588157",
    "sliding_interfaces": "#adb5bd",
}

ROUTING_COLORS = {
    "longitudinal": "#00b4d8",
    "circumferential": "#ffd166",
    "diagonal_cross_body": "#f72585",
    "spiral_helical": "#8338ec",
    "sliding": "#adb5bd",
    "structural_fill": "#6c757d",
}


def build_current_topology():
    geometry = yaml.safe_load((PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml").read_text())
    materials = yaml.safe_load((PROJECT_ROOT / "configs/fascia_materials.yaml").read_text())
    context = load_model(PROJECT_ROOT / geometry["model_path"], geometry["keyframe"], .002)
    topology = build_full_body_superficial_mesh(context, geometry)
    topology = add_deep_regional_sleeves(topology, context, geometry)
    topology = add_reinforced_pathways(topology, geometry)
    topology = add_selected_deep_attachments(topology)
    topology = add_shear_interfaces(topology, materials)
    return context, topology, geometry, materials


def _anatomical_category(edge, topology: FullBodyTopology) -> str:
    text = " ".join((edge.region, edge.edge_class,
                     topology.nodes[edge.node_a].region, topology.nodes[edge.node_b].region)).lower()
    if edge.layer == "interface": return "sliding_interfaces"
    if "plantar" in text: return "plantar_fascia"
    if "achilles" in text or "ankle" in text: return "ankle_achilles_pathways"
    if "fascia_lata" in text or "lateral_leg" in text: return "fascia_lata_it_band"
    if any(t in text for t in ("crural", "thigh", "knee")): return "thigh_crural_fascia"
    if any(t in text for t in ("pelvi", "gluteal", "hip")): return "pelvic_gluteal_fascia"
    if any(t in text for t in ("upper_arm", "forearm", "brachial", "elbow", "wrist", "upper_limb")):
        return "brachial_forearm_sleeves"
    if any(t in text for t in ("scapular", "shoulder", "posterior_cross_body")):
        return "shoulder_scapular_fascia"
    if any(t in text for t in ("abdominal", "chest", "thoracic")):
        return "pectoral_anterior_torso_fascia"
    if "lumbar" in text or "trunk" in text: return "thoracolumbar_fascia"
    if edge.layer == "superficial": return "superficial_whole_body"
    return "deep_fascia"


def _routing_category(edge) -> str:
    cls = edge.edge_class
    if edge.layer == "interface": return "sliding"
    if cls == "superficial_interface": return "structural_fill"
    if "diagonal" in cls and edge.layer != "reinforced": return "structural_fill"
    if "circumferential" in cls or "transverse" in cls: return "circumferential"
    if "longitudinal" in cls: return "longitudinal"
    if "posterior_cross_body" in cls: return "diagonal_cross_body"
    if edge.layer == "reinforced": return "longitudinal"
    return "structural_fill"


def _structural_fill_reason(edge) -> str | None:
    if edge.edge_class == "superficial_interface":
        return "nearest-ring bridge inserted to make adjacent coarse body components one graph"
    if "diagonal" in edge.edge_class and edge.layer != "reinforced":
        return "alternating sleeve/sheet triangulation inserted for mesh connectivity and shear resistance; not an explicit anatomical tract"
    return None


def _orientation(edge) -> str:
    cls = edge.edge_class
    if "circumferential" in cls or "transverse" in cls: return "circumferential/transverse"
    if "longitudinal" in cls: return "longitudinal"
    if "posterior_cross_body" in cls: return "diagonal cross-body"
    if "lateral_leg" in cls: return "longitudinal-lateral"
    if "calf_achilles" in cls: return "posterior longitudinal"
    if "upper_limb" in cls: return "proximodistal"
    if "diagonal" in cls: return "alternating diagonal"
    if edge.layer == "interface": return "radial/nearest-neighbor"
    return "component bridge"


def _region_rationale(region: str) -> str:
    r = region.lower()
    if "plantar" in r: return "Models a load-spreading sheet beneath the foot and continuity with the posterior lower-leg route."
    if "crural" in r or "lower_leg" in r: return "Encircles and spans the lower leg to distribute calf/shank loads toward ankle and knee."
    if "fascia_lata" in r: return "Represents lateral thigh reinforcement between pelvis and lateral knee (IT-band-like engineering abstraction)."
    if "thigh" in r or "knee" in r: return "Sleeve lattice distributes axial and circumferential load across the thigh/knee transition."
    if "pelvi" in r or "gluteal" in r or "hip" in r: return "Provides a pelvic load-transfer hub joining trunk, cross-body, and lower-limb routes."
    if "forearm" in r or "elbow" in r or "upper_arm" in r or "wrist" in r: return "Provides proximodistal arm continuity and distributes loads around the limb circumference."
    if "shoulder" in r or "scapular" in r: return "Routes upper-limb load into trunk and cross-body posterior continuities."
    if "thoracic" in r or "chest" in r: return "Represents thoracic/pectoral-scapular load distribution and upper-limb origins."
    if "abdom" in r: return "Provides anterior trunk continuity between thorax and pelvis."
    if "lumbar" in r: return "Represents thoracolumbar load transfer between shoulder/trunk and opposite pelvis."
    if "neck" in r: return "Completes the superficial torso sleeve superiorly; no dedicated cervical reinforcement is modeled."
    if "ankle" in r or "heel" in r: return "Bridges shank and foot near the modeled Achilles/heel load-transfer zone."
    if "midfoot" in r or "forefoot" in r: return "Extends the superficial foot sleeve anteriorly for foot-wide load distribution."
    return "Generic superficial or deep sleeve segment used for distributed load transfer."


def _material_record(edge, materials: dict) -> dict:
    if edge.layer == "interface":
        c = materials["shear_interface"]
        return {"class": "shear_interface", "stiffness_n_per_m": c["stiffness_n_per_m"],
                "damping_ns_per_m": c["damping_ns_per_m"]}
    m = assign_axial_material(edge, materials)
    return {"class": m.name, "k1_n": m.k1_n, "k2_n": m.k2_n,
            "toe_strain": m.toe_strain, "damping_ns_per_m": m.damping_ns_per_m}


def topology_document(topology: FullBodyTopology, materials: dict) -> dict:
    edge_records = []
    for edge in topology.edges:
        fill = _structural_fill_reason(edge)
        edge_records.append({
            "id": edge.id, "node_a": edge.node_a, "node_b": edge.node_b,
            "region": edge.region, "edge_class": edge.edge_class, "layer": edge.layer,
            "evidence_level": edge.evidence_level, "anatomical_category": _anatomical_category(edge, topology),
            "routing_category": _routing_category(edge), "fiber_orientation": _orientation(edge),
            "material": _material_record(edge, materials), "structural_fill": fill is not None,
            "structural_fill_reason": fill,
        })
    nodes = [{"id": n.id, "region": n.region, "component": n.component, "layer": n.layer,
              "position_m": np.round(n.position, 6).tolist(),
              "attachment": None if n.attachment is None else asdict(n.attachment)} for n in topology.nodes]
    incident = defaultdict(list)
    neighbors = defaultdict(set)
    for rec in edge_records:
        a, b = topology.nodes[rec["node_a"]], topology.nodes[rec["node_b"]]
        incident[a.region].append(rec); incident[b.region].append(rec)
        if a.region != b.region:
            neighbors[a.region].add(b.region); neighbors[b.region].add(a.region)
    regions = []
    for region in sorted({n.region for n in topology.nodes}):
        region_nodes = [n for n in topology.nodes if n.region == region]
        records = incident[region]
        attachments = sorted({n.attachment.name for n in region_nodes if n.attachment})
        regions.append({
            "region_name": region, "node_count": len(region_nodes),
            "incident_edge_count": len({r["id"] for r in records}), "internal_edge_count": sum(
                topology.nodes[r["node_a"]].region == region and topology.nodes[r["node_b"]].region == region for r in records),
            "attachment_points": attachments, "neighboring_fascia_regions": sorted(neighbors[region]),
            "fiber_orientations": sorted({r["fiber_orientation"] for r in records}),
            "material_stiffness_classes": sorted({r["material"]["class"] for r in records}),
            "pathway_types": sorted({r["layer"] for r in records}),
            "anatomical_rationale": _region_rationale(region),
            "structural_fill_edge_count": sum(r["structural_fill"] for r in records),
        })
    return {
        "schema_version": 1, "architecture": "Anatomical Exofascia v1 — current code topology",
        "topology_hash": topology.topology_hash,
        "counts": {"nodes": len(nodes), "edges": len(edge_records),
                   "attachments": sum(n.attachment is not None for n in topology.nodes),
                   "nodes_by_layer": dict(Counter(n.layer for n in topology.nodes)),
                   "edges_by_layer": dict(Counter(e.layer for e in topology.edges)),
                   "edges_by_class": dict(Counter(e.edge_class for e in topology.edges)),
                   "structural_fill_edges": sum(r["structural_fill"] for r in edge_records)},
        "explicit_spiral_helical_pathway_present": False,
        "nodes": nodes, "edges": edge_records, "regions": regions,
        "major_continuities": [
            {"name": "posterior_cross_body", "route": ["left/right shoulder", "deep scapular chest", "thoracolumbar trunk", "opposite pelvis"]},
            {"name": "lateral_leg", "route": ["pelvis", "fascia lata / lateral thigh", "lateral knee"]},
            {"name": "calf_achilles_plantar", "route": ["posterior calf", "Achilles/heel region", "plantar fascia"]},
            {"name": "upper_limb", "route": ["chest/scapula", "shoulder", "brachial sleeve", "forearm"]},
        ],
    }


def _robot_mesh(context, max_faces_per_geom: int = 220) -> list[list[list[float]]]:
    model, data = context.model, context.data
    triangles = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_rgba[gid, 3] <= 0:
            continue
        mid = int(model.geom_dataid[gid]); va = int(model.mesh_vertadr[mid]); vn = int(model.mesh_vertnum[mid])
        fa = int(model.mesh_faceadr[mid]); fn = int(model.mesh_facenum[mid])
        verts = model.mesh_vert[va:va+vn]
        faces = model.mesh_face[fa:fa+fn]
        if fn > max_faces_per_geom:
            faces = faces[np.linspace(0, fn-1, max_faces_per_geom, dtype=int)]
        world = verts @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        triangles.extend(np.round(world[faces], 5).tolist())
    return triangles


def _plot_view(path: Path, document: dict, robot_triangles: list, elev: float, azim: float, title: str,
               legend: bool = False) -> None:
    fig = plt.figure(figsize=(8, 11)); ax = fig.add_subplot(111, projection="3d")
    tri = np.asarray(robot_triangles)
    ax.add_collection3d(PolyCollection([], alpha=0))  # keeps older matplotlib backends stable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(tri, facecolor="#9da4ae", edgecolor="none", alpha=.34))
    pos = np.asarray([n["position_m"] for n in document["nodes"]])
    for e in document["edges"]:
        seg = pos[[e["node_a"], e["node_b"]]]
        color = ROUTING_COLORS[e["routing_category"]]
        width = 1.45 if e["layer"] == "reinforced" else (.35 if e["layer"] == "interface" else .55)
        alpha = .24 if e["structural_fill"] else (.28 if e["layer"] == "interface" else .78)
        ax.plot(*seg.T, color=color, linewidth=width, alpha=alpha)
    attached = [n["id"] for n in document["nodes"] if n["attachment"]]
    ax.scatter(*pos[attached].T, color="#d00000", s=8, alpha=.75)
    if legend:
        for category, color in ROUTING_COLORS.items():
            if category == "spiral_helical":
                continue
            ax.plot([], [], [], color=color, linewidth=2, label=category.replace("_", " "))
        ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.set(xlim=(-.45,.45), ylim=(-.45,.45), zlim=(0,1.85)); ax.set_box_aspect((.9,.9,1.85))
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _interactive_html(document: dict, robot_triangles: list) -> str:
    compact = {"nodes": document["nodes"], "edges": document["edges"], "robot": robot_triangles,
               "anatomical_colors": ANATOMICAL_COLORS, "routing_colors": ROUTING_COLORS}
    payload = json.dumps(compact, separators=(",", ":"))
    anatomical = "".join(f'<label><input type="checkbox" class="anatomy" value="{k}" checked> {k.replace("_"," ")}</label>' for k in ANATOMICAL_COLORS)
    routing = "".join(f'<label><input type="checkbox" class="routing" value="{k}" checked> {k.replace("_"," ")}</label>' for k in ROUTING_COLORS)
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Anatomical Exofascia v1 topology</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>
body{{margin:0;font:14px system-ui;background:#10141b;color:#edf2f4}}header{{padding:12px 18px;background:#171d27}}
.controls{{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px 18px}}fieldset{{border:1px solid #45505f}}
label{{display:inline-block;margin:3px 9px 3px 0}}#plot{{height:82vh}}.note{{color:#bac3ce;font-size:12px}}
</style></head><body><header><strong>Anatomical Exofascia v1 — topology inspector</strong>
<div class="note">Current code topology. Grey robot mesh is visual context. Click edges/nodes to inspect; toggles combine anatomical and routing filters.</div></header>
<div class="controls"><fieldset><legend>Anatomical layer / region</legend>{anatomical}</fieldset>
<fieldset><legend>Fiber routing</legend>{routing}</fieldset></div><div id="plot"></div><script>
const D={payload};
function checked(cls){{return new Set([...document.querySelectorAll('.'+cls+':checked')].map(x=>x.value));}}
function rebuild(){{const aa=checked('anatomy'), rr=checked('routing'), traces=[];
const tx=[],ty=[],tz=[]; for(const t of D.robot){{for(const p of t){{tx.push(p[0]);ty.push(p[1]);tz.push(p[2]);}} tx.push(null);ty.push(null);tz.push(null);}}
traces.push({{type:'scatter3d',mode:'lines',x:tx,y:ty,z:tz,line:{{color:'#717986',width:1}},opacity:.18,name:'G1 body',hoverinfo:'skip'}});
for(const [cat,color] of Object.entries(D.anatomical_colors)){{const x=[],y=[],z=[],text=[]; for(const e of D.edges){{if(e.anatomical_category!==cat||!aa.has(cat)||!rr.has(e.routing_category))continue; const a=D.nodes[e.node_a],b=D.nodes[e.node_b]; x.push(a.position_m[0],b.position_m[0],null);y.push(a.position_m[1],b.position_m[1],null);z.push(a.position_m[2],b.position_m[2],null); const s=e.structural_fill?' [structural_fill]':'';text.push(e.edge_class+s,e.edge_class+s,'');}}
if(x.length)traces.push({{type:'scatter3d',mode:'lines',x,y,z,text,hovertemplate:'%{{text}}<extra>'+cat.replaceAll('_',' ')+'</extra>',line:{{color,width:cat==='sliding_interfaces'?1:3}},name:cat.replaceAll('_',' ')}});}}
const nx=[],ny=[],nz=[],nt=[];for(const n of D.nodes){{nx.push(n.position_m[0]);ny.push(n.position_m[1]);nz.push(n.position_m[2]);nt.push(n.region+'<br>'+n.component+(n.attachment?'<br>attach: '+n.attachment.name:''));}}
traces.push({{type:'scatter3d',mode:'markers',x:nx,y:ny,z:nz,text:nt,hovertemplate:'%{{text}}<extra>node</extra>',marker:{{size:2,color:'#f8f9fa',opacity:.55}},name:'nodes'}});
Plotly.react('plot',traces,{{paper_bgcolor:'#10141b',plot_bgcolor:'#10141b',font:{{color:'#edf2f4'}},margin:{{l:0,r:0,t:0,b:0}},legend:{{x:.01,y:.99}},scene:{{aspectmode:'data',xaxis:{{title:'anterior x'}},yaxis:{{title:'left y'}},zaxis:{{title:'superior z'}}}}}},{{responsive:true}});}}
document.querySelectorAll('input').forEach(x=>x.addEventListener('change',rebuild));rebuild();</script></body></html>'''


def _regions_markdown(document: dict) -> str:
    lines = ["# Anatomical Exofascia v1 region catalog", "",
             "Counts describe the topology generated by the current source/configuration. Edge counts are incident to the region, so an inter-region edge appears in both neighboring rows.", "",
             "| Region | Nodes | Incident edges | Robot attachments | Neighboring regions | Orientations | Material classes | Type | Rationale |", "|---|---:|---:|---|---|---|---|---|---|"]
    for r in document["regions"]:
        vals = [r["region_name"], str(r["node_count"]), str(r["incident_edge_count"]), ", ".join(r["attachment_points"]) or "—",
                ", ".join(r["neighboring_fascia_regions"]) or "—", ", ".join(r["fiber_orientations"]),
                ", ".join(r["material_stiffness_classes"]), ", ".join(r["pathway_types"]), r["anatomical_rationale"]]
        lines.append("| " + " | ".join(v.replace("|", "\\|") for v in vals) + " |")
    return "\n".join(lines) + "\n"


def _review_markdown(document: dict) -> str:
    c = document["counts"]; fill = c["structural_fill_edges"]
    return f"""# Topology review — Anatomical Exofascia v1

## What the current architecture actually is

The current generator produces **{c['nodes']} nodes and {c['edges']} edges**: {c['edges_by_layer'].get('superficial',0)} superficial axial edges, {c['edges_by_layer'].get('deep',0)} deep axial edges, {c['edges_by_layer'].get('reinforced',0)} explicitly reinforced edges, and {c['edges_by_layer'].get('interface',0)} superficial-to-deep sliding links. It contains {c['attachments']} attached nodes after selected deep attachment mirroring and reinforced-terminal anchoring.

The architecture is best described as coarse circumferential/longitudinal sleeve lattices with four explicit reinforced continuity families overlaid. It is not a fiber-by-fiber anatomical reconstruction.

## Match to intended design

- Present explicitly: bilateral posterior shoulder-to-opposite-pelvis routes; pelvis-to-lateral-knee routes; posterior calf-to-plantar routes; chest/scapula-to-forearm routes.
- Present generically: superficial whole-body sleeves, deep regional sleeves, circumferential and longitudinal fiber directions, and weak nearest-neighbor superficial/deep interfaces.
- Not explicit: a distinct continuous thoracolumbar sheet, pectoral sheet, retinaculum, Achilles tendon geometry, or named spiral/helical tract. These concepts are represented only by regions or by segments of generic/reinforced routes.
- The fascia-lata/IT-band analogue is an explicit reinforced route, but it is a one-dimensional track through a sleeve rather than a broad lateral sheet.

## `structural_fill`

**{fill} edges are flagged `structural_fill`.** These are generic alternating diagonal triangulation edges plus nearest-ring bridges between coarse body components. They were generated primarily to stabilize/connect the mesh and are not claimed as named anatomical structures. They are retained unchanged for this review.

The {c['edges_by_layer'].get('interface',0)} sliding links are also algorithmic nearest-neighbor pairs. They have an intended mechanical role (layer coupling with glide), so they are classified as sliding interfaces rather than `structural_fill`, but their exact pairings are not anatomically mapped.

## Provenance discrepancy

The older saved Phase D summary reports 1,306 elements and 204 interfaces. The current configuration requests two superficial neighbors for every one of 204 deep nodes, producing 408 interfaces and 1,510 total edges. This inspection reports current source/configuration behavior; it does not modify mechanics.

## Review verdict

The topology matches the intended design at the level of **body coverage and four major continuity families**, but much of its density comes from generic sleeve meshing. Anatomical specificity is strongest in the 48 reinforced edges and weakest in the {fill} structural-fill edges and exact sliding-interface pairings. Phase E should remain paused until this distinction is accepted or used in a later redesign decision.
"""


def _continuity_diagram(path: Path) -> None:
    routes = [
        ("Shoulder", "Thoracolumbar\nfascia", "Opposite pelvis", "#d00000"),
        ("Pelvis", "Lateral thigh /\nfascia lata", "Knee", "#7209b7"),
        ("Calf", "Achilles / heel\nregion", "Plantar fascia", "#2a9d8f"),
        ("Chest / scapula", "Brachial\nsleeve", "Forearm", "#ff8500"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5)); ax.axis("off")
    for row, (a,b,c,color) in enumerate(routes):
        y = 3-row
        for x,label in zip((0,1.5,3), (a,b,c)):
            ax.text(x,y,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.35",fc="white",ec=color,lw=2))
        ax.annotate("",(1.15,y),(.35,y),arrowprops=dict(arrowstyle="->",lw=2,color=color))
        ax.annotate("",(2.65,y),(1.85,y),arrowprops=dict(arrowstyle="->",lw=2,color=color))
    ax.set_xlim(-.6,3.6); ax.set_ylim(-.6,3.6); ax.set_title("Major intended mechanical continuities")
    fig.tight_layout(); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def run_topology_inspection(output: Path | None = None) -> dict:
    output = output or OUTPUT; output.mkdir(parents=True, exist_ok=True)
    context, topology, _, materials = build_current_topology()
    document = topology_document(topology, materials)
    robot = _robot_mesh(context)
    (output / "fascia_topology.json").write_text(json.dumps(document, indent=2) + "\n")
    (output / "fascia_regions.md").write_text(_regions_markdown(document))
    (output / "TOPOLOGY_REVIEW.md").write_text(_review_markdown(document))
    (output / "fascia_topology_3d.html").write_text(_interactive_html(document, robot))
    # MuJoCo coordinates: +x anterior, +y left. Matplotlib azimuth 0 views the
    # x/z plane; +/-90 views the y/z plane.
    views = {"front": (0,0), "back": (0,180), "left": (0,90), "right": (0,-90)}
    for name, (elev,azim) in views.items():
        _plot_view(output / f"fascia_topology_{name}.png", document, robot, elev, azim, f"Anatomical Exofascia v1 — {name} view")
    _plot_view(output / "fascia_topology.png", document, robot, 12, -60,
               "Anatomical Exofascia v1 — complete architecture", legend=True)
    _continuity_diagram(output / "fascia_continuities.png")
    return document["counts"]


if __name__ == "__main__":
    counts = run_topology_inspection()
    print(f"topology inspection: {counts['nodes']} nodes, {counts['edges']} edges, {counts['structural_fill_edges']} structural_fill")
