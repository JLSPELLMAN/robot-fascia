from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np

from fascia_robot.model import ModelContext

from .full_body_geometry import FullBodyEdge, FullBodyNode, FullBodyTopology


class _DeepBuilder:
    def __init__(self, superficial: FullBodyTopology, evidence_level: str):
        self.nodes = list(superficial.nodes)
        self.edges = list(superficial.edges)
        self.component_rings = dict(superficial.component_rings)
        self.evidence_level = evidence_level

    def add_sleeve(self, component: str, regions: list[str], rings: list[np.ndarray]) -> None:
        ring_ids = []
        for ring_index, points in enumerate(rings):
            ids = []
            for angular_index, point in enumerate(points):
                nid = len(self.nodes); ids.append(nid)
                self.nodes.append(FullBodyNode(nid, regions[ring_index], component, ring_index, angular_index, point.copy(), None, "deep"))
            ring_ids.append(tuple(ids))
            for j in range(len(ids)):
                self._edge(ids[j], ids[(j + 1) % len(ids)], "deep_circumferential", regions[ring_index])
        for r in range(len(ring_ids) - 1):
            count = len(ring_ids[r])
            for j in range(count):
                self._edge(ring_ids[r][j], ring_ids[r + 1][j], "deep_longitudinal", regions[r])
                target = (j + 1) % count if (r + j) % 2 == 0 else (j - 1) % count
                self._edge(ring_ids[r][j], ring_ids[r + 1][target], "deep_diagonal", regions[r])
        self.component_rings[component] = tuple(ring_ids)

    def add_plantar_sheet(self, component: str, side: str, points: np.ndarray) -> None:
        rows, cols = points.shape[:2]
        grid = []
        for i in range(rows):
            ids = []
            for j in range(cols):
                nid = len(self.nodes); ids.append(nid)
                self.nodes.append(FullBodyNode(nid, f"{side}_plantar", component, i, j, points[i, j].copy(), None, "deep"))
            grid.append(tuple(ids))
        for i in range(rows):
            for j in range(cols):
                if i + 1 < rows: self._edge(grid[i][j], grid[i + 1][j], "deep_plantar_longitudinal", f"{side}_plantar")
                if j + 1 < cols: self._edge(grid[i][j], grid[i][j + 1], "deep_plantar_transverse", f"{side}_plantar")
                if i + 1 < rows and j + 1 < cols:
                    self._edge(grid[i][j], grid[i + 1][j + 1], "deep_plantar_diagonal", f"{side}_plantar")
                    self._edge(grid[i][j + 1], grid[i + 1][j], "deep_plantar_diagonal", f"{side}_plantar")
        self.component_rings[component] = tuple(grid)

    def _edge(self, a: int, b: int, edge_class: str, region: str) -> None:
        self.edges.append(FullBodyEdge(len(self.edges), a, b, edge_class, region, self.evidence_level, "deep"))

    def finish(self) -> FullBodyTopology:
        serial = {
            "nodes": [(n.region, n.component, n.ring, n.angular_index, n.attachment.name if n.attachment else None, n.layer) for n in self.nodes],
            "edges": [(e.node_a, e.node_b, e.edge_class, e.region, e.layer) for e in self.edges],
        }
        digest = hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()
        return FullBodyTopology(tuple(self.nodes), tuple(self.edges), digest, self.component_rings)


def _body(context: ModelContext, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0: raise ValueError(f"Missing G1 body: {name}")
    return context.data.xpos[body_id].copy()


def _interpolate(a: np.ndarray, b: np.ndarray, fractions: list[float]) -> list[np.ndarray]:
    return [(1.0 - f) * a + f * b for f in fractions]


def _rings(centers: list[np.ndarray], radii: list[float], count: int) -> list[np.ndarray]:
    output = []
    for i, (center, radius) in enumerate(zip(centers, radii)):
        tangent = centers[min(i + 1, len(centers) - 1)] - centers[max(i - 1, 0)]
        tangent /= np.linalg.norm(tangent)
        reference = np.array([0.0, 0.0, 1.0]) if abs(tangent[2]) < .9 else np.array([1.0, 0.0, 0.0])
        a = np.cross(tangent, reference); a /= np.linalg.norm(a)
        b = np.cross(tangent, a); b /= np.linalg.norm(b)
        output.append(np.asarray([center + radius * (np.cos(2*np.pi*j/count)*a + np.sin(2*np.pi*j/count)*b) for j in range(count)]))
    return output


def add_deep_regional_sleeves(superficial: FullBodyTopology, context: ModelContext, config: dict) -> FullBodyTopology:
    """Phase B steps 4–7 only: regional deep geometry without layer mechanics."""
    deep = config["deep_layers"]; builder = _DeepBuilder(superficial, deep["evidence_level"])

    # Step 4 — trunk deep fascia.
    torso = _body(context, "torso_link"); tc = deep["trunk"]
    trunk_rings = [np.asarray([torso + [rx*np.cos(2*np.pi*j/tc["nodes_per_ring"]), ry*np.sin(2*np.pi*j/tc["nodes_per_ring"]), z] for j in range(tc["nodes_per_ring"])]) for z, rx, ry in zip(tc["local_z_m"], tc["radius_x_m"], tc["radius_y_m"])]
    builder.add_sleeve("deep_trunk", ["deep_lumbar", "deep_abdominal", "deep_thoracic", "deep_scapular_chest"], trunk_rings)

    # Step 5 — bilateral arm sleeves.
    for side in ("left", "right"):
        shoulder, elbow, wrist = (_body(context, f"{side}_{name}") for name in ("shoulder_pitch_link", "elbow_link", "wrist_roll_link"))
        centers = [shoulder, *_interpolate(shoulder, elbow, [.55]), elbow, wrist]
        builder.add_sleeve(f"deep_{side}_arm", [f"deep_{side}_shoulder_girdle", f"deep_{side}_upper_arm", f"deep_{side}_elbow", f"deep_{side}_forearm"], _rings(centers, deep["arms"]["radius_m"], deep["arms"]["nodes_per_ring"]))

    # Step 6 — pelvic, thigh, and crural/lower-leg sleeves.
    pelvis = _body(context, "pelvis"); pc = deep["pelvis"]
    pelvis_rings = [np.asarray([pelvis + [rx*np.cos(2*np.pi*j/pc["nodes_per_ring"]), ry*np.sin(2*np.pi*j/pc["nodes_per_ring"]), z] for j in range(pc["nodes_per_ring"])]) for z, rx, ry in zip(pc["local_z_m"], pc["radius_x_m"], pc["radius_y_m"])]
    builder.add_sleeve("deep_pelvis", ["deep_gluteal_pelvis", "deep_pelvic_fascia"], pelvis_rings)
    for side in ("left", "right"):
        hip, knee, ankle = (_body(context, f"{side}_{name}") for name in ("hip_pitch_link", "knee_link", "ankle_pitch_link"))
        thigh_centers = _interpolate(hip, knee, [0.0, .34, .67, 1.0])
        builder.add_sleeve(f"deep_{side}_thigh", [f"deep_{side}_proximal_thigh", f"deep_{side}_fascia_lata", f"deep_{side}_thigh_sleeve", f"deep_{side}_distal_thigh"], _rings(thigh_centers, deep["thighs"]["radius_m"], deep["thighs"]["nodes_per_ring"]))
        lower_centers = _interpolate(knee, ankle, [0.0, .5, 1.0])
        builder.add_sleeve(f"deep_{side}_crural", [f"deep_{side}_proximal_crural", f"deep_{side}_crural", f"deep_{side}_distal_crural"], _rings(lower_centers, deep["lower_legs"]["radius_m"], deep["lower_legs"]["nodes_per_ring"]))

    # Step 7 — bilateral plantar sheets.
    plantar = deep["plantar"]
    for side in ("left", "right"):
        ankle = _body(context, f"{side}_ankle_pitch_link")
        points = np.asarray([[
            ankle + np.array([x, y, plantar["height_m"]])
            for y in plantar["transverse_offsets_m"]
        ] for x in plantar["longitudinal_offsets_m"]])
        builder.add_plantar_sheet(f"deep_{side}_plantar", side, points)
    return builder.finish()

