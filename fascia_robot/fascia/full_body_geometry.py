from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import mujoco
import numpy as np

from fascia_robot.model import ModelContext

from .attachments import AttachmentZone, ZONE_BY_KEY


@dataclass(frozen=True)
class FullBodyNode:
    id: int
    region: str
    component: str
    ring: int
    angular_index: int
    position: np.ndarray
    attachment: AttachmentZone | None = None
    layer: str = "superficial"


@dataclass(frozen=True)
class FullBodyEdge:
    id: int
    node_a: int
    node_b: int
    edge_class: str
    region: str
    evidence_level: str
    layer: str = "superficial"


@dataclass(frozen=True)
class FullBodyTopology:
    nodes: tuple[FullBodyNode, ...]
    edges: tuple[FullBodyEdge, ...]
    topology_hash: str
    component_rings: dict[str, tuple[tuple[int, ...], ...]]


class _Builder:
    def __init__(self, evidence_level: str):
        self.nodes: list[FullBodyNode] = []
        self.edge_records: list[tuple[int, int, str, str]] = []
        self.component_rings: dict[str, tuple[tuple[int, ...], ...]] = {}
        self.evidence_level = evidence_level

    def add_component(self, name: str, regions: list[str], ring_points: list[np.ndarray]) -> None:
        rings = []
        for ring_index, points in enumerate(ring_points):
            ids = []
            for angular_index, point in enumerate(points):
                nid = len(self.nodes); ids.append(nid)
                self.nodes.append(FullBodyNode(nid, regions[ring_index], name, ring_index, angular_index, point.copy()))
            rings.append(tuple(ids))
            for j in range(len(ids)):
                self.edge_records.append((ids[j], ids[(j + 1) % len(ids)], "superficial_circumferential", regions[ring_index]))
        for r in range(len(rings) - 1):
            if len(rings[r]) != len(rings[r + 1]):
                raise ValueError("Adjacent rings require equal resolution")
            count = len(rings[r])
            for j in range(count):
                self.edge_records.append((rings[r][j], rings[r + 1][j], "superficial_longitudinal", regions[r]))
                diagonal_target = (j + 1) % count if (r + j) % 2 == 0 else (j - 1) % count
                self.edge_records.append((rings[r][j], rings[r + 1][diagonal_target], "superficial_diagonal", regions[r]))
        self.component_rings[name] = tuple(rings)

    def bridge(self, component_a: str, ring_a: int, component_b: str, ring_b: int, count: int, region: str) -> None:
        ids_a = self.component_rings[component_a][ring_a]
        ids_b = self.component_rings[component_b][ring_b]
        candidates = sorted(
            (float(np.linalg.norm(self.nodes[a].position - self.nodes[b].position)), a, b)
            for a in ids_a for b in ids_b
        )
        used_a, used_b, selected = set(), set(), []
        for _, a, b in candidates:
            if a not in used_a and b not in used_b:
                selected.append((a, b)); used_a.add(a); used_b.add(b)
                if len(selected) == count: break
        for a, b in selected:
            self.edge_records.append((a, b, "superficial_interface", region))

    def attach(self, component: str, ring: int, indices: list[int], zone: AttachmentZone) -> None:
        ring_ids = self.component_rings[component][ring]
        for index in indices:
            nid = ring_ids[index % len(ring_ids)]
            self.nodes[nid] = replace(self.nodes[nid], attachment=zone)

    def finish(self) -> FullBodyTopology:
        unique = []
        seen = set()
        for a, b, edge_class, region in self.edge_records:
            key = tuple(sorted((a, b)))
            if a == b or key in seen: continue
            seen.add(key); unique.append((a, b, edge_class, region))
        edges = tuple(FullBodyEdge(i, a, b, cls, region, self.evidence_level) for i, (a, b, cls, region) in enumerate(unique))
        serial = {
            "nodes": [(n.region, n.component, n.ring, n.angular_index, n.attachment.name if n.attachment else None, n.layer) for n in self.nodes],
            "edges": [(e.node_a, e.node_b, e.edge_class, e.region, e.layer) for e in edges],
        }
        digest = hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()
        return FullBodyTopology(tuple(self.nodes), edges, digest, self.component_rings)


def _body_position(context: ModelContext, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0: raise ValueError(f"Missing G1 body: {name}")
    return context.data.xpos[body_id].copy()


def _sleeve_rings(centers: list[np.ndarray], radii: list[float], count: int) -> list[np.ndarray]:
    rings = []
    for i, (center, radius) in enumerate(zip(centers, radii)):
        tangent = centers[min(i + 1, len(centers) - 1)] - centers[max(i - 1, 0)]
        tangent /= np.linalg.norm(tangent)
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(tangent @ reference)) > 0.9: reference = np.array([1.0, 0.0, 0.0])
        axis_a = np.cross(tangent, reference); axis_a /= np.linalg.norm(axis_a)
        axis_b = np.cross(tangent, axis_a); axis_b /= np.linalg.norm(axis_b)
        rings.append(np.asarray([center + radius * (np.cos(2*np.pi*j/count)*axis_a + np.sin(2*np.pi*j/count)*axis_b) for j in range(count)]))
    return rings


def _interpolate_chain(points: list[np.ndarray], fractions: list[tuple[int, int, float]]) -> list[np.ndarray]:
    return [(1.0 - alpha) * points[a] + alpha * points[b] for a, b, alpha in fractions]


def build_full_body_superficial_mesh(context: ModelContext, config: dict) -> FullBodyTopology:
    mesh = config["superficial_mesh"]; builder = _Builder(mesh["evidence_level"])
    torso_center = _body_position(context, "torso_link")
    tc = mesh["torso"]; torso_points = []
    for z, rx, ry in zip(tc["local_z_m"], tc["radius_x_m"], tc["radius_y_m"]):
        torso_points.append(np.asarray([torso_center + [rx*np.cos(2*np.pi*j/tc["nodes_per_ring"]), ry*np.sin(2*np.pi*j/tc["nodes_per_ring"]), z] for j in range(tc["nodes_per_ring"])]))
    builder.add_component("torso", ["pelvis", "lumbar", "abdomen", "thorax", "neck"] , torso_points)

    for side in ("left", "right"):
        shoulder = _body_position(context, f"{side}_shoulder_pitch_link")
        elbow = _body_position(context, f"{side}_elbow_link")
        wrist = _body_position(context, f"{side}_wrist_roll_link")
        arm_centers = _interpolate_chain([shoulder, elbow, wrist], [(0,0,0), (0,1,.5), (1,1,0), (1,2,.5), (2,2,0)])
        builder.add_component(f"{side}_arm", [f"{side}_shoulder", f"{side}_upper_arm", f"{side}_elbow", f"{side}_forearm", f"{side}_wrist"], _sleeve_rings(arm_centers, mesh["arms"]["radius_m"], mesh["arms"]["nodes_per_ring"]))

        hip = _body_position(context, f"{side}_hip_pitch_link")
        knee = _body_position(context, f"{side}_knee_link")
        ankle = _body_position(context, f"{side}_ankle_pitch_link")
        leg_centers = _interpolate_chain([hip, knee, ankle], [(0,0,0), (0,1,.5), (1,1,0), (1,2,.33), (1,2,.70), (2,2,0)])
        builder.add_component(f"{side}_leg", [f"{side}_proximal_thigh", f"{side}_thigh", f"{side}_knee", f"{side}_upper_leg", f"{side}_lower_leg", f"{side}_ankle"], _sleeve_rings(leg_centers, mesh["legs"]["radius_m"], mesh["legs"]["nodes_per_ring"]))

        fc = mesh["feet"]; foot_points = []
        for xoff, ry, rz in zip(fc["longitudinal_offsets_m"], fc["radius_y_m"], fc["radius_z_m"]):
            center = ankle + np.array([xoff, 0.0, 0.0])
            foot_points.append(np.asarray([center + [0.0, ry*np.cos(2*np.pi*j/fc["nodes_per_ring"]), rz*np.sin(2*np.pi*j/fc["nodes_per_ring"])] for j in range(fc["nodes_per_ring"])]))
        builder.add_component(f"{side}_foot", [f"{side}_heel", f"{side}_midfoot", f"{side}_forefoot"], foot_points)

    bridges = int(mesh["connectivity"]["bridge_edges_per_interface"])
    for side in ("left", "right"):
        builder.bridge("torso", 4, f"{side}_arm", 0, bridges, f"{side}_shoulder_interface")
        builder.bridge("torso", 0, f"{side}_leg", 0, bridges, f"{side}_hip_interface")
        builder.bridge(f"{side}_leg", 5, f"{side}_foot", 0, bridges, f"{side}_ankle_interface")

    cardinals = config["attachments"]["torso_cardinal_indices"]
    selected = config["attachments"]["selected_angular_indices"]
    builder.attach("torso", 0, cardinals, ZONE_BY_KEY["pelvis"])
    builder.attach("torso", 2, cardinals, ZONE_BY_KEY["torso"])
    builder.attach("torso", 4, cardinals, ZONE_BY_KEY["torso"])
    for side in ("left", "right"):
        builder.attach(f"{side}_arm", 0, selected, ZONE_BY_KEY[f"{side}_shoulder"])
        builder.attach(f"{side}_arm", 2, selected, ZONE_BY_KEY[f"{side}_elbow"])
        builder.attach(f"{side}_arm", 4, selected, ZONE_BY_KEY[f"{side}_wrist"])
        builder.attach(f"{side}_leg", 0, selected, ZONE_BY_KEY[f"{side}_hip"])
        builder.attach(f"{side}_leg", 2, selected, ZONE_BY_KEY[f"{side}_knee"])
        builder.attach(f"{side}_leg", 5, selected, ZONE_BY_KEY[f"{side}_ankle"])
        builder.attach(f"{side}_foot", 0, selected, ZONE_BY_KEY[f"{side}_ankle"])
        builder.attach(f"{side}_foot", 2, selected, ZONE_BY_KEY[f"{side}_ankle"])
    return builder.finish()
