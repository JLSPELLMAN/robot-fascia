from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class AttachmentSpec:
    body_name: str
    region: str


@dataclass(frozen=True)
class NodeSpec:
    id: int
    ring: int
    angular_index: int
    initial_position: np.ndarray
    attachment: AttachmentSpec | None = None


@dataclass(frozen=True)
class EdgeSpec:
    id: int
    node_a: int
    node_b: int
    edge_class: str
    rest_length: float
    stiffness: float
    damping: float


@dataclass(frozen=True)
class Topology:
    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    topology_hash: str


def node_id(ring: int, angular_index: int, per_ring: int = 8) -> int:
    return ring * per_ring + angular_index % per_ring


def build_torso_topology(config: dict) -> Topology:
    topo = config["topology"]
    rings, count = int(topo["rings"]), int(topo["nodes_per_ring"])
    if (rings, count) != (4, 8):
        raise ValueError("Phase 3 topology is fixed at four rings of eight nodes")

    attachment_map = {
        node_id(3, 2): AttachmentSpec("left_shoulder_pitch_link", "top_left"),
        node_id(3, 6): AttachmentSpec("right_shoulder_pitch_link", "top_right"),
        node_id(3, 0): AttachmentSpec("torso_link", "top_anterior"),
        node_id(3, 4): AttachmentSpec("torso_link", "top_posterior"),
        node_id(0, 2): AttachmentSpec("pelvis", "bottom_left"),
        node_id(0, 6): AttachmentSpec("pelvis", "bottom_right"),
        node_id(0, 0): AttachmentSpec("pelvis", "bottom_anterior"),
        node_id(0, 4): AttachmentSpec("pelvis", "bottom_posterior"),
    }
    nodes: list[NodeSpec] = []
    for r in range(rings):
        for j in range(count):
            angle = 2.0 * np.pi * j / count
            position = np.array([
                topo["ring_radius_x_m"][r] * np.cos(angle),
                topo["ring_radius_y_m"][r] * np.sin(angle),
                topo["ring_z_m"][r],
            ])
            nid = node_id(r, j, count)
            nodes.append(NodeSpec(nid, r, j, position, attachment_map.get(nid)))

    pairs: list[tuple[int, int, str]] = []
    for r in range(rings):
        for j in range(count):
            pairs.append((node_id(r, j), node_id(r, j + 1), "circumferential"))
    for r in range(rings - 1):
        for j in range(count):
            pairs.append((node_id(r, j), node_id(r + 1, j), "longitudinal"))
            if (r + j) % 2 == 0:
                pairs.append((node_id(r, j), node_id(r + 1, j + 1), "diagonal"))
            else:
                pairs.append((node_id(r, j + 1), node_id(r + 1, j), "diagonal"))
    pairs.extend([
        (node_id(3, 2), node_id(0, 6), "cross_body"),
        (node_id(3, 6), node_id(0, 2), "cross_body"),
        (node_id(3, 0), node_id(0, 4), "cross_body"),
        (node_id(3, 4), node_id(0, 0), "cross_body"),
    ])

    positions = np.asarray([node.initial_position for node in nodes])
    edges = []
    for eid, (a, b, edge_class) in enumerate(pairs):
        initial_length = float(np.linalg.norm(positions[b] - positions[a]))
        rest = initial_length * (1.0 - float(config["pretension"]))
        edges.append(EdgeSpec(
            eid, a, b, edge_class, rest,
            float(config["stiffness_n_per_m"][edge_class]),
            float(config["damping_ns_per_m"][edge_class]),
        ))
    serial = [(e.node_a, e.node_b, e.edge_class) for e in edges]
    digest = hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()
    return Topology(tuple(nodes), tuple(edges), digest)
