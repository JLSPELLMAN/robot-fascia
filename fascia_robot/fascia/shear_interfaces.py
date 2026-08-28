from __future__ import annotations

import hashlib
import json

import numpy as np

from .full_body_geometry import FullBodyEdge, FullBodyTopology


SUPERFICIAL_COMPONENT = {
    "deep_trunk": "torso", "deep_pelvis": "torso",
    "deep_left_arm": "left_arm", "deep_right_arm": "right_arm",
    "deep_left_thigh": "left_leg", "deep_right_thigh": "right_leg",
    "deep_left_crural": "left_leg", "deep_right_crural": "right_leg",
    "deep_left_plantar": "left_foot", "deep_right_plantar": "right_foot",
}


def add_shear_interfaces(topology: FullBodyTopology, config: dict) -> FullBodyTopology:
    """Connect every deep node to its nearest superficial regional neighbor."""
    maximum = float(config["shear_interface"]["maximum_pair_distance_m"])
    neighbor_count = int(config["shear_interface"]["superficial_neighbors_per_deep_node"])
    edges = list(topology.edges)
    for node in topology.nodes:
        if node.layer != "deep":
            continue
        superficial_component = SUPERFICIAL_COMPONENT[node.component]
        candidates = [other for other in topology.nodes if other.layer == "superficial" and other.component == superficial_component]
        targets = sorted(candidates, key=lambda other: (float(np.linalg.norm(other.position-node.position)),other.id))[:neighbor_count]
        if len(targets) != neighbor_count: raise ValueError(f"Insufficient shear partners for node {node.id}")
        for target in targets:
            distance = float(np.linalg.norm(target.position - node.position))
            if distance > maximum:
                raise ValueError(f"No local shear partner for node {node.id}: {distance:.4f} m")
            edges.append(FullBodyEdge(len(edges), node.id, target.id, "shear_interface", node.region, "moderate", "interface"))
    serial = {
        "nodes": [(n.region, n.component, n.ring, n.angular_index, n.attachment.name if n.attachment else None, n.layer) for n in topology.nodes],
        "edges": [(e.node_a, e.node_b, e.edge_class, e.region, e.evidence_level, e.layer) for e in edges],
    }
    digest = hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()
    return FullBodyTopology(topology.nodes, tuple(edges), digest, topology.component_rings)
