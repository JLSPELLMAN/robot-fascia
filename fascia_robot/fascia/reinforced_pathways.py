from __future__ import annotations

import hashlib
import json

import numpy as np

from .full_body_geometry import FullBodyEdge, FullBodyTopology


class _PathwayBuilder:
    def __init__(self, topology: FullBodyTopology):
        self.topology = topology
        self.edges = list(topology.edges)

    def ring(self, component: str, ring: int) -> tuple[int, ...]:
        return self.topology.component_rings[component][ring]

    def ranked(self, component: str, ring: int, score) -> list[int]:
        return sorted(self.ring(component, ring), key=lambda nid: (score(self.topology.nodes[nid].position), nid))

    def pathway(self, nodes: list[int], edge_class: str, region: str, evidence: str) -> None:
        for a, b in zip(nodes, nodes[1:]):
            if a == b:
                raise ValueError(f"Degenerate reinforced pathway {edge_class}")
            self.edges.append(FullBodyEdge(len(self.edges), a, b, edge_class, region, evidence, "reinforced"))

    def finish(self) -> FullBodyTopology:
        serial = {
            "nodes": [(n.region, n.component, n.ring, n.angular_index, n.attachment.name if n.attachment else None, n.layer) for n in self.topology.nodes],
            "edges": [(e.node_a, e.node_b, e.edge_class, e.region, e.evidence_level, e.layer) for e in self.edges],
        }
        digest = hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()
        return FullBodyTopology(self.topology.nodes, tuple(self.edges), digest, self.topology.component_rings)


def _posterior_lateral_score(side_sign: float, track: int):
    # Posterior (negative x) dominates; the second track lies slightly more lateral.
    return lambda p: float(p[0] - (1.0 + .35 * track) * side_sign * p[1])


def _lateral_score(side_sign: float):
    return lambda p: float(-side_sign * p[1])


def _posterior_score(p: np.ndarray) -> float:
    return float(p[0])


def _nearest_score(target: np.ndarray):
    return lambda p: float(np.linalg.norm(p - target))


def add_reinforced_pathways(topology: FullBodyTopology, config: dict) -> FullBodyTopology:
    """Phase C steps 8–11: routing only, with no material-law assignment."""
    cfg = config["reinforced_pathways"]
    b = _PathwayBuilder(topology)

    # Step 8 — bilateral posterior cross-body shoulder/TLF/opposite-pelvis paths.
    pc = cfg["posterior_cross_body"]
    for source, target in (("right", "left"), ("left", "right")):
        source_sign = 1.0 if source == "left" else -1.0
        target_sign = -source_sign
        for track in range(pc["tracks_per_direction"]):
            route = [
                b.ranked(f"deep_{source}_arm", 0, _posterior_lateral_score(source_sign, track))[track],
                b.ranked("deep_trunk", 3, _posterior_lateral_score(source_sign, track))[track],
                b.ranked("deep_trunk", 1, _posterior_lateral_score(target_sign, track))[track],
                b.ranked("deep_pelvis", 1, _posterior_lateral_score(target_sign, track))[track],
            ]
            b.pathway(route, "reinforced_posterior_cross_body", f"{source}_shoulder_to_{target}_pelvis", pc["evidence_level"])

    # Step 9 — pelvic origin through lateral thigh to lateral knee.
    lc = cfg["lateral_leg"]
    for side in ("left", "right"):
        sign = 1.0 if side == "left" else -1.0
        for track in range(lc["tracks_per_side"]):
            route = [b.ranked("deep_pelvis", 1, _lateral_score(sign))[track]]
            route += [b.ranked(f"deep_{side}_thigh", ring, _lateral_score(sign))[track] for ring in range(4)]
            b.pathway(route, "reinforced_lateral_leg", f"{side}_pelvis_fascia_lata_to_knee", lc["evidence_level"])

    # Step 10 — posterior calf through heel into the longitudinal plantar sheet.
    ac = cfg["calf_achilles_plantar"]
    for side in ("left", "right"):
        for track in range(ac["tracks_per_side"]):
            crural = [b.ranked(f"deep_{side}_crural", ring, _posterior_score)[track] for ring in range(3)]
            heel_target = topology.nodes[crural[-1]].position
            plantar_rings = topology.component_rings[f"deep_{side}_plantar"]
            plantar = [min(row, key=lambda nid: (_nearest_score(heel_target)(topology.nodes[nid].position), nid)) for row in plantar_rings]
            b.pathway(crural + plantar, "reinforced_calf_achilles_plantar", f"{side}_calf_achilles_plantar", ac["evidence_level"])

    # Step 11 — chest/scapular origin through shoulder, arm, and forearm sleeves.
    uc = cfg["upper_limb_continuity"]
    for side in ("left", "right"):
        sign = 1.0 if side == "left" else -1.0
        for track in range(uc["tracks_per_side"]):
            shoulder_node = b.ranked(f"deep_{side}_arm", 0, _posterior_lateral_score(sign, track))[track]
            target = topology.nodes[shoulder_node].position
            trunk_node = min(b.ring("deep_trunk", 3), key=lambda nid: (float(np.linalg.norm(topology.nodes[nid].position - target)), nid))
            route = [trunk_node]
            route += [b.ranked(f"deep_{side}_arm", ring, _posterior_lateral_score(sign, track))[track] for ring in range(4)]
            b.pathway(route, "reinforced_upper_limb_continuity", f"{side}_chest_scapula_to_forearm", uc["evidence_level"])

    return b.finish()
