from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np

from .full_body_geometry import FullBodyTopology
from .shear_interfaces import SUPERFICIAL_COMPONENT
from .attachments import ZONE_BY_KEY


def _endpoint_zone(component: str, node_region: str):
    if component == "deep_trunk": return ZONE_BY_KEY["torso"]
    if component == "deep_pelvis": return ZONE_BY_KEY["pelvis"]
    side = "left" if "left" in component else "right"
    if component.endswith("_arm"):
        if "forearm" in node_region: return ZONE_BY_KEY[f"{side}_wrist"]
        return ZONE_BY_KEY[f"{side}_shoulder"]
    if component.endswith("_thigh"): return ZONE_BY_KEY[f"{side}_knee"]
    if component.endswith("_crural"): return ZONE_BY_KEY[f"{side}_ankle"]
    if component.endswith("_plantar"): return ZONE_BY_KEY[f"{side}_ankle"]
    raise ValueError(f"No reinforced endpoint zone for {component}")


def add_selected_deep_attachments(topology: FullBodyTopology) -> FullBodyTopology:
    """Mirror each selected superficial attachment onto one nearby deep node."""
    nodes = list(topology.nodes); used: set[int] = set()
    superficial_attachments = [node for node in nodes if node.layer == "superficial" and node.attachment is not None]
    for superficial in superficial_attachments:
        candidates = [node for node in nodes if node.layer == "deep" and node.id not in used
                      and SUPERFICIAL_COMPONENT[node.component] == superficial.component]
        if not candidates:
            raise ValueError(f"No unused deep attachment candidate for superficial node {superficial.id}")
        deep = min(candidates, key=lambda node: (float(np.linalg.norm(node.position-superficial.position)), node.id))
        nodes[deep.id] = replace(deep, attachment=superficial.attachment); used.add(deep.id)
    # Reinforced routes must be loaded at their terminal nodes; otherwise a
    # tension-only internal route can translate as a slack mechanism.
    groups: dict[tuple[str,str], list] = {}
    for edge in topology.edges:
        if edge.layer == "reinforced": groups.setdefault((edge.edge_class,edge.region),[]).append(edge)
    for edges in groups.values():
        degree: dict[int,int] = {}
        for edge in edges:
            degree[edge.node_a]=degree.get(edge.node_a,0)+1; degree[edge.node_b]=degree.get(edge.node_b,0)+1
        for nid,count in degree.items():
            if count == 1:
                node=nodes[nid]; nodes[nid]=replace(node,attachment=_endpoint_zone(node.component,node.region))
    serial = {
        "nodes": [(n.region,n.component,n.ring,n.angular_index,n.attachment.name if n.attachment else None,n.layer) for n in nodes],
        "edges": [(e.node_a,e.node_b,e.edge_class,e.region,e.evidence_level,e.layer) for e in topology.edges],
    }
    digest=hashlib.sha256(json.dumps(serial,separators=(",",":")).encode()).hexdigest()
    return FullBodyTopology(tuple(nodes),topology.edges,digest,topology.component_rings)
