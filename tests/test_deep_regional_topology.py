from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import yaml

from fascia_robot.fascia.deep_regional_geometry import add_deep_regional_sleeves
from fascia_robot.fascia.full_body_geometry import build_full_body_superficial_mesh
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


def _build():
    config = yaml.safe_load((PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml").read_text())
    context = load_model(PROJECT_ROOT / config["model_path"], config["keyframe"], .002)
    superficial = build_full_body_superficial_mesh(context, config)
    return superficial, add_deep_regional_sleeves(superficial, context, config)


def test_phase_b_exact_size_and_preserves_phase_a():
    superficial, combined = _build()
    assert len(superficial.nodes) == 208 and len(superficial.edges) == 560
    assert len(combined.nodes) == 412 and len(combined.edges) == 1054
    assert sum(n.layer == "deep" for n in combined.nodes) == 204
    assert sum(e.layer == "deep" for e in combined.edges) == 494
    assert combined.nodes[:208] == superficial.nodes
    assert combined.edges[:560] == superficial.edges
    assert sum(n.attachment is not None for n in combined.nodes) == 44
    assert all(n.attachment is None for n in combined.nodes if n.layer == "deep")


def test_required_deep_regions_and_components_exist():
    _, topology = _build()
    regions = {n.region for n in topology.nodes if n.layer == "deep"}
    required = {"deep_lumbar", "deep_abdominal", "deep_thoracic", "deep_scapular_chest", "deep_gluteal_pelvis", "deep_pelvic_fascia"}
    for side in ("left", "right"):
        required |= {f"deep_{side}_shoulder_girdle", f"deep_{side}_upper_arm", f"deep_{side}_forearm", f"deep_{side}_fascia_lata", f"deep_{side}_crural", f"{side}_plantar"}
    assert required <= regions
    assert len({n.component for n in topology.nodes if n.layer == "deep"}) == 10


def test_each_deep_regional_structure_is_connected():
    _, topology = _build()
    adjacency = defaultdict(set)
    for edge in topology.edges:
        if edge.layer == "deep":
            adjacency[edge.node_a].add(edge.node_b); adjacency[edge.node_b].add(edge.node_a)
    for component in {n.component for n in topology.nodes if n.layer == "deep"}:
        ids = {n.id for n in topology.nodes if n.component == component and n.layer == "deep"}
        seen = set(); queue = deque([next(iter(ids))])
        while queue:
            node = queue.popleft()
            if node in seen: continue
            seen.add(node); queue.extend(adjacency[node] - seen)
        assert seen == ids


def test_phase_b_is_deterministic_finite_and_has_no_phase_c_pathways():
    _, first = _build(); _, second = _build()
    assert first.topology_hash == second.topology_hash
    assert np.isfinite(np.asarray([n.position for n in first.nodes])).all()
    forbidden = ("reinforced", "cross_body", "achilles", "spiral_pathway", "it_band")
    assert not any(any(token in e.edge_class for token in forbidden) for e in first.edges)
