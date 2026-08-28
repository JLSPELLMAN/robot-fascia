from __future__ import annotations

from collections import Counter

import yaml

from fascia_robot.fascia.deep_regional_geometry import add_deep_regional_sleeves
from fascia_robot.fascia.full_body_geometry import build_full_body_superficial_mesh
from fascia_robot.fascia.reinforced_pathways import add_reinforced_pathways
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


def _build():
    config = yaml.safe_load((PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml").read_text())
    context = load_model(PROJECT_ROOT / config["model_path"], config["keyframe"], .002)
    phase_a = build_full_body_superficial_mesh(context, config)
    phase_b = add_deep_regional_sleeves(phase_a, context, config)
    return phase_b, add_reinforced_pathways(phase_b, config)


def test_phase_c_adds_exactly_four_reinforcement_families_without_nodes():
    phase_b, phase_c = _build()
    assert len(phase_c.nodes) == len(phase_b.nodes) == 412
    assert len(phase_c.edges) == 1102
    reinforced = [edge for edge in phase_c.edges if edge.layer == "reinforced"]
    assert len(reinforced) == 48
    assert Counter(edge.edge_class for edge in reinforced) == {
        "reinforced_posterior_cross_body": 12,
        "reinforced_lateral_leg": 8,
        "reinforced_calf_achilles_plantar": 12,
        "reinforced_upper_limb_continuity": 16,
    }


def test_phase_c_preserves_phase_b_and_has_evidence_labels():
    phase_b, phase_c = _build()
    assert phase_c.nodes == phase_b.nodes
    assert phase_c.edges[:len(phase_b.edges)] == phase_b.edges
    reinforced = phase_c.edges[len(phase_b.edges):]
    assert all(edge.evidence_level in {"strong", "moderate", "exploratory"} for edge in reinforced)
    assert {edge.evidence_level for edge in reinforced} == {"strong", "moderate"}


def test_reinforced_families_span_the_required_regions():
    _, topology = _build()
    by_class = {}
    for edge in topology.edges:
        if edge.layer == "reinforced":
            by_class.setdefault(edge.edge_class, set()).update((topology.nodes[edge.node_a].component, topology.nodes[edge.node_b].component))
    assert {"deep_left_arm", "deep_right_arm", "deep_trunk", "deep_pelvis"} <= by_class["reinforced_posterior_cross_body"]
    assert {"deep_pelvis", "deep_left_thigh", "deep_right_thigh"} <= by_class["reinforced_lateral_leg"]
    assert {"deep_left_crural", "deep_right_crural", "deep_left_plantar", "deep_right_plantar"} <= by_class["reinforced_calf_achilles_plantar"]
    assert {"deep_trunk", "deep_left_arm", "deep_right_arm"} <= by_class["reinforced_upper_limb_continuity"]


def test_phase_c_is_deterministic_and_geometry_only():
    _, first = _build(); _, second = _build()
    assert first.topology_hash == second.topology_hash
    assert all(not hasattr(edge, "stiffness") for edge in first.edges if edge.layer == "reinforced")
