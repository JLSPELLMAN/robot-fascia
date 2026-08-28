import numpy as np
import yaml

from fascia_robot.fascia.full_body_geometry import build_full_body_superficial_mesh
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


def topology():
    config = yaml.safe_load((PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml").read_text())
    context = load_model(PROJECT_ROOT / config["model_path"], config["keyframe"], 0.002)
    return build_full_body_superficial_mesh(context, config)


def test_phase_a_mesh_covers_required_whole_body_regions():
    mesh = topology(); regions = {node.region for node in mesh.nodes}
    required_tokens = ("neck", "shoulder", "upper_arm", "forearm", "pelvis", "thigh", "knee", "lower_leg", "ankle", "heel", "midfoot", "forefoot")
    for token in required_tokens:
        assert any(token in region for region in regions), token
    assert 150 <= len(mesh.nodes) < 300
    assert len(mesh.edges) >= 500


def test_phase_a_mesh_is_one_connected_graph():
    mesh = topology(); adjacency = {node.id: set() for node in mesh.nodes}
    for edge in mesh.edges:
        adjacency[edge.node_a].add(edge.node_b); adjacency[edge.node_b].add(edge.node_a)
    visited, stack = set(), [0]
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current); stack.extend(adjacency[current] - visited)
    assert len(visited) == len(mesh.nodes)


def test_selected_attachments_are_minor_subset_and_use_valid_bodies():
    mesh = topology(); attached = [node for node in mesh.nodes if node.attachment]
    assert 0 < len(attached) < 0.30 * len(mesh.nodes)
    assert len({node.attachment.name for node in attached}) >= 10
    assert all(node.attachment.body_name for node in attached)


def test_topology_is_deterministic_and_geometrically_finite():
    first, second = topology(), topology()
    assert first.topology_hash == second.topology_hash
    assert np.all(np.isfinite(np.asarray([node.position for node in first.nodes])))
    assert len({(min(edge.node_a, edge.node_b), max(edge.node_a, edge.node_b)) for edge in first.edges}) == len(first.edges)

