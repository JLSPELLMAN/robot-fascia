import yaml

from fascia_robot.fascia.geometry import build_torso_topology
from fascia_robot.runner import PROJECT_ROOT


def config():
    return yaml.safe_load((PROJECT_ROOT / "configs/fascia_phase3.yaml").read_text())


def test_phase3_topology_is_one_connected_32_node_84_edge_graph():
    topology = build_torso_topology(config())
    assert len(topology.nodes) == 32
    assert len(topology.edges) == 84
    assert len([n for n in topology.nodes if n.attachment]) == 8
    assert {e.edge_class for e in topology.edges} == {"circumferential", "longitudinal", "diagonal", "cross_body"}
    adjacency = {node.id: set() for node in topology.nodes}
    for edge in topology.edges:
        adjacency[edge.node_a].add(edge.node_b); adjacency[edge.node_b].add(edge.node_a)
    visited, stack = set(), [0]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node); stack.extend(adjacency[node] - visited)
    assert len(visited) == 32


def test_topology_generation_is_deterministic():
    first = build_torso_topology(config())
    second = build_torso_topology(config())
    assert first.topology_hash == second.topology_hash
