import numpy as np
import yaml

from fascia_robot.fascia.geometry import build_torso_topology
from fascia_robot.fascia.network import FasciaNetwork
from fascia_robot.runner import PROJECT_ROOT


def make_network(damping=True):
    config = yaml.safe_load((PROJECT_ROOT / "configs/fascia_phase3.yaml").read_text())
    if not damping:
        config["damping_ns_per_m"] = {key: 0.0 for key in config["damping_ns_per_m"]}
    topology = build_torso_topology(config)
    network = FasciaNetwork(topology, 0.01, config["solver"])
    network.equilibrate(attachments(topology))
    return topology, network


def attachments(topology):
    return {n.id: n.initial_position.copy() for n in topology.nodes if n.attachment}


def test_zero_pretension_undeformed_network_has_zero_force():
    config = yaml.safe_load((PROJECT_ROOT / "configs/fascia_phase3.yaml").read_text())
    config["pretension"] = 0.0
    topology = build_torso_topology(config)
    network = FasciaNetwork(topology, 0.01, config["solver"])
    network.equilibrate(attachments(topology))
    snapshot = network.step(attachments(topology))
    assert np.max(np.abs(snapshot.tensions)) < 1e-10
    assert np.max(np.abs(snapshot.attachment_forces)) < 1e-10
    assert np.sum(snapshot.stored_energies) < 1e-12
    assert np.sum(snapshot.cumulative_dissipation) < 1e-12


def test_left_shoulder_stretch_propagates_contralaterally_and_balances():
    topology, network = make_network(damping=False)
    boundary = attachments(topology)
    left = next(n.id for n in topology.nodes if n.attachment and n.attachment.region == "top_left")
    boundary[left] += np.array([0.0, 0.01, 0.0])
    for _ in range(125):
        snapshot = network.step(boundary)
    active = np.abs(snapshot.tensions) > 0.01
    assert np.count_nonzero(active) >= 12
    assert {e.edge_class for e in topology.edges if active[e.id]} == {"circumferential", "longitudinal", "diagonal", "cross_body"}
    regions = [topology.nodes[nid].attachment.region for nid in network.attachment_ids]
    forces = {region: np.linalg.norm(snapshot.attachment_forces[i]) for i, region in enumerate(regions)}
    assert forces["top_right"] > 0.01
    assert forces["bottom_right"] > 0.01
    assert snapshot.free_force_residual < 2e-6
    assert np.linalg.norm(snapshot.net_force) < 5e-6
    assert np.linalg.norm(snapshot.net_torque) < 5e-6


def test_internal_edge_forces_are_equal_and_opposite():
    topology, network = make_network(damping=False)
    boundary = attachments(topology)
    left = next(n.id for n in topology.nodes if n.attachment and n.attachment.region == "top_left")
    boundary[left][1] += 0.005
    snapshot = network.step(boundary)
    assert np.linalg.norm(np.sum(snapshot.nodal_forces, axis=0)) < 1e-12
    assert np.min(snapshot.tensions) >= -1e-12
