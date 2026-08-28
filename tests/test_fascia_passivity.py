import numpy as np
import yaml

from fascia_robot.fascia.geometry import build_torso_topology
from fascia_robot.fascia.network import FasciaNetwork
from fascia_robot.runner import PROJECT_ROOT
from fascia_robot.trajectory import minimum_jerk_phase


def run_cycle(damping: bool):
    config = yaml.safe_load((PROJECT_ROOT / "configs/fascia_phase3.yaml").read_text())
    if not damping:
        config["damping_ns_per_m"] = {key: 0.0 for key in config["damping_ns_per_m"]}
    topology = build_torso_topology(config)
    network = FasciaNetwork(topology, 0.01, config["solver"])
    base = {n.id: n.initial_position.copy() for n in topology.nodes if n.attachment}
    network.equilibrate(base)
    initial_energy = network.initial_stored_energy
    left = next(n.id for n in topology.nodes if n.attachment and n.attachment.region == "top_left")
    history = []
    for time in np.linspace(0.0, 2.0, 201):
        if time <= 1.0:
            offset = 0.01 * minimum_jerk_phase(time, 0.0, 1.0)
        else:
            offset = 0.01 * (1.0 - minimum_jerk_phase(time, 1.0, 2.0))
        current = {nid: pos.copy() for nid, pos in base.items()}; current[left][1] += offset
        snapshot = network.step(current); history.append(snapshot)
    return network, history, initial_energy


def test_pure_elastic_closed_cycle_does_not_create_energy():
    network, history, initial_energy = run_cycle(False)
    assert abs(np.sum(history[-1].stored_energies) - initial_energy) < 1e-8
    assert abs(network.cumulative_attachment_work) < 2e-6
    assert np.sum(history[-1].cumulative_dissipation) == 0.0


def test_damping_is_nonnegative_and_energy_balance_closes():
    network, history, _ = run_cycle(True)
    dissipation = np.array([np.sum(s.cumulative_dissipation) for s in history])
    assert np.min(np.diff(dissipation)) >= -1e-12
    assert dissipation[-1] > 0.0
    assert np.min(history[-1].tensions) >= -1e-12
    # The unilateral active-set transition is nonsmooth; the residual remains
    # below 0.1 mJ and does not indicate net energy creation.
    assert abs(history[-1].energy_balance_residual) < 1e-4
