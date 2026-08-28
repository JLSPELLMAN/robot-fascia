from __future__ import annotations

import numpy as np
import yaml

from fascia_robot.fascia.deep_regional_geometry import add_deep_regional_sleeves
from fascia_robot.fascia.deep_attachments import add_selected_deep_attachments
from fascia_robot.fascia.full_body_geometry import build_full_body_superficial_mesh
from fascia_robot.fascia.material_models import AnatomicalMechanics, assign_axial_material, nonlinear_elastic_energy, nonlinear_elastic_force
from fascia_robot.fascia.reinforced_pathways import add_reinforced_pathways
from fascia_robot.fascia.shear_interfaces import add_shear_interfaces
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


def _build():
    geometry = yaml.safe_load((PROJECT_ROOT / "configs/fascia_anatomical_v1.yaml").read_text())
    materials = yaml.safe_load((PROJECT_ROOT / "configs/fascia_materials.yaml").read_text())
    context = load_model(PROJECT_ROOT / geometry["model_path"], geometry["keyframe"], .002)
    topology = build_full_body_superficial_mesh(context, geometry)
    topology = add_deep_regional_sleeves(topology, context, geometry)
    topology = add_reinforced_pathways(topology, geometry)
    topology = add_selected_deep_attachments(topology)
    topology = add_shear_interfaces(topology, materials)
    return topology, materials, AnatomicalMechanics(topology, materials)


def test_phase_d_adds_weak_interfaces_without_welding_or_new_nodes():
    topology, config, _ = _build()
    interfaces = [edge for edge in topology.edges if edge.layer == "interface"]
    assert len(topology.nodes) == 412 and len(topology.edges) == 1510
    assert len(interfaces) == 408
    assert len({edge.node_a for edge in interfaces if topology.nodes[edge.node_a].layer == "deep"} |
               {edge.node_b for edge in interfaces if topology.nodes[edge.node_b].layer == "deep"}) == 204
    deep_attached = sum(node.attachment is not None and node.layer == "deep" for node in topology.nodes)
    assert 44 <= deep_attached < 0.35*204
    assert config["shear_interface"]["stiffness_n_per_m"] < config["axial_materials"]["deep_trunk"]["k2_n"]


def test_directional_and_regional_materials_are_anisotropic():
    topology, config, mechanics = _build()
    deep_long = next(e for e in topology.edges if e.edge_class == "deep_longitudinal" and "trunk" in topology.nodes[e.node_a].component)
    deep_circ = next(e for e in topology.edges if e.edge_class == "deep_circumferential" and "trunk" in topology.nodes[e.node_a].component)
    assert assign_axial_material(deep_long, config).k1_n > assign_axial_material(deep_circ, config).k1_n
    lateral = next(e for e in topology.edges if e.edge_class == "reinforced_lateral_leg")
    arm = next(e for e in topology.edges if e.edge_class == "reinforced_upper_limb_continuity")
    assert assign_axial_material(lateral, config).k2_n > assign_axial_material(arm, config).k2_n
    assert len(mechanics.materials) == len(topology.edges)


def test_nonlinear_law_is_tension_only_continuous_and_stiffens_after_toe():
    topology, config, _ = _build()
    edge = next(e for e in topology.edges if e.layer == "reinforced")
    material = assign_axial_material(edge, config); toe = material.toe_strain
    assert nonlinear_elastic_force(-.1, material) == 0.0
    assert nonlinear_elastic_energy(-.1, .2, material) == 0.0
    assert abs(nonlinear_elastic_force(toe-1e-9, material)-nonlinear_elastic_force(toe+1e-9, material)) < 1e-5
    low_slope = (nonlinear_elastic_force(.5*toe, material)-nonlinear_elastic_force(.25*toe, material))/(.25*toe)
    high_slope = (nonlinear_elastic_force(3*toe, material)-nonlinear_elastic_force(2*toe, material))/toe
    assert high_slope > low_slope


def test_zero_deformation_has_zero_force_and_energy():
    _, _, mechanics = _build(); snapshot = mechanics.evaluate(mechanics.initial)
    assert np.max(np.abs(snapshot.nodal_forces)) < 1e-12
    assert snapshot.stored_energy < 1e-12
    assert snapshot.dissipation_rate < 1e-12


def test_internal_forces_balance_and_create_no_net_momentum():
    topology, _, mechanics = _build(); positions = mechanics.initial.copy()
    node = next(n for n in topology.nodes if n.component == "deep_left_arm" and n.ring == 0)
    positions[node.id] += np.array([.006, .004, .002])
    snapshot = mechanics.evaluate(positions)
    assert np.linalg.norm(snapshot.net_internal_force) < 1e-10
    assert np.linalg.norm(snapshot.net_internal_torque) < 1e-10
    assert min(state.tension for edge, state in zip(topology.edges, snapshot.edge_states) if edge.layer != "interface") >= 0.0


def test_elastic_closed_cycle_returns_energy_and_damping_only_dissipates():
    topology, config, elastic = _build(); base = elastic.initial.copy()
    moving = next(n.id for n in topology.nodes if n.component == "deep_left_arm" and n.ring == 0)
    peak = base.copy(); peak[moving] += np.array([.008, .004, 0.0])
    assert elastic.evaluate(peak).stored_energy > 0.0
    assert elastic.evaluate(base).stored_energy < 1e-12
    dt = .002; cumulative = 0.0; boundary_work = 0.0; previous = base.copy(); previous_support = np.zeros(3)
    for phase in np.concatenate((np.linspace(0, 1, 101), np.linspace(1, 0, 101)[1:])):
        current = base.copy(); current[moving] += phase*np.array([.008, .004, 0.0])
        velocity = (current-previous)/dt
        snapshot = elastic.evaluate(current, velocity)
        assert snapshot.dissipation_rate >= -1e-12
        support = -snapshot.nodal_forces[moving]
        boundary_work += .5*float((previous_support+support) @ (current[moving]-previous[moving]))
        cumulative += snapshot.dissipation_rate*dt; previous = current; previous_support = support
    assert cumulative > 0.0
    assert elastic.evaluate(base).stored_energy < 1e-12
    assert abs(boundary_work-cumulative) < 5e-6


def test_shear_interface_allows_finite_glide_with_passive_reaction():
    topology, config, mechanics = _build(); positions = mechanics.initial.copy()
    edge = next(e for e in topology.edges if e.layer == "interface")
    deep = edge.node_a if topology.nodes[edge.node_a].layer == "deep" else edge.node_b
    positions[deep] += np.array([0.0, .003, 0.0])
    snapshot = mechanics.evaluate(positions)
    state = snapshot.edge_states[edge.id]
    expected = .5*config["shear_interface"]["stiffness_n_per_m"]*.003**2
    assert state.stored_energy > 0.0
    assert state.stored_energy <= expected + 1e-12
    assert np.linalg.norm(snapshot.nodal_forces[deep]) > 0.0
