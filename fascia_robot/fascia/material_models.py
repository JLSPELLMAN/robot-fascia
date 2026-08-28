from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .full_body_geometry import FullBodyEdge, FullBodyTopology


@dataclass(frozen=True)
class AxialMaterial:
    name: str
    k1_n: float
    k2_n: float
    toe_strain: float
    exponent: float
    damping_ns_per_m: float


@dataclass(frozen=True)
class EdgeState:
    length: float
    rest_length: float
    strain: float
    extension_rate: float
    elastic_force: float
    damping_force: float
    tension: float
    stored_energy: float
    dissipation_rate: float


@dataclass(frozen=True)
class MechanicalSnapshot:
    nodal_forces: np.ndarray
    edge_states: tuple[EdgeState, ...]
    stored_energy: float
    dissipation_rate: float
    net_internal_force: np.ndarray
    net_internal_torque: np.ndarray


def nonlinear_elastic_force(strain: float, material: AxialMaterial) -> float:
    if strain <= 0.0:
        return 0.0
    if strain <= material.toe_strain:
        return material.k1_n * strain
    excess = strain - material.toe_strain
    return material.k1_n * material.toe_strain + material.k2_n * excess**material.exponent


def nonlinear_elastic_energy(strain: float, rest_length: float, material: AxialMaterial) -> float:
    if strain <= 0.0:
        return 0.0
    toe = material.toe_strain
    if strain <= toe:
        integral = .5 * material.k1_n * strain**2
    else:
        excess = strain - toe
        integral = (.5 * material.k1_n * toe**2 + material.k1_n * toe * excess
                    + material.k2_n * excess**(material.exponent + 1.0) / (material.exponent + 1.0))
    return rest_length * integral


def _material_key(edge: FullBodyEdge) -> str:
    if edge.layer == "superficial": return "superficial"
    if edge.layer == "reinforced": return edge.edge_class.removeprefix("reinforced_")
    if edge.layer == "deep":
        if "trunk" in edge.region or edge.region in {"deep_lumbar", "deep_abdominal", "deep_thoracic", "deep_scapular_chest"}: return "deep_trunk"
        if "pelvi" in edge.region or "gluteal" in edge.region: return "deep_pelvis"
        if "plantar" in edge.edge_class: return "deep_plantar"
        if "crural" in edge.region: return "deep_crural"
        if "thigh" in edge.region or "fascia_lata" in edge.region: return "deep_thigh"
        return "deep_arm"
    raise ValueError(f"No axial material for layer {edge.layer}")


def _direction_key(edge: FullBodyEdge) -> str:
    if edge.layer == "reinforced": return "reinforced"
    if "plantar_longitudinal" in edge.edge_class: return "plantar_longitudinal"
    if "plantar_transverse" in edge.edge_class: return "plantar_transverse"
    if "circumferential" in edge.edge_class: return "circumferential"
    if "longitudinal" in edge.edge_class: return "longitudinal"
    return "diagonal"


def assign_axial_material(edge: FullBodyEdge, config: dict) -> AxialMaterial:
    key = _material_key(edge); values = config["axial_materials"][key]
    multiplier = float(config["direction_multipliers"][_direction_key(edge)])
    return AxialMaterial(key, float(values["k1_n"])*multiplier, float(values["k2_n"])*multiplier,
                         float(values["toe_strain"]), float(values["exponent"]), float(values["damping_ns_per_m"])*multiplier)


class AnatomicalMechanics:
    """Stateless passive force evaluator for Phase D full-body topology."""
    def __init__(self, topology: FullBodyTopology, config: dict):
        self.topology = topology; self.config = config
        self.initial = np.asarray([node.position for node in topology.nodes], dtype=float)
        pretension = float(config["pretension"])
        by_layer = config.get("pretension_by_layer", {})
        self.rest_lengths = np.asarray([
            np.linalg.norm(self.initial[e.node_b]-self.initial[e.node_a]) *
            (1.0-(0.0 if e.layer=="interface" else float(by_layer.get(e.layer,pretension))))
            for e in topology.edges
        ])
        self.rest_vectors = np.asarray([self.initial[e.node_b] - self.initial[e.node_a] for e in topology.edges])
        self.materials = tuple(None if e.layer == "interface" else assign_axial_material(e, config) for e in topology.edges)

    def evaluate(self, positions: np.ndarray, velocities: np.ndarray | None = None) -> MechanicalSnapshot:
        positions = np.asarray(positions, dtype=float)
        velocities = np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=float)
        forces = np.zeros_like(positions); states = []; total_energy = 0.0; total_dissipation = 0.0
        for i, edge in enumerate(self.topology.edges):
            a, b = edge.node_a, edge.node_b
            delta = positions[b] - positions[a]; relative_velocity = velocities[b] - velocities[a]
            length = float(np.linalg.norm(delta)); rest = float(self.rest_lengths[i])
            if edge.layer == "interface":
                if length <= 1e-12: raise ValueError("Collapsed shear interface")
                direction = delta/length; extension = length-rest
                rate = float(relative_velocity @ direction)
                k = float(self.config["shear_interface"]["stiffness_n_per_m"])
                c = float(self.config["shear_interface"]["damping_ns_per_m"])
                elastic_scalar = k*extension; damping_scalar = c*rate
                vector_force = (elastic_scalar+damping_scalar)*direction
                forces[a] += vector_force; forces[b] -= vector_force
                energy = .5*k*extension**2; dissipation = c*rate**2
                states.append(EdgeState(length, rest, extension/rest, rate, elastic_scalar, damping_scalar,
                                        elastic_scalar+damping_scalar, energy, dissipation))
            else:
                if length <= 1e-12: raise ValueError("Collapsed axial edge")
                direction = delta / length; rate = float(relative_velocity @ direction); strain = (length-rest)/rest
                material = self.materials[i]; elastic = nonlinear_elastic_force(strain, material)
                trial = elastic + material.damping_ns_per_m*rate
                tension = max(0.0, trial); damping_force = tension-elastic
                vector_force = tension*direction; forces[a] += vector_force; forces[b] -= vector_force
                energy = nonlinear_elastic_energy(strain, rest, material)
                dissipation = max(0.0, damping_force*rate)
                states.append(EdgeState(length, rest, strain, rate, elastic, damping_force, tension, energy, dissipation))
            total_energy += energy; total_dissipation += dissipation
        center = np.mean(positions, axis=0)
        return MechanicalSnapshot(forces, tuple(states), total_energy, total_dissipation,
                                  np.sum(forces, axis=0), np.sum(np.cross(positions-center, forces), axis=0))
