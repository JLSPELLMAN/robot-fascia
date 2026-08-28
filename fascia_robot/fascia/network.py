from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .geometry import Topology


@dataclass
class NetworkSnapshot:
    positions: np.ndarray
    velocities: np.ndarray
    lengths: np.ndarray
    rest_lengths: np.ndarray
    extensions: np.ndarray
    extension_rates: np.ndarray
    strains: np.ndarray
    elastic_forces: np.ndarray
    damping_forces: np.ndarray
    tensions: np.ndarray
    powers: np.ndarray
    stored_energies: np.ndarray
    cumulative_dissipation: np.ndarray
    nodal_forces: np.ndarray
    attachment_forces: np.ndarray
    net_force: np.ndarray
    net_torque: np.ndarray
    free_force_residual: float
    energy_balance_residual: float
    solver_success: bool
    solver_iterations: int


class FasciaNetwork:
    """Connected massless graph solved at quasi-static equilibrium."""

    def __init__(self, topology: Topology, dt: float, solver_config: dict):
        self.topology = topology
        self.dt = float(dt)
        self.force_tolerance = float(solver_config["force_tolerance_n"])
        self.position_tolerance = float(solver_config["position_tolerance_m"])
        self.maximum_iterations = int(solver_config["maximum_iterations"])
        self.a = np.array([e.node_a for e in topology.edges], dtype=int)
        self.b = np.array([e.node_b for e in topology.edges], dtype=int)
        self.rest = np.array([e.rest_length for e in topology.edges])
        self.k = np.array([e.stiffness for e in topology.edges])
        self.c = np.array([e.damping for e in topology.edges])
        self.unilateral_smoothing_n = 1e-5
        self.positions = np.asarray([n.initial_position for n in topology.nodes], dtype=float)
        self.previous_positions = self.positions.copy()
        self.previous_lengths = self._lengths(self.positions)
        self.cumulative_dissipation = np.zeros(len(topology.edges))
        self.attachment_ids = np.array([n.id for n in topology.nodes if n.attachment is not None], dtype=int)
        self.free_ids = np.array([n.id for n in topology.nodes if n.attachment is None], dtype=int)
        self.initial_stored_energy = self._elastic_energy(self.positions)
        self.previous_stored_energy = self.initial_stored_energy
        self.cumulative_attachment_work = 0.0
        self.previous_attachment_forces = np.zeros((len(self.attachment_ids), 3))

    def _positive_force(self, raw: np.ndarray) -> np.ndarray:
        """C1 tension-only projection with a 10 µN transition interval."""
        eps = self.unilateral_smoothing_n
        return np.where(raw <= 0.0, 0.0, np.where(raw < eps, raw**2 / (2.0 * eps), raw - 0.5 * eps))

    def _positive_potential(self, raw: np.ndarray, slope: np.ndarray) -> np.ndarray:
        """Integral of `_positive_force(raw)` with respect to edge length."""
        eps = self.unilateral_smoothing_n
        low = raw**3 / (6.0 * eps * slope)
        high = (0.5 * raw**2 - 0.5 * eps * raw + eps**2 / 6.0) / slope
        return np.where(raw <= 0.0, 0.0, np.where(raw < eps, low, high))

    def equilibrate(self, attachment_positions: dict[int, np.ndarray]) -> NetworkSnapshot:
        """Find the pretension equilibrium and reset all energy bookkeeping."""
        damping = self.c.copy()
        self.c[:] = 0.0
        snapshot = self.step(attachment_positions)
        for _ in range(19):
            if snapshot.free_force_residual <= 2.0 * self.force_tolerance:
                break
            snapshot = self.step(attachment_positions)
        self.c[:] = damping
        self.previous_positions = self.positions.copy()
        self.previous_lengths = self._lengths(self.positions)
        self.cumulative_dissipation[:] = 0.0
        self.initial_stored_energy = self._elastic_energy(self.positions)
        self.previous_stored_energy = self.initial_stored_energy
        self.cumulative_attachment_work = 0.0
        self.previous_attachment_forces = -snapshot.attachment_forces.copy()
        return snapshot

    def _lengths(self, positions: np.ndarray) -> np.ndarray:
        return np.linalg.norm(positions[self.b] - positions[self.a], axis=1)

    def _elastic_energy(self, positions: np.ndarray) -> float:
        raw = self.k * (self._lengths(positions) - self.rest)
        return float(np.sum(self._positive_potential(raw, self.k)))

    def _objective_gradient(self, flat: np.ndarray, fixed: np.ndarray) -> tuple[float, np.ndarray]:
        positions = fixed.copy()
        positions[self.free_ids] = flat.reshape(-1, 3)
        delta = positions[self.b] - positions[self.a]
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths < 1e-9):
            return 1e20, np.zeros_like(flat)
        directions = delta / lengths[:, None]
        extension = lengths - self.rest
        length_step = lengths - self.previous_lengths
        rate = length_step / self.dt
        raw_axial = self.k * extension + self.c * rate
        axial = self._positive_force(raw_axial)
        # Smooth unilateral incremental potential. Slack edges contribute no
        # force and active-set transitions do not stall the equilibrium solve.
        effective_stiffness = self.k + self.c / self.dt
        objective = np.sum(self._positive_potential(raw_axial, effective_stiffness))
        forces = np.zeros_like(positions)
        np.add.at(forces, self.a, axial[:, None] * directions)
        np.add.at(forces, self.b, -axial[:, None] * directions)
        gradient = -forces[self.free_ids].ravel()
        return float(objective), gradient

    def step(self, attachment_positions: dict[int, np.ndarray]) -> NetworkSnapshot:
        old_positions = self.positions.copy()
        fixed = self.positions.copy()
        for nid in self.attachment_ids:
            fixed[nid] = attachment_positions[int(nid)]
        result = minimize(
            self._objective_gradient,
            self.positions[self.free_ids].ravel(),
            args=(fixed,), jac=True, method="BFGS",
            options={"gtol": self.force_tolerance, "maxiter": self.maximum_iterations},
        )
        positions = fixed
        positions[self.free_ids] = result.x.reshape(-1, 3)
        velocities = (positions - old_positions) / self.dt
        delta = positions[self.b] - positions[self.a]
        lengths = np.linalg.norm(delta, axis=1)
        directions = delta / lengths[:, None]
        extensions = lengths - self.rest
        # Backward difference is the same discrete dL/dt used by the implicit
        # damping potential, keeping equilibrium and passivity accounting aligned.
        rates = (lengths - self.previous_lengths) / self.dt
        raw_elastic = self.k * extensions
        elastic = self._positive_force(raw_elastic)
        raw_tension = raw_elastic + self.c * rates
        tension = self._positive_force(raw_tension)
        active = tension > 0.0
        damping = tension - elastic
        nodal = np.zeros_like(positions)
        np.add.at(nodal, self.a, tension[:, None] * directions)
        np.add.at(nodal, self.b, -tension[:, None] * directions)
        stored = self._positive_potential(raw_elastic, self.k)
        # Actual dashpot contribution after unilateral projection. The product
        # is non-negative for a passive engaged cable; clipping protects only
        # against floating-point noise at the smooth active-set transition.
        dissipated_increment = np.maximum(damping * rates * self.dt, 0.0)
        self.cumulative_dissipation += dissipated_increment
        # `attachment_forces` are fascia-on-robot forces. Their negatives are
        # the support forces doing work on the massless network boundary.
        attachment_forces = nodal[self.attachment_ids]
        support_forces = -attachment_forces
        center = np.mean(positions[self.attachment_ids], axis=0)
        net_force = np.sum(attachment_forces, axis=0)
        net_torque = np.sum(np.cross(positions[self.attachment_ids] - center, attachment_forces), axis=0)
        attachment_displacement = positions[self.attachment_ids] - old_positions[self.attachment_ids]
        average_reaction = 0.5 * (self.previous_attachment_forces + support_forces)
        self.cumulative_attachment_work += float(np.sum(average_reaction * attachment_displacement))
        total_stored = float(np.sum(stored))
        energy_residual = self.cumulative_attachment_work - (total_stored - self.initial_stored_energy) - float(np.sum(self.cumulative_dissipation))
        free_residual = float(np.max(np.linalg.norm(nodal[self.free_ids], axis=1)))
        snapshot = NetworkSnapshot(
            positions.copy(), velocities, lengths, self.rest.copy(), extensions,
            rates, extensions / self.rest, elastic, damping, tension,
            tension * rates, stored, self.cumulative_dissipation.copy(), nodal,
            attachment_forces, net_force, net_torque, free_residual,
            energy_residual, bool(result.success or free_residual <= 10 * self.force_tolerance), int(result.nit),
        )
        self.previous_positions = positions.copy()
        self.positions = positions.copy()
        self.previous_lengths = lengths.copy()
        self.previous_stored_energy = total_stored
        self.previous_attachment_forces = support_forces.copy()
        return snapshot
