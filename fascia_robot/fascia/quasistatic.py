from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .material_models import AnatomicalMechanics, MechanicalSnapshot


@dataclass(frozen=True)
class EquilibriumResult:
    positions: np.ndarray
    snapshot: MechanicalSnapshot
    free_force_residual_n: float
    success: bool
    iterations: int
    message: str


class QuasiStaticAnatomicalSolver:
    """Energy minimizer for massless internal nodes with prescribed attachments."""
    def __init__(self, mechanics: AnatomicalMechanics, config: dict):
        self.mechanics = mechanics; self.config = config["quasistatic_solver"]
        self.attachment_ids = np.asarray([n.id for n in mechanics.topology.nodes if n.attachment is not None], dtype=int)
        self.free_ids = np.asarray([n.id for n in mechanics.topology.nodes if n.attachment is None], dtype=int)
        self.positions = mechanics.initial.copy()

    def _objective(self, flat: np.ndarray, fixed: np.ndarray) -> tuple[float, np.ndarray]:
        positions = fixed.copy(); positions[self.free_ids] = flat.reshape(-1, 3)
        snapshot = self.mechanics.evaluate(positions)
        return snapshot.stored_energy, -snapshot.nodal_forces[self.free_ids].ravel()

    def solve(self, attachments: dict[int, np.ndarray], initial: np.ndarray | None = None) -> EquilibriumResult:
        fixed = self.positions.copy() if initial is None else np.asarray(initial, dtype=float).copy()
        for nid in self.attachment_ids:
            fixed[nid] = attachments.get(int(nid), fixed[nid])
        maximum = float(self.config["maximum_displacement_m"])
        origin = self.mechanics.initial[self.free_ids].ravel()
        bounds = list(zip(origin-maximum, origin+maximum))
        result = minimize(self._objective, fixed[self.free_ids].ravel(), args=(fixed,), jac=True,
                          method="L-BFGS-B", bounds=bounds,
                          options={"gtol": float(self.config["force_tolerance_n"]), "maxiter": int(self.config["maximum_iterations"]), "ftol": 1e-14, "maxls": 40})
        positions = fixed.copy(); positions[self.free_ids] = result.x.reshape(-1, 3)
        snapshot = self.mechanics.evaluate(positions)
        residual = float(np.max(np.linalg.norm(snapshot.nodal_forces[self.free_ids], axis=1)))
        self.positions = positions.copy()
        tolerance = float(self.config["force_tolerance_n"])
        return EquilibriumResult(positions, snapshot, residual, bool(result.success and residual <= 10*tolerance), int(result.nit), str(result.message))
