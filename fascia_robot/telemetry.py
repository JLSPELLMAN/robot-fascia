from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .model import ModelContext


@dataclass
class Recorder:
    context: ModelContext
    time: list[float] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    qacc: list[np.ndarray] = field(default_factory=list)
    torque: list[np.ndarray] = field(default_factory=list)
    power: list[np.ndarray] = field(default_factory=list)
    com: list[np.ndarray] = field(default_factory=list)
    torso_quat: list[np.ndarray] = field(default_factory=list)
    target: list[np.ndarray] = field(default_factory=list)
    external_force: list[np.ndarray] = field(default_factory=list)

    def sample(self, target: np.ndarray, external_force: np.ndarray) -> None:
        m, d = self.context.model, self.context.data
        self.time.append(float(d.time))
        self.qpos.append(d.qpos.copy())
        self.qvel.append(d.qvel.copy())
        self.qacc.append(d.qacc.copy())
        self.torque.append(d.actuator_force.copy())
        self.power.append(d.actuator_force * d.qvel[self.context.actuator_dof_ids])
        self.com.append(d.subtree_com[self.context.robot_root_body_id].copy())
        torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.torso_quat.append(d.xquat[torso_id].copy())
        self.target.append(target.copy())
        self.external_force.append(external_force.copy())

    def arrays(self) -> dict[str, np.ndarray]:
        out = {name: np.asarray(getattr(self, name)) for name in (
            "time", "qpos", "qvel", "qacc", "torque", "power", "com",
            "torso_quat", "target", "external_force"
        )}
        dt = np.gradient(out["time"])
        out["jerk"] = np.gradient(out["qacc"], axis=0) / dt[:, None]
        out["actuator_energy_abs"] = np.cumsum(np.abs(out["power"]) * dt[:, None], axis=0)
        return out


def write_csv(path: Path, arrays: dict[str, np.ndarray], context: ModelContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_groups = {
        "qpos": [f"qpos_{i}" for i in range(context.model.nq)],
        "qvel": [f"qvel_{i}" for i in range(context.model.nv)],
        "qacc": [f"qacc_{i}" for i in range(context.model.nv)],
        "jerk": [f"jerk_{i}" for i in range(context.model.nv)],
        "torque": [f"torque_{n}" for n in context.actuator_names],
        "power": [f"power_{n}" for n in context.actuator_names],
        "actuator_energy_abs": [f"energy_abs_{n}" for n in context.actuator_names],
        "target": [f"target_{n}" for n in context.actuator_names],
        "com": ["com_x", "com_y", "com_z"],
        "torso_quat": ["torso_qw", "torso_qx", "torso_qy", "torso_qz"],
        "external_force": ["external_fx", "external_fy", "external_fz"],
    }
    header = ["time"] + [h for group in scalar_groups.values() for h in group]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, t in enumerate(arrays["time"]):
            writer.writerow([t] + [v for group in scalar_groups for v in arrays[group][i]])

