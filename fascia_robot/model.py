from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class ModelContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    home_ctrl: np.ndarray
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    actuator_dof_ids: np.ndarray
    robot_root_body_id: int


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj, index) or f"unnamed_{index}"


def load_model(path: Path, keyframe: str, timestep: float) -> ModelContext:
    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    model.opt.timestep = timestep
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        raise ValueError(f"Missing keyframe: {keyframe}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    joint_names = tuple(_name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt))
    actuator_names = tuple(_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu))
    actuator_joint_ids = model.actuator_trnid[:, 0].astype(int)
    actuator_dof_ids = model.jnt_dofadr[actuator_joint_ids].astype(int)

    # Menagerie's stand keyframe includes ctrl, but deriving it from joint qpos
    # makes the mapping explicit and guards against absent keyframe controls.
    home_ctrl = np.empty(model.nu)
    for actuator_id, joint_id in enumerate(actuator_joint_ids):
        home_ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
    data.ctrl[:] = home_ctrl

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if root_id < 0:
        raise ValueError("G1 pelvis body not found")
    return ModelContext(model, data, home_ctrl, joint_names, actuator_names, actuator_dof_ids, root_id)


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if value < 0:
        raise ValueError(f"Actuator not found: {name}")
    return value

