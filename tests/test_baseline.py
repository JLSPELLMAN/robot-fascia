from pathlib import Path

import mujoco
import numpy as np

from fascia_robot.model import actuator_id, load_model
from fascia_robot.trajectory import minimum_jerk_phase


ROOT = Path(__file__).resolve().parents[1]


def test_g1_loads_and_steps():
    ctx = load_model(ROOT / "models/g1/scene.xml", "stand", 0.002)
    assert ctx.model.nu == 29
    assert ctx.model.nq == 36
    assert ctx.model.nv == 35
    ctx.data.ctrl[:] = ctx.home_ctrl
    mujoco.mj_step(ctx.model, ctx.data)
    assert np.all(np.isfinite(ctx.data.qpos))


def test_expected_arm_and_shoulder_exist():
    ctx = load_model(ROOT / "models/g1/scene.xml", "stand", 0.002)
    assert actuator_id(ctx.model, "left_shoulder_pitch_joint") >= 0
    assert mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link") >= 0


def test_minimum_jerk_endpoints():
    assert minimum_jerk_phase(0.0, 1.0, 2.0) == 0.0
    assert minimum_jerk_phase(3.0, 1.0, 2.0) == 1.0
    assert np.isclose(minimum_jerk_phase(1.5, 1.0, 2.0), 0.5)

