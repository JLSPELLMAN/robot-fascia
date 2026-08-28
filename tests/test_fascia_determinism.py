import numpy as np
import yaml
import mujoco

from fascia_robot.fascia.coupling import RobotFasciaCoupling
from fascia_robot.fascia.geometry import build_torso_topology
from fascia_robot.fascia.network import FasciaNetwork
from fascia_robot.model import load_model
from fascia_robot.runner import PROJECT_ROOT


def run_once():
    config = yaml.safe_load((PROJECT_ROOT / "configs/fascia_phase3.yaml").read_text())
    topology = build_torso_topology(config); network = FasciaNetwork(topology, 0.01, config["solver"])
    boundary = {n.id: n.initial_position.copy() for n in topology.nodes if n.attachment}
    network.equilibrate(boundary)
    left = next(n.id for n in topology.nodes if n.attachment and n.attachment.region == "top_left")
    boundary[left][1] += 0.007
    for _ in range(20): snapshot = network.step(boundary)
    return snapshot


def test_network_solver_is_deterministic():
    first, second = run_once(), run_once()
    for field in ("positions", "tensions", "attachment_forces", "stored_energies", "cumulative_dissipation"):
        assert np.array_equal(getattr(first, field), getattr(second, field))


def coupled_run():
    fconfig = yaml.safe_load((PROJECT_ROOT / "configs/fascia_phase3.yaml").read_text())
    econfig = yaml.safe_load((PROJECT_ROOT / "configs/experiments.yaml").read_text())
    model_cfg = econfig["model"]
    ctx = load_model(PROJECT_ROOT / model_cfg["path"], model_cfg["keyframe"], model_cfg["timestep"])
    topology = build_torso_topology(fconfig); coupling = RobotFasciaCoupling(ctx, topology, fconfig)
    shoulder = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
    for step in range(400):
        ctx.data.ctrl[:] = ctx.home_ctrl; ctx.data.xfrc_applied[:] = 0.0
        if 150 <= step < 190:
            ctx.data.xfrc_applied[shoulder, 1] = 120.0
        snapshot = coupling.compute_and_apply(); mujoco.mj_step(ctx.model, ctx.data)
    return ctx.data.qpos.copy(), snapshot.tensions.copy(), snapshot.cumulative_dissipation.copy()


def test_coupled_g1_fascia_run_is_deterministic():
    first, second = coupled_run(), coupled_run()
    for first_array, second_array in zip(first, second):
        assert np.array_equal(first_array, second_array)
