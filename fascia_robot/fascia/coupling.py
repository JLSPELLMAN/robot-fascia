from __future__ import annotations

import mujoco
import numpy as np

from fascia_robot.model import ModelContext

from .geometry import Topology
from .network import FasciaNetwork, NetworkSnapshot


class RobotFasciaCoupling:
    """Maps massless network boundary reactions to MuJoCo body wrenches."""

    def __init__(self, context: ModelContext, topology: Topology, config: dict):
        self.context = context
        self.topology = topology
        model, data = context.model, context.data
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        torso_rotation = data.xmat[torso_id].reshape(3, 3)
        torso_origin = data.xpos[torso_id]
        world_positions = np.asarray([torso_origin + torso_rotation @ n.initial_position for n in topology.nodes])

        self.body_ids: dict[int, int] = {}
        self.local_offsets: dict[int, np.ndarray] = {}
        for node in topology.nodes:
            if node.attachment is None:
                continue
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node.attachment.body_name)
            if body_id < 0:
                raise ValueError(f"Missing attachment body: {node.attachment.body_name}")
            rotation = data.xmat[body_id].reshape(3, 3)
            self.body_ids[node.id] = body_id
            self.local_offsets[node.id] = rotation.T @ (world_positions[node.id] - data.xpos[body_id])

        world_topology = _topology_with_positions(topology, world_positions)
        self.network = FasciaNetwork(world_topology, model.opt.timestep, config["solver"])
        self.network.equilibrate(self.attachment_positions())

    def attachment_positions(self) -> dict[int, np.ndarray]:
        data = self.context.data
        positions = {}
        for nid, body_id in self.body_ids.items():
            rotation = data.xmat[body_id].reshape(3, 3)
            positions[nid] = data.xpos[body_id] + rotation @ self.local_offsets[nid]
        return positions

    def compute_and_apply(self) -> NetworkSnapshot:
        snapshot = self.network.step(self.attachment_positions())
        data = self.context.data
        for index, nid in enumerate(self.network.attachment_ids):
            body_id = self.body_ids[int(nid)]
            force = snapshot.attachment_forces[index]
            application = snapshot.positions[int(nid)]
            torque = np.cross(application - data.xipos[body_id], force)
            data.xfrc_applied[body_id, :3] += force
            data.xfrc_applied[body_id, 3:] += torque
        return snapshot


def _topology_with_positions(topology: Topology, positions: np.ndarray) -> Topology:
    from .geometry import EdgeSpec, NodeSpec, Topology
    nodes = tuple(NodeSpec(n.id, n.ring, n.angular_index, positions[n.id].copy(), n.attachment) for n in topology.nodes)
    edges = []
    for edge in topology.edges:
        length = float(np.linalg.norm(positions[edge.node_b] - positions[edge.node_a]))
        source_length = float(np.linalg.norm(
            topology.nodes[edge.node_b].initial_position - topology.nodes[edge.node_a].initial_position
        ))
        rest_ratio = edge.rest_length / source_length
        edges.append(EdgeSpec(edge.id, edge.node_a, edge.node_b, edge.edge_class, length * rest_ratio, edge.stiffness, edge.damping))
    return Topology(nodes, tuple(edges), topology.topology_hash)
