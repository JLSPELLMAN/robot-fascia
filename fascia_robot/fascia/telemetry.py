from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .geometry import Topology
from .network import NetworkSnapshot


class FasciaRecorder:
    def __init__(self, topology: Topology):
        self.topology = topology
        self.samples: list[tuple[float, NetworkSnapshot]] = []

    def sample(self, time: float, snapshot: NetworkSnapshot) -> None:
        self.samples.append((float(time), snapshot))

    def write(self, directory: Path, metadata: dict) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        self._write_nodes(directory / "fascia_nodes.csv")
        self._write_edges(directory / "fascia_edges.csv")
        self._write_attachments(directory / "fascia_attachments.csv")
        summary = self.summary(metadata)
        with (directory / "fascia_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def _write_nodes(self, path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "node_id", "ring", "angular_index", "attached", "region", "px", "py", "pz", "vx", "vy", "vz", "residual_fx", "residual_fy", "residual_fz"])
            for time, snap in self.samples:
                for node in self.topology.nodes:
                    writer.writerow([time, node.id, node.ring, node.angular_index, node.attachment is not None, node.attachment.region if node.attachment else "", *snap.positions[node.id], *snap.velocities[node.id], *snap.nodal_forces[node.id]])

    def _write_edges(self, path: Path) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "edge_id", "node_a", "node_b", "edge_class", "length", "rest_length", "extension", "extension_rate", "strain", "stiffness", "damping", "elastic_force", "damping_force", "total_tension", "power", "stored_energy", "cumulative_damping_dissipation"])
            for time, snap in self.samples:
                for edge in self.topology.edges:
                    i = edge.id
                    writer.writerow([time, i, edge.node_a, edge.node_b, edge.edge_class, snap.lengths[i], snap.rest_lengths[i], snap.extensions[i], snap.extension_rates[i], snap.strains[i], edge.stiffness, edge.damping, snap.elastic_forces[i], snap.damping_forces[i], snap.tensions[i], snap.powers[i], snap.stored_energies[i], snap.cumulative_dissipation[i]])

    def _write_attachments(self, path: Path) -> None:
        attached = [n for n in self.topology.nodes if n.attachment]
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "node_id", "region", "body", "fx", "fy", "fz", "torque_x", "torque_y", "torque_z"])
            for time, snap in self.samples:
                center = np.mean(snap.positions[[n.id for n in attached]], axis=0)
                for index, node in enumerate(attached):
                    force = snap.attachment_forces[index]
                    torque = np.cross(snap.positions[node.id] - center, force)
                    writer.writerow([time, node.id, node.attachment.region, node.attachment.body_name, *force, *torque])

    def summary(self, metadata: dict) -> dict:
        snapshots = [s for _, s in self.samples]
        final = snapshots[-1]
        peak_tension = max(float(np.max(np.abs(s.tensions))) for s in snapshots)
        peak_strain = max(float(np.max(np.abs(s.strains))) for s in snapshots)
        peak_free_residual = max(s.free_force_residual for s in snapshots)
        edge_peak = np.max(np.abs(np.asarray([s.tensions for s in snapshots])), axis=0)
        active_edges = int(np.count_nonzero(edge_peak > 0.01))
        active_classes = sorted({edge.edge_class for edge in self.topology.edges if edge_peak[edge.id] > 0.01})
        attached = [n for n in self.topology.nodes if n.attachment]
        attachment_history = np.asarray([s.attachment_forces for s in snapshots])
        regional_peak_forces = {
            node.attachment.region: float(np.max(np.linalg.norm(attachment_history[:, i], axis=1)))
            for i, node in enumerate(attached)
        }
        return {
            **metadata,
            "topology_hash": self.topology.topology_hash,
            "node_count": len(self.topology.nodes),
            "edge_count": len(self.topology.edges),
            "attachment_count": len([n for n in self.topology.nodes if n.attachment]),
            "peak_edge_tension_n": peak_tension,
            "peak_edge_strain": peak_strain,
            "active_edges_above_0_01_n": active_edges,
            "active_edge_classes_above_0_01_n": active_classes,
            "regional_peak_attachment_force_n": regional_peak_forces,
            "peak_free_node_force_residual_n": peak_free_residual,
            "peak_net_attachment_force_n": max(float(np.linalg.norm(s.net_force)) for s in snapshots),
            "peak_net_attachment_torque_nm": max(float(np.linalg.norm(s.net_torque)) for s in snapshots),
            "final_net_attachment_force_n": final.net_force.tolist(),
            "final_net_attachment_torque_nm": final.net_torque.tolist(),
            "peak_stored_elastic_energy_j": max(float(np.sum(s.stored_energies)) for s in snapshots),
            "final_cumulative_damping_dissipation_j": float(np.sum(final.cumulative_dissipation)),
            "final_energy_balance_residual_j": final.energy_balance_residual,
            "all_solver_steps_converged": all(s.solver_success for s in snapshots),
        }
