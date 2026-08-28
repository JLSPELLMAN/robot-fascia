from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import Topology
from .network import NetworkSnapshot


COLORS = {"circumferential": "#2389da", "longitudinal": "#28a745", "diagonal": "#f28e2b", "cross_body": "#8b5cf6"}


def plot_network(path: Path, topology: Topology, initial: np.ndarray, snapshot: NetworkSnapshot) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for edge in topology.edges:
        ids = [edge.node_a, edge.node_b]
        ax.plot(*initial[ids].T, color="0.75", alpha=0.2, linewidth=0.7)
        scale = abs(snapshot.tensions[edge.id]) / max(1e-12, np.max(np.abs(snapshot.tensions)))
        ax.plot(*snapshot.positions[ids].T, color=COLORS[edge.edge_class], alpha=0.25 + 0.75 * scale, linewidth=0.7 + 3.0 * scale)
    attached = np.array([n.id for n in topology.nodes if n.attachment])
    free = np.array([n.id for n in topology.nodes if not n.attachment])
    ax.scatter(*snapshot.positions[free].T, color="black", s=12, label="free node")
    ax.scatter(*snapshot.positions[attached].T, color="#d62728", marker="s", s=55, label="attachment")
    ax.set_xlabel("anterior x (m)"); ax.set_ylabel("left y (m)"); ax.set_zlabel("superior z (m)")
    ax.set_box_aspect((1, 1.2, 1.2)); ax.legend()
    ax.set_title("Phase 3 connected torso exofascia — color/width indicate edge class/tension")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
