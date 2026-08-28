from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_arm(path: Path, a: dict[str, np.ndarray], actuator_id: int, dof_id: int) -> None:
    qpos_id = dof_id + 1
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(a["time"], np.rad2deg(a["target"][:, actuator_id]), "--", label="command")
    axes[0].plot(a["time"], np.rad2deg(a["qpos"][:, qpos_id]), label="measured")
    axes[0].set_ylabel("Shoulder pitch (deg)"); axes[0].legend()
    axes[1].plot(a["time"], a["jerk"][:, dof_id]); axes[1].set_ylabel("Jerk (rad/s³)")
    axes[2].plot(a["time"], a["torque"][:, actuator_id], label="torque (N·m)")
    axes[2].plot(a["time"], a["power"][:, actuator_id], label="power (W)")
    axes[2].set_xlabel("Time (s)"); axes[2].legend()
    fig.suptitle("Unitree G1 baseline — arm raise")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_perturbation(path: Path, a: dict[str, np.ndarray], cfg: dict) -> None:
    pre = (a["time"] >= cfg["force_start"] - 0.2) & (a["time"] < cfg["force_start"])
    baseline = np.mean(a["com"][pre], axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(a["time"], a["external_force"][:, 1]); axes[0].set_ylabel("Lateral force (N)")
    axes[1].plot(a["time"], 1000 * (a["com"][:, 1] - baseline[1])); axes[1].set_ylabel("Lateral COM (mm)")
    axes[2].plot(a["time"], np.sum(np.abs(a["power"]), axis=1)); axes[2].set_ylabel("Σ |power| (W)")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle("Unitree G1 baseline — lateral shoulder perturbation")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

