"""Massless, passive, connected torso exofascia mechanics."""

from .geometry import build_torso_topology
from .network import FasciaNetwork

__all__ = ["FasciaNetwork", "build_torso_topology"]
