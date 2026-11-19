"""Algoritmo de Christofides (1.5-aproximado) em versões graduais."""

from .v0_naive import solve as solve_v0
from .v1_optimized import solve as solve_v1

VERSIONS = {"v0": solve_v0, "v1": solve_v1}

__all__ = ["solve_v0", "solve_v1", "VERSIONS"]
