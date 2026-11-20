"""Implementações do algoritmo Twice-around-the-tree em duas versões (`v0` baseline e `v1` otimizada).

Expõe `solve_v0`, `solve_v1`, o dicionário `VERSIONS` e o alias padrão `solve` apontando
para `solve_v1`, preservando compatibilidade direta com `src.algorithms.warm_start` e com
os scripts de experimentação.
"""

from .v0_naive import solve as solve_v0
from .v1_optimized import solve as solve_v1

ALGORITHM_NAME = "twice_around_the_tree"
VERSIONS = {"v0": solve_v0, "v1": solve_v1}
solve = solve_v1

__all__ = ["ALGORITHM_NAME", "VERSIONS", "solve", "solve_v0", "solve_v1"]
