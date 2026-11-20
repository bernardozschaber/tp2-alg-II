"""Algoritmo Exato Branch-and-Bound para o TSP (Aulas 11 e 12) em 4 versões graduais:

- v0_naive      : Fiel ao slide 15 da Aula 12 (Best-First Heap, cópia de listas, bound O(N²) em CPU)
- v1_gpu        : 1ª Otimização — Avaliação de bounds e expansão em lote na GPU (--use-gpu)
- v2_dfs_memory : 2ª Otimização — Refatoração de memória: Best-Child-First DFS in-place O(N) (+ --use-gpu)
- v3_fast_bound : 3ª Otimização — Bound incremental O(1), bitmask inteiro, quebra de simetria (+ --use-gpu)
"""

from .v0_naive import solve as solve_v0
from .v1_gpu import solve as solve_v1
from .v2_dfs_memory import solve as solve_v2
from .v3_fast_bound import solve as solve_v3

VERSIONS = {
    "v0": solve_v0,
    "v1": solve_v1,
    "v2": solve_v2,
    "v3": solve_v3,
}

__all__ = ["solve_v0", "solve_v1", "solve_v2", "solve_v3", "VERSIONS"]
