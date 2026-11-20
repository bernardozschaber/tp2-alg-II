"""Implementações graduais do algoritmo exato Branch-and-Bound para o TSP simétrico.

Progressão das quatro versões (mesmo contrato de entrada/saída e métricas de busca):
  - `v0`: Best-First Search (`heapq`), cópia explícita de listas de caminho e recálculo
          completo do limite inferior em O(N²) por candidato.
  - `v1`: Best-First Search com extração em lotes (`batch_size`) e avaliação paralelizada
          de custos/limites inferiores via GPU (`--use-gpu` / CuPy) ou NumPy vetorizado.
  - `v2`: Best-Child-First DFS *in-place* (memória estritamente Θ(N), `max_queue_size <= N`),
          com pré-expansão da fronteira de nível 2 na GPU e avaliação vetorizada local.
  - `v3`: Best-Child-First DFS *in-place* com atualização incremental do limite inferior em
          Θ(1) escalar, conjunto de não visitados em bitmask inteiro, quebra de simetria
          reversa em nível interno (`unvis_mask < (1 << v1)`) e pré-expansão GPU.
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
