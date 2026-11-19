"""Twice-around-the-tree (2-aproximado) — CONTRATO para implementação pelo restante do grupo.

Assinatura esperada (já consumida pelo pipeline de warm-start e experimentos):

    solve(instance: TSPInstance, profiler: ProfilerSession | None = None) -> TourResult

Requisitos do contrato (Aula 13, slides do Twice-around-the-tree):
  * `tour`: permutação de {0..N-1} começando em 0, sem repetir o vértice inicial.
  * `cost`: `instance.tour_cost(tour)`.
  * Profiling por subrotina usando `prof.profile(...)` com AO MENOS os nomes:
      - "mst"          (árvore geradora mínima a partir da raiz 0)
      - "preorder"     (caminhamento em pré-ordem / shortcutting)
    para que os gráficos e regressões tratem os 3 algoritmos uniformemente.
  * Sugestão: reaproveitar `prim_dense` de `src.algorithms.christofides.v1_optimized`
    (mesma AGM do Christofides, facilitando comparações justas).

Enquanto `solve` levantar NotImplementedError, o warm-start "twice_around" é
marcado como indisponível (NA) pelos experimentos, sem quebrar o restante.
"""

from __future__ import annotations

from typing import Optional

from src.algorithms.common import TourResult
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "twice_around_the_tree"


def solve(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    raise NotImplementedError(
        "Twice-around-the-tree: a ser implementado pelo restante do grupo (ver docstring do módulo)."
    )
