"""Contrato de interface para o algoritmo aproximado Twice-around-the-tree.

Assinatura esperada para integração direta com os warm-starts e scripts de experimento:

    solve(instance: TSPInstance, profiler: ProfilerSession | None = None) -> TourResult

Requisitos operacionais do contrato:
  - Retornar `TourResult` com `tour` permutando `{0..N-1}` a partir do vértice `0` (sem
    repetir `0` no final) e `cost == instance.tour_cost(tour)`.
  - Instrumentar as subrotinas usando `prof.profile("mst")` e `prof.profile("preorder")`
    para manter compatibilidade com as tabelas e gráficos de profiling.
  - A função `prim_dense` em `src.algorithms.christofides.v1_optimized` pode ser
    reaproveitada diretamente para o cálculo da AGM em Θ(N²).

Enquanto a implementação não for integrada, `solve` lança `NotImplementedError`, tratado
graciosamente por `warm_start.run_warm_start` retornando `None`.
"""

from __future__ import annotations

from typing import Optional

from src.algorithms.common import TourResult
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "twice_around_the_tree"


def solve(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    """Ponto de entrada reservado para a implementação do Twice-around-the-tree."""
    raise NotImplementedError(
        "Twice-around-the-tree pendente de integração (ver contrato na docstring do módulo)."
    )
