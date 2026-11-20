"""Twice-around-the-tree v1: implementação otimizada com Prim denso Θ(N²) e DFS iterativa Θ(N).

Refatorações algorítmicas e estruturais aplicadas sobre a versão `v0_naive`:
  1. Elimina completamente o `NetworkX` e a instanciação do grafo completo `nx.Graph`
     (N(N-1)/2 dicionários Python), operando diretamente sobre a matriz contígua `dist_matrix`.
  2. Substitui Kruskal O(N² log N) por `prim_dense` vetorizado em NumPy (compartilhado em
     `src.algorithms.common`), reduzindo a AGM para Θ(N²) estrito e espaço auxiliar Θ(N).
  3. Elimina a duplicação explícita de arestas (`nx.MultiGraph`), o circuito euleriano de
     tamanho 2(N-1) e o `shortcut_walk` com tabela hash (`set`): pelo teorema clássico, a
     ordem de primeira visita de um passeio euleriano em 2T é equivalente a um percurso
     DFS em pré-ordem diretamente sobre a AGM T.
  4. Utiliza DFS em pré-ordem iterativa com pilha explícita e listas de adjacência compactas
     em tempo e espaço Θ(N), evitando `RecursionError` em árvores degeneradas/caminhos longos
     e preservando a ordem natural de descoberta das arestas geradas por `prim_dense`.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from src.algorithms.common import TourResult, prim_dense
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "twice_around_v1_optimized"


def preorder_dfs_iterative(n: int, edges: List[Tuple[int, int]], start: int = 0) -> List[int]:
    """Percorre a AGM em pré-ordem usando pilha explícita e listas de adjacência compactas.

    Estruturas de dados:
      - `adj`: lista de adjacência com exatamente 2(N-1) inteiros (sem objetos `MultiGraph`).
      - `visited`: vetor booleano de tamanho N evitando tabela hash (`set`).
      - `stack`: pilha explícita de profundidade máxima N (imune a `RecursionError`).

    Os vizinhos de cada vértice são empilhados em ordem reversa (`reversed(adj[u])`) para que
    o desempilhamento LIFO visite os filhos exatamente na ordem natural de inserção do Prim.
    Complexidade: Θ(N) em tempo e Θ(N) em espaço auxiliar.
    """
    if n == 0:
        return []

    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * n
    stack = [start]
    tour: List[int] = []

    while stack:
        u = stack.pop()
        # Evita revisitar o vértice pai na lista de adjacência não direcionada da árvore.
        if visited[u]:
            continue
        visited[u] = True
        tour.append(u)
        # Empilha em ordem reversa para compensar o comportamento LIFO e preservar a ordem do Prim.
        for v in reversed(adj[u]):
            if not visited[v]:
                stack.append(v)

    return tour


def solve(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    """Executa o Twice-around-the-tree v1 com profiling isolado em `mst` e `preorder`."""
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    dist = instance.dist_matrix
    t0 = time.perf_counter()

    # Etapa 1: AGM densa por Prim vetorizado em Θ(N²) e espaço auxiliar Θ(N).
    with prof.profile("mst"):
        mst_edges, degree = prim_dense(dist, root=0)

    # Etapa 2: percurso DFS iterativo em pré-ordem diretamente sobre T em Θ(N).
    with prof.profile("preorder"):
        tour = preorder_dfs_iterative(n, mst_edges, start=0)

    elapsed = time.perf_counter() - t0
    # Grau máximo na AGM (feature topológica útil para análise de hubs nas instâncias).
    mst_max_degree = int(degree.max()) if n > 0 else 0
    return TourResult(
        algorithm=ALGORITHM_NAME,
        instance=instance.name,
        n=n,
        tour=tour,
        cost=instance.tour_cost(tour),
        elapsed_s=elapsed,
        profile=prof.summary_dict(),
        extra={
            "mst_cost": float(sum(dist[u, v] for u, v in mst_edges)),
            "mst_max_degree": mst_max_degree,
        },
    )
