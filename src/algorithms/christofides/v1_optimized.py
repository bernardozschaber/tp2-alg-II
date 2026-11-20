"""Christofides v1 (refatorado).

Refatorações em relação ao v0:
  1. Reaproveita a matriz de distâncias NumPy da instância (sem grafo completo em objetos).
  2. AGM por Prim denso vetorizado: Θ(N²) tempo e Θ(N) memória extra — ótimo para grafo
     completo (|E| = Θ(N²)), sem heap e sem Θ(N²) dicionários de aresta.
  3. Vértices ímpares por vetor de graus (Θ(N)).
  4. Matching perfeito de peso mínimo: o NetworkX (Blossom de Edmonds, O(|I|³)) é usado
     apenas como suporte, instanciando um grafo SÓ com os |I| vértices ímpares.
  5. Circuito euleriano por Hierholzer iterativo sobre listas de adjacência do
     multigrafo T ⊎ M (|E| = (N-1) + |I|/2 = Θ(N)), em Θ(N).
  6. Shortcutting em Θ(N).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from src.algorithms.common import TourResult, shortcut_walk
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "christofides_v1_optimized"


def prim_dense(dist: np.ndarray, root: int = 0) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """AGM por Prim com vetor de chaves (sem heap): Θ(N²). Retorna (arestas, graus)."""
    n = dist.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    key[root] = 0.0
    edges: List[Tuple[int, int]] = []
    degree = np.zeros(n, dtype=np.int64)
    for _ in range(n):
        u = int(np.argmin(key))
        in_tree[u] = True
        key[u] = np.inf
        if parent[u] >= 0:
            p = int(parent[u])
            edges.append((p, u))
            degree[p] += 1
            degree[u] += 1
        row = dist[u]
        improve = (~in_tree) & (row < key)
        key[improve] = row[improve]
        parent[improve] = u
    return edges, degree


def min_weight_perfect_matching(dist: np.ndarray, odd: List[int]) -> List[Tuple[int, int]]:
    """Matching perfeito de peso mínimo (Blossom/NetworkX) restrito ao subgrafo induzido por `odd`."""
    G = nx.Graph()
    k = len(odd)
    for i in range(k):
        u = odd[i]
        for j in range(i + 1, k):
            v = odd[j]
            G.add_edge(u, v, weight=float(dist[u, v]))
    return [(int(u), int(v)) for u, v in nx.min_weight_matching(G, weight="weight")]


def hierholzer(n: int, edges: List[Tuple[int, int]], start: int = 0) -> List[int]:
    """Circuito euleriano (Hierholzer iterativo) em multigrafo dado por lista de arestas. Θ(|E|)."""
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    used = [False] * len(edges)
    ptr = [0] * n
    stack = [start]
    circuit: List[int] = []
    while stack:
        u = stack[-1]
        while ptr[u] < len(adj[u]) and used[adj[u][ptr[u]][1]]:
            ptr[u] += 1
        if ptr[u] == len(adj[u]):
            circuit.append(stack.pop())
        else:
            v, eid = adj[u][ptr[u]]
            used[eid] = True
            stack.append(v)
    circuit.reverse()
    return circuit


def solve(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    dist = instance.dist_matrix
    t0 = time.perf_counter()

    with prof.profile("mst"):
        mst_edges, degree = prim_dense(dist, root=0)

    with prof.profile("odd_vertices"):
        odd = [int(v) for v in np.flatnonzero(degree % 2 == 1)]

    with prof.profile("min_weight_matching"):
        matching = min_weight_perfect_matching(dist, odd)

    with prof.profile("eulerian_circuit"):
        walk = hierholzer(n, mst_edges + matching, start=0)

    with prof.profile("shortcut"):
        tour = shortcut_walk(walk, start=0)

    elapsed = time.perf_counter() - t0
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
            "matching_cost": float(sum(dist[u, v] for u, v in matching)),
            "n_odd": len(odd),
        },
    )
