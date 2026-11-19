"""Christofides v0 (baseline "ingênuo", estilo código bruto gerado por IA).

Representa a implementação típica que delega tudo ao NetworkX sobre o grafo completo:
  1. Constrói um `nx.Graph` completo com N(N-1)/2 arestas, cada uma com um dicionário
     de atributos (distância recalculada em Python puro via `math.dist`).
  2. AGM via `nx.minimum_spanning_tree` (Kruskal genérico, O(E log E) = O(N² log N)).
  3. Vértices de grau ímpar da AGM.
  4. Matching perfeito de peso mínimo via `nx.min_weight_matching` sobre o subgrafo
     induzido (`G.subgraph(I)`), que é uma *view* do grafo completo.
  5. Multigrafo AGM ⊎ M, circuito euleriano via `nx.eulerian_circuit`.
  6. Shortcutting.

Gargalos esperados (a serem confirmados pelo profiling): overhead de memória dos
dicionários de aresta do NetworkX (Θ(N²) objetos Python) e Kruskal genérico.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import networkx as nx

from src.algorithms.common import TourResult, shortcut_walk
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "christofides_v0_naive"


def solve(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    coords = [tuple(p) for p in instance.coords.tolist()]
    t0 = time.perf_counter()

    with prof.profile("build_complete_graph"):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u + 1, n):
                G.add_edge(u, v, weight=math.dist(coords[u], coords[v]))

    with prof.profile("mst"):
        T = nx.minimum_spanning_tree(G, weight="weight")

    with prof.profile("odd_vertices"):
        odd = [v for v, d in T.degree() if d % 2 == 1]

    with prof.profile("min_weight_matching"):
        M = nx.min_weight_matching(G.subgraph(odd), weight="weight")

    with prof.profile("eulerian_circuit"):
        H = nx.MultiGraph()
        H.add_nodes_from(range(n))
        H.add_edges_from(T.edges())
        H.add_edges_from(M)
        walk = [u for u, _ in nx.eulerian_circuit(H, source=0)]

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
            "mst_cost": float(T.size(weight="weight")),
            "matching_cost": float(sum(G[u][v]["weight"] for u, v in M)),
            "n_odd": len(odd),
        },
    )
