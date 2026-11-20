"""Christofides v0: implementação baseline totalmente apoiada em objetos `networkx`.

Fluxo operacional executado passo a passo:
  1. `build_complete_graph`: instancia um `nx.Graph` completo com N(N-1)/2 arestas em
     dicionários Python, calculando cada distância euclidiana via `math.dist`.
  2. `mst`: extrai a Árvore Geradora Mínima (AGM) via `nx.minimum_spanning_tree` (Kruskal
     genérico sobre todas as arestas, O(N² log N)).
  3. `odd_vertices`: filtra todos os vértices com grau ímpar na AGM (cardinalidade sempre par).
  4. `min_weight_matching`: executa `nx.min_weight_matching` (Blossom de Edmonds, O(|I|³))
     sobre a view `G.subgraph(odd)` do grafo completo.
  5. `eulerian_circuit`: constrói um `nx.MultiGraph` contendo as arestas da AGM unidas às
     arestas do emparelhamento (preservando multiplicidade quando uma aresta aparece em
     ambos) e obtém o circuito euleriano via `nx.eulerian_circuit`.
  6. `shortcut`: remove repetições de vértices em Θ(N) retornando um ciclo hamiltoniano.

Complexidade: tempo O(N³) dominado pelo matching (com constantes altas na construção do
grafo completo e no Kruskal O(N² log N)); espaço Θ(N²) em objetos/dicionários Python.
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
    """Executa o Christofides v0 instrumentando todas as 6 etapas internas."""
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    # Converte as coordenadas para tuplas Python puras para consumo direto por `math.dist`.
    coords = [tuple(p) for p in instance.coords.tolist()]
    t0 = time.perf_counter()

    # Etapa 1: construção explícita do grafo completo não direcionado com N(N-1)/2 arestas.
    round_to_int = instance.round_to_int
    with prof.profile("build_complete_graph"):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u + 1, n):
                w = math.dist(coords[u], coords[v])
                if round_to_int:
                    w = float(math.floor(w + 0.5))
                G.add_edge(u, v, weight=w)

    # Etapa 2: AGM usando Kruskal genérico do NetworkX sobre o grafo completo.
    with prof.profile("mst"):
        T = nx.minimum_spanning_tree(G, weight="weight")

    # Etapa 3: seleção do subconjunto I de vértices com grau ímpar na AGM.
    with prof.profile("odd_vertices"):
        odd = [v for v, d in T.degree() if d % 2 == 1]

    # Etapa 4: emparelhamento perfeito de peso mínimo no subgrafo induzido por I.
    with prof.profile("min_weight_matching"):
        M = nx.min_weight_matching(G.subgraph(odd), weight="weight")

    # Etapa 5: união das arestas (T ⊎ M) em multigrafo (todos os graus passam a ser pares)
    # e extração de um circuito euleriano a partir do vértice 0.
    with prof.profile("eulerian_circuit"):
        H = nx.MultiGraph()
        H.add_nodes_from(range(n))
        H.add_edges_from(T.edges())
        H.add_edges_from(M)
        walk = [u for u, _ in nx.eulerian_circuit(H, source=0)]

    # Etapa 6: eliminação de vértices repetidos formando o circuito hamiltoniano final.
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
