"""Twice-around-the-tree v0: implementação baseline literal sobre objetos `networkx`.

Fluxo operacional executado passo a passo (seguindo diretamente a prova do fator 2):
  1. `build_complete_graph`: instancia um `nx.Graph` completo com N(N-1)/2 arestas em
     dicionários Python, calculando cada distância euclidiana via `math.dist`.
  2. `mst`: extrai a Árvore Geradora Mínima (AGM) via `nx.minimum_spanning_tree` (Kruskal
     genérico sobre todas as N(N-1)/2 arestas, O(N² log N)).
  3. `eulerian_circuit`: duplica explicitamente todas as N-1 arestas da AGM em um
     `nx.MultiGraph` (tornando todos os graus pares, com peso total 2·c(T)) e extrai o
     circuito euleriano fechado de 2(N-1) arestas via `nx.eulerian_circuit`.
  4. `shortcut`: remove visitas repetidas em Θ(N) via tabela hash (`shortcut_walk`),
     fechando o circuito hamiltoniano final com garantia c(H) <= 2·c(T) <= 2·OPT.

Complexidade: tempo O(N² log N) dominado pelo Kruskal sobre o grafo completo (com altas
constantes na alocação de dicionários do `nx.Graph` e do `nx.MultiGraph`); espaço Θ(N²)
em objetos Python.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import networkx as nx

from src.algorithms.common import TourResult, shortcut_walk
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "twice_around_v0_naive"


def solve(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    """Executa o Twice-around-the-tree v0 instrumentando todas as 4 etapas internas."""
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    coords = [tuple(p) for p in instance.coords.tolist()]
    round_to_int = instance.round_to_int
    t0 = time.perf_counter()

    # Etapa 1: construção explícita do grafo completo não direcionado com N(N-1)/2 arestas.
    with prof.profile("build_complete_graph"):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u + 1, n):
                w = math.dist(coords[u], coords[v])
                if round_to_int:
                    # Padrão de arredondamento inteiro da TSPLIB: nint(d) = floor(d + 0.5).
                    w = float(math.floor(w + 0.5))
                G.add_edge(u, v, weight=w)

    # Etapa 2: AGM usando Kruskal genérico do NetworkX sobre todas as arestas do grafo completo.
    with prof.profile("mst"):
        T = nx.minimum_spanning_tree(G, weight="weight")

    # Etapa 3: duplicação explícita das arestas de T em um MultiGraph (2T) e circuito euleriano.
    with prof.profile("eulerian_circuit"):
        H = nx.MultiGraph()
        H.add_nodes_from(range(n))
        mst_edges = list(T.edges())
        # Inserir cada aresta duas vezes torna todos os graus pares, garantindo circuito euleriano de custo 2·c(T).
        H.add_edges_from(mst_edges)
        H.add_edges_from(mst_edges)
        if n <= 1:
            # `nx.eulerian_circuit` retorna iterador vazio em grafos sem arestas (N <= 1).
            walk = list(range(n))
        else:
            walk = [u for u, _ in nx.eulerian_circuit(H, source=0)]

    # Etapa 4: eliminação de vértices repetidos formando o ciclo hamiltoniano final.
    with prof.profile("shortcut"):
        tour = shortcut_walk(walk, start=0)

    elapsed = time.perf_counter() - t0
    mst_max_degree = max((int(d) for _, d in T.degree()), default=0)
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
            "mst_max_degree": mst_max_degree,
        },
    )
