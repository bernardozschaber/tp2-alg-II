"""Christofides v1: implementação otimizada sobre matriz NumPy, Prim denso e Hierholzer.

Operação empírica e estruturas adotadas em relação à versão `v0_naive`:
  1. Elimina a instanciação de `nx.Graph` completo (N(N-1)/2 dicionários Python), operando
     diretamente sobre a matriz contígua `dist_matrix` `(N, N)` da instância.
  2. `prim_dense`: computa a AGM em Θ(N²) estrito e memória auxiliar Θ(N) usando vetores
     NumPy (`in_tree`, `key`, `parent`, `degree`), sem fila de prioridades.
  3. `odd_vertices`: extrai o conjunto I diretamente via `np.flatnonzero(degree % 2 == 1)`.
  4. `min_weight_perfect_matching`: instancia um `nx.Graph` reduzido contendo APENAS os
     |I| vértices de grau ímpar e delega o matching perfeito mínimo ao algoritmo Blossom
     (`nx.min_weight_matching`, O(|I|³)).
  5. `hierholzer`: extrai o circuito euleriano sobre listas de adjacência compactas do
     multigrafo T ⊎ M em tempo linear Θ(N) usando pilha explícita e ponteiro por vértice.
  6. `shortcut_walk`: elimina vértices duplicados em Θ(N).
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
    """Computa a AGM em grafo completo denso pelo algoritmo de Prim sem heap.

    Estruturas de dados (todas de tamanho N):
      - `in_tree[v]`: indica se o vértice `v` já foi incorporado à AGM.
      - `key[v]`    : menor peso conhecido ligando `v` a algum vértice em `in_tree`
                      (definido como +inf após `v` entrar na árvore para permitir `argmin`
                      direto sem alocar máscaras temporárias).
      - `parent[v]` : vértice já presente na árvore que realiza `key[v]`.
      - `degree[v]` : grau acumulado de `v` na AGM (evita nova passagem para filtrar ímpares).

    Complexidade: Θ(N²) em tempo (N iterações de varredura linear vetorizada) e Θ(N) espaço.
    """
    n = dist.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    key[root] = 0.0

    edges: List[Tuple[int, int]] = []
    degree = np.zeros(n, dtype=np.int64)

    for _ in range(n):
        # Seleciona o vértice fora da árvore com menor aresta de corte.
        u = int(np.argmin(key))
        in_tree[u] = True
        # Marca `key[u] = inf` in-place para que `u` nunca volte a ser escolhido por `argmin`.
        key[u] = np.inf

        if parent[u] >= 0:
            p = int(parent[u])
            edges.append((p, u))
            degree[p] += 1
            degree[u] += 1

        # Relaxamento vetorizado: atualiza as chaves dos vértices ainda fora da árvore.
        row = dist[u]
        improve = (~in_tree) & (row < key)
        key[improve] = row[improve]
        parent[improve] = u

    return edges, degree


def min_weight_perfect_matching(dist: np.ndarray, odd: List[int]) -> List[Tuple[int, int]]:
    """Calcula o emparelhamento perfeito de peso mínimo sobre o subgrafo completo induzido por `odd`.

    Cria um `nx.Graph` restrito exclusivamente aos |I| vértices de grau ímpar (|I| pares de
    arestas = |I|(|I|-1)/2) e invoca `nx.min_weight_matching`, que garante cardinalidade
    máxima (emparelhamento perfeito em grafo completo de ordem par) via algoritmo Blossom.
    Complexidade: O(|I|³) em tempo e O(|I|²) em memória.
    """
    G = nx.Graph()
    k = len(odd)
    for i in range(k):
        u = odd[i]
        for j in range(i + 1, k):
            v = odd[j]
            G.add_edge(u, v, weight=float(dist[u, v]))
    return [(int(u), int(v)) for u, v in nx.min_weight_matching(G, weight="weight")]


def hierholzer(n: int, edges: List[Tuple[int, int]], start: int = 0) -> List[int]:
    """Extrai um circuito euleriano em multigrafo conexo de graus pares (Hierholzer iterativo).

    Representação e mecânica:
      - Cada aresta recebe um identificador único `eid in 0..|E|-1` para suportar arestas
        paralelas (quando uma mesma aresta pertence simultaneamente à AGM e ao matching).
      - `used[eid]` controla se a aresta já foi atravessada em qualquer um dos sentidos.
      - `ptr[u]` avança monotonicamente na lista de adjacência `adj[u]`, garantindo que cada
        entrada de adjacência seja inspecionada no máximo uma vez em toda a execução.
      - Quando `u` não tem mais arestas incidentes livres, é desempilhado para `circuit`;
        ao final, inverter `circuit` fornece o circuito na ordem direta a partir de `start`.

    Complexidade: Θ(N + |E|) = Θ(N) em tempo e espaço (pois |E| = (N-1) + |I|/2 < 1.5·N).
    """
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
        # Descarta arestas já consumidas a partir do ponteiro atual de `u`.
        while ptr[u] < len(adj[u]) and used[adj[u][ptr[u]][1]]:
            ptr[u] += 1
        if ptr[u] == len(adj[u]):
            # Vértice saturado: consolida na saída do circuito euleriano.
            circuit.append(stack.pop())
        else:
            # Consome a próxima aresta livre e avança na pilha.
            v, eid = adj[u][ptr[u]]
            used[eid] = True
            stack.append(v)

    circuit.reverse()
    return circuit


def solve(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    """Executa o Christofides v1 com profiling isolado por subrotina."""
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    dist = instance.dist_matrix
    t0 = time.perf_counter()

    # Etapa 1: AGM densa por Prim vetorizado em Θ(N²).
    with prof.profile("mst"):
        mst_edges, degree = prim_dense(dist, root=0)

    # Etapa 2: filtragem linear dos vértices de grau ímpar na AGM.
    with prof.profile("odd_vertices"):
        odd = [int(v) for v in np.flatnonzero(degree % 2 == 1)]

    # Etapa 3: emparelhamento perfeito de peso mínimo sobre o subgrafo induzido por `odd`.
    with prof.profile("min_weight_matching"):
        matching = min_weight_perfect_matching(dist, odd)

    # Etapa 4: circuito euleriano sobre o multigrafo T ⊎ M em Θ(N).
    with prof.profile("eulerian_circuit"):
        walk = hierholzer(n, mst_edges + matching, start=0)

    # Etapa 5: remoção de repetições (shortcutting) gerando o ciclo hamiltoniano final.
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
