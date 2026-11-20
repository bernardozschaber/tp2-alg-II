"""Branch-and-Bound v0 (baseline literal do slide 15 da Aula 12 — `bnb-tsp(A, n)`).

Características fiéis ao pseudocódigo dos slides (Aula 12, pp. 13-15):
  1. Busca Best-First usando fila de prioridades (`heapq`).
  2. Cada nó na fila armazena uma cópia explícita da lista de vértices visitados:
     `(bound, level, cost, path_list)`.
  3. Teste de pertinência linear em lista: `if k not in node_s`.
  4. Cálculo da função `bound(node.s U {k})` do zero para cada candidato, varrendo
     todos os N vértices e suas arestas incidentes em O(N²) por avaliação.

Gargalos propositais (para auditoria experimental na Seção 2 do artigo SBC):
  - Explosão de memória O(b^N * N) na fila de prioridades (Best-First Search).
  - Recomputação O(N²) do limite inferior a cada filho gerado.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import List, Optional

import numpy as np

from src.algorithms.common import TourResult
from src.algorithms.warm_start import run_warm_start
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "bnb_v0_naive"
EPS = 1e-9


def compute_bound_naive(
    dist: np.ndarray, path: List[int], n: int, is_integer_metric: bool = False
) -> float:
    """Calcula o limite inferior do slide 14 da Aula 12 recomputando do zero em O(N²).

    Soma para cada vértice v in V as duas menores arestas incidentes, respeitando
    as arestas já fixadas no caminho parcial `path`.
    """
    m = len(path)
    # Identifica para cada vértice quais vizinhos já estão fixados no caminho parcial
    fixed_neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(m - 1):
        u = path[i]
        v = path[i + 1]
        fixed_neighbors[u].append(v)
        fixed_neighbors[v].append(u)
    if m == n + 1 and path[-1] == path[0]:
        # Ciclo fechado: a última aresta (path[n-1], 0) já foi contada no loop acima
        pass

    total_sum = 0.0
    for v in range(n):
        fixed = fixed_neighbors[v]
        if len(fixed) >= 2:
            total_sum += float(dist[v, fixed[0]]) + float(dist[v, fixed[1]])
        elif len(fixed) == 1:
            u_fixed = fixed[0]
            best_other = float("inf")
            for w in range(n):
                if w != v and w != u_fixed:
                    c = float(dist[v, w])
                    if c < best_other:
                        best_other = c
            total_sum += float(dist[v, u_fixed]) + best_other
        else:
            min1 = float("inf")
            min2 = float("inf")
            for w in range(n):
                if w != v:
                    c = float(dist[v, w])
                    if c < min1:
                        min2 = min1
                        min1 = c
                    elif c < min2:
                        min2 = c
            total_sum += min1 + min2

    half = total_sum / 2.0
    return float(math.ceil(half - EPS)) if is_integer_metric else half


def solve(
    instance: TSPInstance,
    warm_start: Optional[str] = "cold_start",
    timeout_s: float = 1800.0,
    use_gpu: bool = False,
    profiler: Optional[ProfilerSession] = None,
) -> TourResult:
    """Executa o Branch-and-Bound v0 (literal do slide 15 da Aula 12)."""
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    dist = instance.dist_matrix
    is_int = instance.round_to_int
    t_start = time.perf_counter()

    # 1. Inicialização do limite superior (Warm-Start)
    best = float("inf")
    best_tour: List[int] = []
    initial_ub = float("inf")
    warmup_time_s = 0.0

    if warm_start is not None:
        t_ws = time.perf_counter()
        with prof.profile("warm_start"):
            ws_res = run_warm_start(warm_start, instance)
        warmup_time_s = time.perf_counter() - t_ws
        if ws_res is not None:
            best = float(ws_res.cost)
            best_tour = list(ws_res.tour)
            initial_ub = best

    # 2. Busca Branch-and-Bound Best-First (Slide 15, Aula 12)
    nodes_expanded = 0
    nodes_pruned_gen = 0
    nodes_pruned_visit = 0
    max_depth = 1
    max_queue_size = 1
    timed_out = False
    tie_counter = 0

    t_bnb_start = time.perf_counter()
    with prof.profile("bnb_tree_search"):
        root_bound = compute_bound_naive(dist, [0], n, is_int)
        # Tupla no heap: (bound, tie_breaker, level, cost, path_list)
        # Note que no slide `root = (bound([0]), 0, 0, [0])`, aqui `level` é nº de vértices na rota parcial
        queue = [(root_bound, tie_counter, 1, 0.0, [0])]

        while queue:
            if (nodes_expanded & 63) == 0 and (time.perf_counter() - t_start) >= timeout_s:
                timed_out = True
                break

            if len(queue) > max_queue_size:
                max_queue_size = len(queue)

            node_bound, _, node_level, node_cost, node_s = heapq.heappop(queue)

            if node_level > max_depth:
                max_depth = node_level

            # Folha completa (ciclo fechado retornando a 0: level == n + 1)
            if node_level > n:
                if node_cost + EPS < best:
                    best = node_cost
                    best_tour = node_s[:-1]
                continue

            if node_bound >= best - EPS:
                nodes_pruned_visit += 1
                continue

            nodes_expanded += 1
            last_v = node_s[-1]

            if node_level < n:
                for k in range(1, n):
                    if k not in node_s:
                        edge_cost = float(dist[last_v, k])
                        if math.isinf(edge_cost):
                            continue
                        new_s = node_s + [k]  # cópia explícita da lista
                        child_bound = compute_bound_naive(dist, new_s, n, is_int)
                        if child_bound < best - EPS:
                            tie_counter += 1
                            heapq.heappush(
                                queue,
                                (child_bound, tie_counter, node_level + 1, node_cost + edge_cost, new_s),
                            )
                        else:
                            nodes_pruned_gen += 1
            else:
                # node_level == n: todos os vértices foram visitados, resta fechar o ciclo com 0
                edge_cost = float(dist[last_v, 0])
                if not math.isinf(edge_cost):
                    total_cost = node_cost + edge_cost
                    if total_cost < best - EPS:
                        tie_counter += 1
                        heapq.heappush(
                            queue,
                            (total_cost, tie_counter, node_level + 1, total_cost, node_s + [0]),
                        )
                    else:
                        nodes_pruned_gen += 1

    bnb_search_time_s = time.perf_counter() - t_bnb_start
    total_elapsed_s = time.perf_counter() - t_start

    return TourResult(
        algorithm=ALGORITHM_NAME,
        instance=instance.name,
        n=n,
        tour=best_tour,
        cost=best,
        elapsed_s=total_elapsed_s,
        profile=prof.summary_dict(),
        extra={
            "version": "v0_naive",
            "warm_start": warm_start or "none",
            "initial_ub": initial_ub,
            "warmup_time_s": warmup_time_s,
            "bnb_search_time_s": bnb_search_time_s,
            "nodes_expanded": nodes_expanded,
            "nodes_pruned": nodes_pruned_gen + nodes_pruned_visit,
            "nodes_pruned_at_generation": nodes_pruned_gen,
            "nodes_pruned_at_visit": nodes_pruned_visit,
            "max_depth_reached": max_depth,
            "max_queue_size": max_queue_size,
            "timed_out": timed_out,
            "status": "NA" if timed_out else "OPTIMAL",
            "use_gpu": False,
            "gpu_backend": "cpu_sequential",
        },
    )
