"""Branch-and-Bound v0: busca Best-First com heap, cópia de listas e bound O(N²).

Funcionamento operacional:
  - Mantém uma fila de prioridades mínima (`heapq`) ordenada pelo limite inferior `bound`
    de cada estado parcial `(bound, tie_counter, level, cost, path_list)`.
  - `path_list` é uma lista Python explícita duplicada (`node_s + [k]`) a cada transição,
    e o teste de pertinência `k not in node_s` faz varredura linear O(N).
  - `compute_bound_naive` reconstrói do zero, para cada candidato gerado, a soma das duas
    menores arestas incidentes a cada um dos N vértices (respeitando arestas já fixadas
    em `path`), com custo O(N²) por avaliação.
  - A fila cresce proporcionalmente à largura da árvore de busca aberta, podendo atingir
    complexidade espacial exponencial O(b^N · N) antes da convergência.
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
# Tolerância numérica para evitar podas indevidas ou laços por erro de ponto flutuante.
EPS = 1e-9


def compute_bound_naive(
    dist: np.ndarray, path: List[int], n: int, is_integer_metric: bool = False
) -> float:
    """Computa do zero o limite inferior de relaxação de grau 2 para o caminho parcial `path`.

    Regras aplicadas vértice a vértice (`v in 0..n-1`):
      - 2 arestas já fixadas em `path` (vértice interno): soma exatamente essas 2 arestas.
      - 1 aresta fixada em `path` (extremidade inicial ou final): soma a aresta fixada com
        a menor outra aresta incidente em `v` (excluindo `v` e o vizinho já fixado).
      - 0 arestas fixadas (vértice ainda não visitado): soma as 2 menores arestas incidentes.

    Como cada aresta do circuito hamiltoniano incide em exatamente 2 vértices, divide-se o
    somatório total por 2 (aplicando teto `ceil(x - EPS)` apenas quando os pesos são inteiros).
    Complexidade: O(N²) por chamada.
    """
    m = len(path)
    # Mapeia, para cada vértice, quais vizinhos já estão conectados a ele no caminho parcial.
    fixed_neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(m - 1):
        u = path[i]
        v = path[i + 1]
        fixed_neighbors[u].append(v)
        fixed_neighbors[v].append(u)

    total_sum = 0.0
    for v in range(n):
        fixed = fixed_neighbors[v]
        if len(fixed) >= 2:
            # Vértice interno do caminho: ambas as arestas incidentes já estão definidas.
            total_sum += float(dist[v, fixed[0]]) + float(dist[v, fixed[1]])
        elif len(fixed) == 1:
            # Extremidade do caminho: busca linear pela menor segunda aresta distinta da fixada.
            u_fixed = fixed[0]
            best_other = float("inf")
            for w in range(n):
                if w != v and w != u_fixed:
                    c = float(dist[v, w])
                    if c < best_other:
                        best_other = c
            total_sum += float(dist[v, u_fixed]) + best_other
        else:
            # Vértice livre: busca linear pelas duas menores arestas incidentes em `v`.
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
    """Executa a busca exata Branch-and-Bound v0 com limite superior inicial e timeout."""
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    dist = instance.dist_matrix
    is_int = instance.round_to_int
    t_start = time.perf_counter()

    # 1. Inicialização do limite superior (upper bound `best`) via heurística de warm-start.
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

    # 2. Contadores empíricos da árvore de busca.
    nodes_expanded = 0
    nodes_pruned_gen = 0    # Filhos descartados antes de entrar na fila (bound >= best)
    nodes_pruned_visit = 0  # Nós retirados da fila cujo bound ficou >= best após melhorias
    max_depth = 1
    max_queue_size = 1
    timed_out = False
    # Contador monotônico de desempate no heap para evitar comparação direta entre listas Python.
    tie_counter = 0

    t_bnb_start = time.perf_counter()
    with prof.profile("bnb_tree_search"):
        root_bound = compute_bound_naive(dist, [0], n, is_int)
        # Estrutura de cada entrada no min-heap: (bound, tie_counter, level, cost, path_list).
        # `level` contabiliza quantos vértices estão atualmente em `path_list` (raiz = 1).
        queue = [(root_bound, tie_counter, 1, 0.0, [0])]

        while queue:
            # Checa o timeout a cada 64 expansões para reduzir chamadas ao relógio do SO.
            if (nodes_expanded & 63) == 0 and (time.perf_counter() - t_start) >= timeout_s:
                timed_out = True
                break

            if len(queue) > max_queue_size:
                max_queue_size = len(queue)

            node_bound, _, node_level, node_cost, node_s = heapq.heappop(queue)

            if node_level > max_depth:
                max_depth = node_level

            # Caso base: `level == n + 1` indica ciclo fechado `[0, ..., v_{n-1}, 0]`.
            if node_level > n:
                if node_cost + EPS < best:
                    best = node_cost
                    best_tour = node_s[:-1]
                continue

            # Poda na visita: `best` diminuiu desde que este nó foi enfileirado.
            if node_bound >= best - EPS:
                nodes_pruned_visit += 1
                continue

            nodes_expanded += 1
            last_v = node_s[-1]

            if node_level < n:
                # Expande todos os vértices ainda não presentes em `node_s`.
                for k in range(1, n):
                    if k not in node_s:
                        edge_cost = float(dist[last_v, k])
                        if math.isinf(edge_cost):
                            continue
                        # Aloca uma nova lista copiando todo o prefixo atual + candidato `k`.
                        new_s = node_s + [k]
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
                # Todos os N vértices foram visitados: avalia a aresta de fechamento `(last_v, 0)`.
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

            if len(queue) > max_queue_size:
                max_queue_size = len(queue)

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
