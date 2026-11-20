"""Branch-and-Bound v1 (1ª Otimização — Aceleração via GPU com `--use-gpu`).

Mantém a estratégia Best-First Search (fila de prioridades `heapq` com listas de caminho)
da versão `v0_naive`, mas substitui o recálculo sequencial O(N²) do `bound` em Python
pela avaliação paralelizada em lote na GPU (`GPUBackend.batch_evaluate_children_bounds`,
via CuPy/CUDA quando `--use-gpu` está ativo nas máquinas do laboratório, ou NumPy
vetorizado como fallback).

Objetivo na progressão gradual:
  - Acelerar imediatamente a avaliação dos limites inferiores para viabilizar a
    experimentação.
  - Evidenciar que, mesmo acelerando o cálculo do `bound` em lote na GPU, a fila de
    prioridades Best-First ainda acumula um número exponencial de estados em memória
    (`max_queue_size`), motivando a 2ª otimização (`v2_dfs_memory`).
"""

from __future__ import annotations

import heapq
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from src.algorithms.common import TourResult
from src.algorithms.warm_start import run_warm_start
from src.core.gpu_utils import get_backend
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "bnb_v1_gpu"
EPS = 1e-9


def solve(
    instance: TSPInstance,
    warm_start: Optional[str] = "cold_start",
    timeout_s: float = 1800.0,
    use_gpu: bool = True,
    batch_size: int = 128,
    profiler: Optional[ProfilerSession] = None,
) -> TourResult:
    """Executa o Branch-and-Bound v1 (Best-First com avaliação de bounds em lote na GPU)."""
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    dist = instance.dist_matrix
    is_int = instance.round_to_int
    backend = get_backend(use_gpu=use_gpu)
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

    # 2. Pré-computação das menores arestas no dispositivo (GPU / Vetorizado)
    with prof.profile("bnb_gpu_precompute"):
        dist_dev, min1_val_dev, min1_idx_dev, min2_val_dev = backend.precompute_min_edges_device(
            dist
        )
        pair_sums_cpu = backend.to_cpu(min1_val_dev + min2_val_dev)
        root_total_sum = float(np.sum(pair_sums_cpu))
        root_bound = (
            float(math.ceil(root_total_sum / 2.0 - EPS))
            if is_int
            else root_total_sum / 2.0
        )

    # 3. Busca Best-First com expansão e avaliação de bounds em lote na GPU
    nodes_expanded = 0
    nodes_pruned_gen = 0
    nodes_pruned_visit = 0
    max_depth = 1
    max_queue_size = 1
    timed_out = False
    tie_counter = 0

    t_bnb_start = time.perf_counter()
    with prof.profile("bnb_tree_search"):
        # Estado na fila: (bound, tie_breaker, level, cost, unvis_sum, path_list)
        root_unvis_sum = root_total_sum - float(pair_sums_cpu[0])
        queue: List[Tuple[float, int, int, float, float, List[int]]] = [
            (root_bound, tie_counter, 1, 0.0, root_unvis_sum, [0])
        ]

        while queue:
            if (time.perf_counter() - t_start) >= timeout_s:
                timed_out = True
                break

            if len(queue) > max_queue_size:
                max_queue_size = len(queue)

            # Retira um lote de até `batch_size` nós promissores da fila Best-First
            active_parents: List[Tuple[float, int, float, float, List[int]]] = []
            best_improved_in_batch = False
            while queue and len(active_parents) < batch_size:
                node_bound, _, node_level, node_cost, node_unvis_sum, node_s = heapq.heappop(queue)
                if node_level > max_depth:
                    max_depth = node_level

                if node_level > n:
                    if node_cost + EPS < best:
                        best = node_cost
                        best_tour = node_s[:-1]
                        best_improved_in_batch = True
                    continue

                if node_bound >= best - EPS:
                    nodes_pruned_visit += 1
                    continue

                if node_level == n:
                    # Último passo: fechar o ciclo voltando para 0
                    nodes_expanded += 1
                    last_v = node_s[-1]
                    total_cost = node_cost + float(dist[last_v, 0])
                    if total_cost < best - EPS:
                        best = total_cost
                        best_tour = list(node_s)
                        best_improved_in_batch = True
                        if n + 1 > max_depth:
                            max_depth = n + 1
                    else:
                        nodes_pruned_gen += 1
                else:
                    active_parents.append((node_bound, node_level, node_cost, node_unvis_sum, node_s))

            if best_improved_in_batch and active_parents:
                # Re-filtra nós já retirados no mesmo lote cujo bound ficou >= novo `best`
                still_promising: List[Tuple[float, int, float, float, List[int]]] = []
                for p_item in active_parents:
                    if p_item[0] < best - EPS:
                        still_promising.append(p_item)
                    else:
                        nodes_pruned_visit += 1
                active_parents = still_promising

            if not active_parents:
                continue

            nodes_expanded += len(active_parents)

            # Constrói os tensores planos de todas as transições filhas do lote para a GPU
            parent_indices: List[int] = []
            first_v_list: List[int] = []
            last_v_list: List[int] = []
            cand_k_list: List[int] = []
            p_cost_list: List[float] = []
            unvis_sum_list: List[float] = []

            for p_idx, (_, p_level, p_cost, p_unvis, p_s) in enumerate(active_parents):
                visited_set = set(p_s)
                v1 = p_s[1] if p_level >= 2 else -1
                u = p_s[-1]
                for k in range(1, n):
                    if k not in visited_set:
                        parent_indices.append(p_idx)
                        first_v_list.append(v1)
                        last_v_list.append(u)
                        cand_k_list.append(k)
                        p_cost_list.append(p_cost)
                        unvis_sum_list.append(p_unvis)

            if not cand_k_list:
                continue

            # Avaliação em lote na GPU (ou NumPy vetorizado)
            new_costs, bounds = backend.batch_evaluate_children_bounds(
                dist_dev=dist_dev,
                min1_val=min1_val_dev,
                min1_idx=min1_idx_dev,
                min2_val=min2_val_dev,
                first_vertices=np.asarray(first_v_list, dtype=np.int64),
                last_vertices=np.asarray(last_v_list, dtype=np.int64),
                candidates=np.asarray(cand_k_list, dtype=np.int64),
                parent_costs=np.asarray(p_cost_list, dtype=np.float64),
                unvisited_sums=np.asarray(unvis_sum_list, dtype=np.float64),
                is_integer_metric=is_int,
            )

            # Filtra os filhos promissores (bounds < best - EPS)
            keep_mask = bounds < (best - EPS)
            num_total = len(bounds)
            surviving_indices = np.flatnonzero(keep_mask)
            nodes_pruned_gen += int(num_total - len(surviving_indices))

            for idx in surviving_indices:
                i = int(idx)
                p_idx = parent_indices[i]
                _, p_level, _, p_unvis, p_s = active_parents[p_idx]
                k = cand_k_list[i]
                c_bound = float(bounds[i])
                c_cost = float(new_costs[i])
                c_unvis = p_unvis - float(pair_sums_cpu[k])
                tie_counter += 1
                heapq.heappush(
                    queue,
                    (c_bound, tie_counter, p_level + 1, c_cost, c_unvis, p_s + [k]),
                )

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
            "version": "v1_gpu",
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
            "use_gpu": use_gpu,
            "gpu_backend": backend.backend_name,
        },
    )
