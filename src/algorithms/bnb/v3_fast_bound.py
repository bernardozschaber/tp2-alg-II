"""Branch-and-Bound v3 (3ª Otimização — Tempo Assintótico O(1), Bitmask e Quebra de Simetria + GPU).

Acumula as melhorias de hardware (`v1_gpu`) e memória Θ(N) (`v2_dfs_memory`) e elimina os
gargalos de tempo por nó expandido:
  1. Atualização Incremental em Θ(1) do Limite Inferior:
     Em vez de reconstruir vetores ou varrer N vértices a cada candidato, mantém a soma
     residual dos vértices não visitados (`unvis_sum`) e atualiza o `bound` em 4 operações
     aritméticas escalares O(1).
  2. Representação de Visitados por Bitmask Inteiro (`unvisited_mask`):
     Itera apenas sobre os bits ativos (`lowbit = rem & -rem`) usando tabela de lookup O(1),
     eliminando varreduras lineares sobre vértices já visitados.
  3. Quebra de Simetria Reversa por Bitmask em Nível Interno:
     Para eliminar ciclos reversos redundantes (0 -> v1 -> ... -> v_{N-1} -> 0 equivalentes a
     0 -> v_{N-1} -> ... -> v1 -> 0), impõe canonicamente v1 < v_{N-1}. Em representação
     bitwise, se em qualquer nível intermediário `unvisited_mask < (1 << v1)`, significa
     que todos os vértices restantes possuem índice menor que `v1` (logo é impossível
     terminar em v_{N-1} > v1), permitindo podar a subárvore inteira em O(1).
  4. Pré-expansão e Ordenação de Fronteira em Lote na GPU (`--use-gpu`):
     Quando `use_gpu=True`, avalia a fronteira inicial de nível 2 ((N-1)(N-2) caminhos
     0 -> v1 -> v2) em paralelo via `GPUBackend`, ordenando as subárvores mais promissoras
     globalmente antes de disparar o núcleo recursivo O(1).
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np

from src.algorithms.common import TourResult
from src.algorithms.warm_start import run_warm_start
from src.core.gpu_utils import get_backend
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "bnb_v3_fast_bound"
EPS = 1e-9


def solve(
    instance: TSPInstance,
    warm_start: Optional[str] = "cold_start",
    timeout_s: float = 1800.0,
    use_gpu: bool = False,
    profiler: Optional[ProfilerSession] = None,
) -> TourResult:
    """Executa o Branch-and-Bound v3 (Bitmask + Bound Incremental Θ(1) + Simetria + GPU)."""
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

    # 2. Pré-computação para atualização incremental Θ(1) e tabelas bitwise
    with prof.profile("bnb_precompute_o1"):
        dist_dev, min1_val_dev, min1_idx_dev, min2_val_dev = backend.precompute_min_edges_device(
            dist
        )
        min1_val_cpu = backend.to_cpu(min1_val_dev)
        min1_idx_cpu = backend.to_cpu(min1_idx_dev)
        min2_val_cpu = backend.to_cpu(min2_val_dev)
        pair_sums_cpu = min1_val_cpu + min2_val_cpu
        root_total_sum = float(np.sum(pair_sums_cpu))

        # Listas nativas em Python para acesso escalar O(1) sem overhead de boxing do NumPy
        dist_rows: List[List[float]] = dist.tolist()
        min1_v: List[float] = min1_val_cpu.tolist()
        min1_i: List[int] = [int(x) for x in min1_idx_cpu.tolist()]
        min2_v: List[float] = min2_val_cpu.tolist()
        pair_s: List[float] = pair_sums_cpu.tolist()
        bit_to_vertex = {1 << k: k for k in range(n)}

    # 3. Núcleo DFS In-Place com Bitmask, Bound Incremental O(1) e Quebra de Simetria
    path = [0] * n
    nodes_expanded = 0
    nodes_pruned_gen = 0
    nodes_pruned_visit = 0
    max_depth = 1
    max_stack_depth = 1
    timed_out = False

    def dfs_o1(
        level: int,
        u: int,
        v1: int,
        v1_bit: int,
        root_other: float,
        cost: float,
        unvis_sum: float,
        unvis_mask: int,
        stack_depth: int,
    ) -> None:
        nonlocal best, best_tour, nodes_expanded, nodes_pruned_gen, nodes_pruned_visit
        nonlocal max_depth, max_stack_depth, timed_out

        if timed_out:
            return
        if (nodes_expanded & 255) == 0 and (time.perf_counter() - t_start) >= timeout_s:
            timed_out = True
            return

        if level > max_depth:
            max_depth = level
        if stack_depth > max_stack_depth:
            max_stack_depth = stack_depth

        # Caso base: todos os N vértices foram visitados (unvis_mask == 0)
        if unvis_mask == 0:
            nodes_expanded += 1
            # Pela quebra de simetria, u (que é v_{N-1}) já satisfaz u > v1
            total_cost = cost + dist_rows[u][0]
            if total_cost < best - EPS:
                best = total_cost
                best_tour = list(path)
                if n + 1 > max_depth:
                    max_depth = n + 1
            else:
                nodes_pruned_gen += 1
            return

        # Poda de simetria em nível interno O(1):
        # Se todos os vértices não visitados têm índice < v1, é impossível terminar com v_{N-1} > v1
        if unvis_mask < v1_bit:
            nodes_pruned_gen += 1
            return

        nodes_expanded += 1
        dist_u = dist_rows[u]

        # Avalia todos os filhos não visitados em Θ(1) por candidato
        children: List[Tuple[float, int, float, float, int]] = []
        rem = unvis_mask
        is_last_choice = level == n - 1

        while rem:
            lowbit = rem & -rem
            rem ^= lowbit
            k = bit_to_vertex[lowbit]

            if is_last_choice and k < v1:
                nodes_pruned_gen += 1
                continue

            c_cost = cost + dist_u[k]
            c_unvis = unvis_sum - pair_s[k]
            k_other = min2_v[k] if min1_i[k] == u else min1_v[k]
            double_sum = 2.0 * c_cost + root_other + k_other + c_unvis
            c_bound = float(math.ceil(0.5 * double_sum - EPS)) if is_int else 0.5 * double_sum

            if c_bound < best - EPS:
                children.append((c_bound, k, c_cost, c_unvis, lowbit))
            else:
                nodes_pruned_gen += 1

        if not children:
            return

        # Ordena localmente os filhos sobreviventes pelo menor bound (Best-Child-First)
        children.sort(key=lambda item: item[0])

        for c_bound, k, c_cost, c_unvis, k_bit in children:
            if c_bound >= best - EPS:
                nodes_pruned_visit += 1
                continue
            path[level] = k
            dfs_o1(
                level + 1,
                k,
                v1,
                v1_bit,
                root_other,
                c_cost,
                c_unvis,
                unvis_mask ^ k_bit,
                stack_depth + 1,
            )
            if timed_out:
                return

    t_bnb_start = time.perf_counter()
    with prof.profile("bnb_tree_search"):
        full_unvis_mask = ((1 << n) - 1) ^ 1
        root_unvis_sum = root_total_sum - pair_s[0]

        if use_gpu and n >= 4:
            # Pré-expansão em lote na GPU da fronteira de nível 2 (0 -> v1 -> v2)
            nodes_expanded += 1  # raiz 0 expandida
            v1_list: List[int] = []
            v2_list: List[int] = []
            p_cost_list: List[float] = []
            unvis_sum_list: List[float] = []

            for v1 in range(1, n - 1):  # v1 < n-1 pela quebra de simetria
                c1 = dist_rows[0][v1]
                u1 = root_unvis_sum - pair_s[v1]
                nodes_expanded += 1
                for v2 in range(1, n):
                    if v2 == v1:
                        continue
                    v1_list.append(v1)
                    v2_list.append(v2)
                    p_cost_list.append(c1)
                    unvis_sum_list.append(u1)

            v1_arr = np.asarray(v1_list, dtype=np.int64)
            v2_arr = np.asarray(v2_list, dtype=np.int64)
            new_costs, bounds = backend.batch_evaluate_children_bounds(
                dist_dev=dist_dev,
                min1_val=min1_val_dev,
                min1_idx=min1_idx_dev,
                min2_val=min2_val_dev,
                first_vertices=v1_arr,
                last_vertices=v1_arr,
                candidates=v2_arr,
                parent_costs=np.asarray(p_cost_list, dtype=np.float64),
                unvisited_sums=np.asarray(unvis_sum_list, dtype=np.float64),
                is_integer_metric=is_int,
            )

            promising_idx = np.flatnonzero(bounds < (best - EPS))
            nodes_pruned_gen += int(len(bounds) - len(promising_idx))
            order = promising_idx[np.argsort(bounds[promising_idx])]
            for pos, idx in enumerate(order):
                i = int(idx)
                b_val = float(bounds[i])
                if b_val >= best - EPS:
                    nodes_pruned_visit += int(len(order) - pos)
                    break
                v1 = v1_list[i]
                v2 = v2_list[i]
                c2 = float(new_costs[i])
                u2 = unvis_sum_list[i] - pair_s[v2]
                v1_bit = 1 << v1
                v2_bit = 1 << v2
                root_other = min2_v[0] if min1_i[0] == v1 else min1_v[0]

                path[1] = v1
                path[2] = v2
                dfs_o1(
                    3,
                    v2,
                    v1,
                    v1_bit,
                    root_other,
                    c2,
                    u2,
                    full_unvis_mask ^ v1_bit ^ v2_bit,
                    3,
                )
                if timed_out:
                    break
        else:
            # Expansão da raiz (nível 1: escolha de v1 in {1, ..., n-2})
            nodes_expanded += 1
            lvl1_children: List[Tuple[float, int, float, float, float, int]] = []
            for v1 in range(1, n - 1 if n >= 3 else n):
                c1 = dist_rows[0][v1]
                u1 = root_unvis_sum - pair_s[v1]
                root_other = min2_v[0] if min1_i[0] == v1 else min1_v[0]
                v1_other = min2_v[v1] if min1_i[v1] == 0 else min1_v[v1]
                double_sum = 2.0 * c1 + root_other + v1_other + u1
                b1 = float(math.ceil(0.5 * double_sum - EPS)) if is_int else 0.5 * double_sum
                if b1 < best - EPS:
                    lvl1_children.append((b1, v1, c1, u1, root_other, 1 << v1))
                else:
                    nodes_pruned_gen += 1

            lvl1_children.sort(key=lambda item: item[0])
            for b1, v1, c1, u1, root_other, v1_bit in lvl1_children:
                if b1 >= best - EPS:
                    nodes_pruned_visit += 1
                    continue
                path[1] = v1
                dfs_o1(
                    2,
                    v1,
                    v1,
                    v1_bit,
                    root_other,
                    c1,
                    u1,
                    full_unvis_mask ^ v1_bit,
                    2,
                )
                if timed_out:
                    break

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
            "version": "v3_fast_bound",
            "warm_start": warm_start or "none",
            "initial_ub": initial_ub,
            "warmup_time_s": warmup_time_s,
            "bnb_search_time_s": bnb_search_time_s,
            "nodes_expanded": nodes_expanded,
            "nodes_pruned": nodes_pruned_gen + nodes_pruned_visit,
            "nodes_pruned_at_generation": nodes_pruned_gen,
            "nodes_pruned_at_visit": nodes_pruned_visit,
            "max_depth_reached": max_depth,
            "max_queue_size": max_stack_depth,
            "timed_out": timed_out,
            "status": "NA" if timed_out else "OPTIMAL",
            "use_gpu": use_gpu,
            "gpu_backend": backend.backend_name,
        },
    )
