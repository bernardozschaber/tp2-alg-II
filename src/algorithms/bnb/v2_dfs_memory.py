"""Branch-and-Bound v2: Best-Child-First DFS in-place com memória Θ(N) e fronteira GPU.

Funcionamento operacional:
  1. Elimina a fila de prioridades global (`heapq`) de `v0_naive` e `v1_gpu`, percorrendo a
     árvore de busca em profundidade (DFS) com um único vetor mutável `path` (tamanho N) e
     um único vetor booleano `visited` (tamanho N), restaurados ao desempilhar da recursão.
     Com isso, o espaço ativo da árvore fica estritamente limitado a Θ(N) (`max_queue_size <= N`).
  2. Em cada nó interno, calcula os novos custos e bounds de todos os candidatos livres de
     uma vez usando operações vetorizadas NumPy locais (sem transferências PCIe repetidas
     para vetores pequenos de tamanho <= N), descarta os filhos com `bound >= best - EPS`
     e ordena os sobreviventes pelo menor `bound` antes de descer recursivamente.
  3. Quando `--use-gpu` está ativo (e `N >= 4`), despacha a fronteira completa de nível 2
     (todos os `(N-1)(N-2)` caminhos `0 -> v1 -> v2`) em um único lote para a GPU via
     `GPUBackend.batch_evaluate_children_bounds`, ordenando globalmente as subárvores mais
     promissoras antes de iniciar o DFS in-place em cada ramo sobrevivente.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.algorithms.common import TourResult
from src.algorithms.warm_start import run_warm_start
from src.core.gpu_utils import get_backend
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession

ALGORITHM_NAME = "bnb_v2_dfs_memory"
EPS = 1e-9


def solve(
    instance: TSPInstance,
    warm_start: Optional[str] = "cold_start",
    timeout_s: float = 1800.0,
    use_gpu: bool = False,
    profiler: Optional[ProfilerSession] = None,
) -> TourResult:
    """Executa o Branch-and-Bound v2 (DFS in-place Θ(N) com ordenação local e fronteira GPU)."""
    prof = profiler if profiler is not None else ProfilerSession(track_memory=False)
    n = instance.dimension
    dist = instance.dist_matrix
    is_int = instance.round_to_int
    backend = get_backend(use_gpu=use_gpu)
    t_start = time.perf_counter()

    # 1. Inicialização do limite superior (`best`) via warm-start.
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

    # 2. Pré-computação das duas menores arestas incidentes (mantidas em CPU e, se ativa, na GPU).
    with prof.profile("bnb_gpu_precompute"):
        dist_dev, min1_val_dev, min1_idx_dev, min2_val_dev = backend.precompute_min_edges_device(
            dist
        )
        min1_val_cpu = backend.to_cpu(min1_val_dev)
        min1_idx_cpu = backend.to_cpu(min1_idx_dev)
        min2_val_cpu = backend.to_cpu(min2_val_dev)
        pair_sums_cpu = min1_val_cpu + min2_val_cpu
        root_total_sum = float(np.sum(pair_sums_cpu))

    # 3. Estado único mutável in-place: alocado uma única vez com tamanho N.
    path = [0] * n
    visited = np.zeros(n, dtype=bool)
    visited[0] = True

    nodes_expanded = 0
    nodes_pruned_gen = 0
    nodes_pruned_visit = 0
    max_depth = 1
    max_stack_depth = 1
    timed_out = False

    def dfs(level: int, cost: float, unvis_sum: float, stack_depth: int) -> None:
        """Expande recursivamente o estado atual `path[:level]` modificando `path`/`visited` in-place."""
        nonlocal best, best_tour, nodes_expanded, nodes_pruned_gen, nodes_pruned_visit
        nonlocal max_depth, max_stack_depth, timed_out

        if timed_out:
            return
        if (nodes_expanded & 63) == 0 and (time.perf_counter() - t_start) >= timeout_s:
            timed_out = True
            return

        if level > max_depth:
            max_depth = level
        if stack_depth > max_stack_depth:
            max_stack_depth = stack_depth

        nodes_expanded += 1
        u = path[level - 1]

        # Caso base: todos os N vértices estão em `path`; avalia o retorno à raiz 0.
        if level == n:
            total_cost = cost + float(dist[u, 0])
            if total_cost < best - EPS:
                best = total_cost
                best_tour = list(path)
                if n + 1 > max_depth:
                    max_depth = n + 1
            else:
                nodes_pruned_gen += 1
            return

        # Coleta os índices não visitados e avalia todos os filhos via broadcasting NumPy local.
        cands = np.flatnonzero(~visited)
        m = len(cands)
        if m == 0:
            return

        new_costs = cost + dist[u, cands]
        # Segunda aresta livre da raiz 0: escalar fixo para level >= 2, ou vetor por candidato na raiz.
        if level >= 2:
            v1 = path[1]
            root_other = min2_val_cpu[0] if min1_idx_cpu[0] == v1 else min1_val_cpu[0]
        else:
            root_other = np.where(min1_idx_cpu[0] == cands, min2_val_cpu[0], min1_val_cpu[0])

        # Segunda aresta livre de cada candidato `k` distinta da aresta de chegada `(u, k)`.
        k_other = np.where(min1_idx_cpu[cands] == u, min2_val_cpu[cands], min1_val_cpu[cands])
        rem_unvis = unvis_sum - pair_sums_cpu[cands]
        total_double = 2.0 * new_costs + root_other + k_other + rem_unvis
        bounds = np.ceil(total_double / 2.0 - EPS) if is_int else (total_double / 2.0)

        # Poda na geração: descarta imediatamente os filhos com `bound >= best - EPS`.
        promising_idx = np.flatnonzero(bounds < (best - EPS))
        nodes_pruned_gen += int(m - len(promising_idx))
        if len(promising_idx) == 0:
            return

        # Ordena localmente os filhos sobreviventes do menor para o maior bound (Best-Child-First).
        order = promising_idx[np.argsort(bounds[promising_idx])]
        for pos, idx in enumerate(order):
            i = int(idx)
            c_bound = float(bounds[i])
            if c_bound >= best - EPS:
                # Se um irmão anterior melhorou `best`, todos os demais em `order` também são >= best.
                nodes_pruned_visit += int(len(order) - pos)
                break

            k = int(cands[i])
            c_cost = float(new_costs[i])
            c_unvis = float(rem_unvis[i])

            # Aplica transição in-place, desce na árvore e desfaz ao retornar (backtracking).
            path[level] = k
            visited[k] = True
            dfs(level + 1, c_cost, c_unvis, stack_depth + 1)
            visited[k] = False

            if timed_out:
                return

    t_bnb_start = time.perf_counter()
    with prof.profile("bnb_tree_search"):
        root_unvis_sum = root_total_sum - float(pair_sums_cpu[0])
        if use_gpu and n >= 4:
            # Gera a fronteira completa de nível 2 (0 -> v1 -> v2) e avalia em lote único na GPU.
            nodes_expanded += 1
            v1_list: List[int] = []
            v2_list: List[int] = []
            p_cost_list: List[float] = []
            unvis_sum_list: List[float] = []

            for v1 in range(1, n):
                c1 = float(dist[0, v1])
                u1 = root_unvis_sum - float(pair_sums_cpu[v1])
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
                if float(bounds[i]) >= best - EPS:
                    nodes_pruned_visit += int(len(order) - pos)
                    break
                v1 = v1_list[i]
                v2 = v2_list[i]
                c2 = float(new_costs[i])
                u2 = unvis_sum_list[i] - float(pair_sums_cpu[v2])
                path[1] = v1
                path[2] = v2
                visited[v1] = True
                visited[v2] = True
                dfs(3, c2, u2, 3)
                visited[v2] = False
                visited[v1] = False
                if timed_out:
                    break
        else:
            dfs(1, 0.0, root_unvis_sum, 1)

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
            "version": "v2_dfs_memory",
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
