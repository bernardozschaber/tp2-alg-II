"""As cinco estratégias de warm-start (limite superior inicial) do enunciado.

  1. cold_start          : 0 → 1 → ... → N-1 → 0                          Θ(N)
  2. nearest_neighbor    : vizinho mais próximo a partir de 0 (Aula 13)    Θ(N²)
  3. convex_hull_insert  : Graham Scan (Aula 05) + inserção mais barata     O(N log N) + O(N³)*
  4. twice_around        : resultado do Twice-around-the-tree (grupo)
  5. christofides        : resultado do Christofides v1

* Inserção mais barata ingênua: a cada passo avalia todos os pares (vértice livre,
  aresta da sub-rota) — matriz |R|×|C| vetorizada em NumPy, O(N²) por passo, O(N³) total.
  Para N ≤ 75 isso é irrelevante frente ao B&B, mas é documentado para a modelagem.
"""

from __future__ import annotations

import time
from functools import cmp_to_key
from typing import Callable, Dict, List, Optional

import numpy as np

from src.algorithms import twice_around
from src.algorithms.christofides import solve_v1 as christofides_v1
from src.algorithms.common import TourResult
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession


def _result(name: str, instance: TSPInstance, tour: List[int], t0: float,
            prof: ProfilerSession, extra: Optional[dict] = None) -> TourResult:
    return TourResult(
        algorithm=name,
        instance=instance.name,
        n=instance.dimension,
        tour=tour,
        cost=instance.tour_cost(tour),
        elapsed_s=time.perf_counter() - t0,
        profile=prof.summary_dict(),
        extra=extra or {},
    )


# --------------------------------------------------------------------------- 1. Cold start
def cold_start(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    prof = profiler or ProfilerSession(track_memory=False)
    t0 = time.perf_counter()
    with prof.profile("cold_start"):
        tour = list(range(instance.dimension))
    return _result("cold_start", instance, tour, t0, prof)


# --------------------------------------------------------------------------- 2. Vizinho mais próximo
def nearest_neighbor(instance: TSPInstance, profiler: Optional[ProfilerSession] = None,
                     start: int = 0) -> TourResult:
    prof = profiler or ProfilerSession(track_memory=False)
    t0 = time.perf_counter()
    with prof.profile("nearest_neighbor"):
        n = instance.dimension
        dist = instance.dist_matrix
        visited = np.zeros(n, dtype=bool)
        tour = [start]
        visited[start] = True
        cur = start
        for _ in range(n - 1):
            row = np.where(visited, np.inf, dist[cur])
            cur = int(np.argmin(row))
            visited[cur] = True
            tour.append(cur)
    return _result("nearest_neighbor", instance, tour, t0, prof)


# --------------------------------------------------------------------------- 3. Fecho convexo
def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Produto vetorial (a-o)×(b-o): >0 curva à esquerda (anti-horário), <0 à direita, 0 colinear."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def graham_scan(points: np.ndarray) -> List[int]:
    """Varredura de Graham (Aula 05). Retorna índices do fecho em sentido anti-horário.

    - Âncora p0: menor y, mais à esquerda em empate.
    - Ordenação por ângulo polar em relação a p0 via produto vetorial (sem atan2,
      evitando erro de ponto flutuante em empates).
    - Pontos com o mesmo ângulo: mantém apenas o mais distante de p0 (como no slide),
      o que também trata a hipótese "sem 3 colineares" do slide.
    - Remove pi enquanto pi-1, pi, pi+1 não formar curva estritamente à esquerda.
    """
    n = len(points)
    if n < 3:
        return list(range(n))
    p0 = min(range(n), key=lambda i: (points[i][1], points[i][0]))
    origin = points[p0]

    def d2(i: int) -> float:
        return float(np.sum((points[i] - origin) ** 2))

    def cmp(i: int, j: int) -> int:
        c = _cross(origin, points[i], points[j])
        if c > 0:
            return -1
        if c < 0:
            return 1
        return -1 if d2(i) < d2(j) else (1 if d2(i) > d2(j) else 0)

    others = sorted((i for i in range(n) if i != p0 and d2(i) > 0), key=cmp_to_key(cmp))
    # mantém só o mais distante entre pontos de mesmo ângulo polar
    filtered: List[int] = []
    for i in others:
        if filtered and _cross(origin, points[filtered[-1]], points[i]) == 0:
            filtered[-1] = i  # `others` está em ordem crescente de distância no empate
        else:
            filtered.append(i)

    hull = [p0]
    for i in filtered:
        while len(hull) >= 2 and _cross(points[hull[-2]], points[hull[-1]], points[i]) <= 0:
            hull.pop()
        hull.append(i)
    return hull


def cheapest_insertion(dist: np.ndarray, subtour: List[int]) -> List[int]:
    """Insere iterativamente o vértice livre de menor custo Δ(u,v,w) = c(u,v)+c(v,w)-c(u,w)."""
    n = dist.shape[0]
    tour = list(subtour)
    in_tour = np.zeros(n, dtype=bool)
    in_tour[tour] = True
    while len(tour) < n:
        free = np.flatnonzero(~in_tour)
        a = np.asarray(tour)
        b = np.roll(a, -1)
        # delta[r, e] = custo de inserir free[r] entre a[e] e b[e]
        delta = dist[np.ix_(free, a)] + dist[np.ix_(free, b)] - dist[a, b][None, :]
        r, e = np.unravel_index(int(np.argmin(delta)), delta.shape)
        v = int(free[r])
        tour.insert(int(e) + 1, v)
        in_tour[v] = True
    return tour


def convex_hull_insertion(instance: TSPInstance,
                          profiler: Optional[ProfilerSession] = None) -> TourResult:
    prof = profiler or ProfilerSession(track_memory=False)
    t0 = time.perf_counter()
    with prof.profile("graham_scan"):
        hull = graham_scan(instance.coords)
    with prof.profile("cheapest_insertion"):
        tour = cheapest_insertion(instance.dist_matrix, hull)
    i0 = tour.index(0)
    tour = tour[i0:] + tour[:i0]
    return _result("convex_hull_insertion", instance, tour, t0, prof,
                   extra={"hull_size": len(hull)})


# --------------------------------------------------------------------------- 4 e 5. Aproximativos
def twice_around_warm(instance: TSPInstance,
                      profiler: Optional[ProfilerSession] = None) -> TourResult:
    res = twice_around.solve(instance, profiler)
    res.algorithm = "twice_around"
    return res


def christofides_warm(instance: TSPInstance,
                      profiler: Optional[ProfilerSession] = None) -> TourResult:
    res = christofides_v1(instance, profiler)
    res.algorithm = "christofides"
    return res


WARM_STARTS: Dict[str, Callable[..., TourResult]] = {
    "cold_start": cold_start,
    "nearest_neighbor": nearest_neighbor,
    "convex_hull_insertion": convex_hull_insertion,
    "twice_around": twice_around_warm,
    "christofides": christofides_warm,
}


def run_warm_start(name: str, instance: TSPInstance,
                   profiler: Optional[ProfilerSession] = None) -> Optional[TourResult]:
    """Executa a estratégia `name`. Retorna None se ainda não implementada (ex.: twice_around)."""
    try:
        return WARM_STARTS[name](instance, profiler)
    except NotImplementedError:
        return None
