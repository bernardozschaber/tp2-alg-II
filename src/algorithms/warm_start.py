"""Heurísticas construtivas de inicialização de limite superior (warm-start) para o TSP.

Implementa as cinco estratégias suportadas pelo Branch-and-Bound:
  1. `cold_start`           : ciclo trivial `0 -> 1 -> ... -> N-1 -> 0`            [Θ(N)]
  2. `nearest_neighbor`     : inserção gulosa pelo vértice não visitado mais perto [Θ(N²)]
  3. `convex_hull_insertion`: fecho convexo (Graham Scan, O(N log N)) + inserção
                              mais barata exata com heap preguiçoso (O(N² log N))
  4. `twice_around`         : delega para `src.algorithms.twice_around.solve`
  5. `christofides`         : delega para `src.algorithms.christofides.solve_v1`
"""

from __future__ import annotations

import heapq
import time
from functools import cmp_to_key
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.algorithms import twice_around
from src.algorithms.christofides import solve_v1 as christofides_v1
from src.algorithms.common import TourResult
from src.core.instance import TSPInstance
from src.core.profiling import ProfilerSession


def _build_result(
    name: str,
    instance: TSPInstance,
    tour: List[int],
    t0: float,
    prof: ProfilerSession,
    extra: Optional[dict] = None,
) -> TourResult:
    """Empacota uma rota válida em `TourResult` com custo exato e resumo de profiling."""
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


# --------------------------------------------------------------------------- 1. Cold Start
def cold_start(instance: TSPInstance, profiler: Optional[ProfilerSession] = None) -> TourResult:
    """Gera a solução sequencial ingênua [0, 1, ..., N-1]. Complexidade: Θ(N)."""
    prof = profiler or ProfilerSession(track_memory=False)
    t0 = time.perf_counter()
    with prof.profile("cold_start"):
        tour = list(range(instance.dimension))
    return _build_result("cold_start", instance, tour, t0, prof)


# --------------------------------------------------------------------------- 2. Nearest Neighbor
def nearest_neighbor(
    instance: TSPInstance, profiler: Optional[ProfilerSession] = None, start: int = 0
) -> TourResult:
    """Heurística gulosa do vizinho mais próximo a partir de `start`.

    A cada passo, seleciona via `np.argmin` o vértice ainda não marcado em `visited` que
    possui menor distância até o vértice corrente `cur`.
    Complexidade: Θ(N²) em tempo e Θ(N) em espaço auxiliar.
    """
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
            # Mascara vértices já visitados com +inf para selecionar o mínimo entre os livres.
            row = np.where(visited, np.inf, dist[cur])
            cur = int(np.argmin(row))
            visited[cur] = True
            tour.append(cur)

    return _build_result("nearest_neighbor", instance, tour, t0, prof)


# --------------------------------------------------------------------------- 3. Convex Hull Insertion
def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Produto vetorial 2D `(a - o) x (b - o)`.

    Sinal geométrico:
      - `> 0`: giro estritamente à esquerda (sentido anti-horário).
      - `< 0`: giro à direita (sentido horário).
      - `= 0`: pontos `o`, `a` e `b` são colineares.
    """
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def graham_scan(points: np.ndarray) -> List[int]:
    """Computa os índices do fecho convexo 2D em sentido anti-horário via Varredura de Graham.

    Operação passo a passo:
      1. Seleciona o pivô `p0` de menor coordenada `y` (menor `x` em caso de empate).
      2. Ordena os demais pontos (descartando cópias coincidentes com `p0`) pelo ângulo polar
         anti-horário em torno de `p0` usando apenas o sinal do produto vetorial `_cross`
         (evitando erros numéricos de `atan2`), desempatando por distância crescente a `p0`.
      3. Para múltiplos pontos com o mesmo ângulo polar em relação a `p0`, retém apenas o
         mais distante, eliminando pontos colineares internos antes da varredura.
      4. Mantém a pilha `hull`: antes de empilhar `i`, desempilha o topo sempre que os dois
         últimos vértices da pilha e `i` não formarem giro estritamente à esquerda (`<= 0`).

    Complexidade: O(N log N) dominada pela ordenação polar; espaço Θ(N).
    """
    n = len(points)
    if n < 3:
        return list(range(n))

    # 1. Ponto âncora inferior-esquerdo (garantidamente vértice extremo do fecho).
    p0 = min(range(n), key=lambda i: (points[i][1], points[i][0]))
    origin = points[p0]

    def dist_sq(i: int) -> float:
        dx = points[i][0] - origin[0]
        dy = points[i][1] - origin[1]
        return float(dx * dx + dy * dy)

    def polar_cmp(i: int, j: int) -> int:
        c = _cross(origin, points[i], points[j])
        if c > 0.0:
            return -1  # `i` tem ângulo polar menor que `j`
        if c < 0.0:
            return 1
        # Empate angular (colineares com `origin`): ordena do mais próximo ao mais distante.
        di, dj = dist_sq(i), dist_sq(j)
        return -1 if di < dj else (1 if di > dj else 0)

    others = sorted(
        (i for i in range(n) if i != p0 and dist_sq(i) > 0.0),
        key=cmp_to_key(polar_cmp),
    )

    # 2. Filtra pontos colineares ao pivô mantendo apenas o ponto mais distante em cada raio.
    filtered: List[int] = []
    for i in others:
        if filtered and _cross(origin, points[filtered[-1]], points[i]) == 0.0:
            filtered[-1] = i
        else:
            filtered.append(i)

    # 3. Varredura em pilha mantendo invariante de convexidade estrita (giros à esquerda).
    hull = [p0]
    for i in filtered:
        while len(hull) >= 2 and _cross(points[hull[-2]], points[hull[-1]], points[i]) <= 0.0:
            hull.pop()
        hull.append(i)
    return hull


def cheapest_insertion(dist: np.ndarray, subtour: List[int]) -> List[int]:
    """Insere os vértices internos no ciclo inicial pelo critério exato de inserção mais barata.

    Critério guloso: a cada passo, seleciona o par `(v, (u, w))` com `v` fora da rota e
    `(u, w)` aresta ativa da rota que minimiza o acréscimo de comprimento:
        Δ(u, v, w) = dist[u, v] + dist[v, w] - dist[u, w].

    Refatoração assintótica (O(N³) -> O(N² log N)):
      - Uma implementação ingênua recalcula a matriz completa `|livres| x |arestas|` a cada
        iteração, custando O(N²) por inserção e O(N³) no total.
      - Aqui, o ciclo atual é mantido em lista circular estática `next_node[u] = w` (inserção
        em Θ(1)) e as inserções candidatas ficam em uma fila de prioridades global (`heapq`)
        com remoção preguiçosa (*lazy deletion*):
          1. Inicialização: insere no heap todas as tuplas `(Δ, v, u, w)` para cada vértice
             livre `v` e cada aresta `(u, w)` do fecho inicial (≤ N·h inserções).
          2. Extração: ao retirar `(Δ, v, u, w)` do topo, valida em Θ(1) se `v` ainda está
             livre (`not in_tour[v]`) e se a aresta `(u, w)` ainda existe (`next_node[u] == w`).
             Entradas obsoletas são descartadas imediatamente.
          3. Atualização incremental: ao inserir `v` entre `u` e `w`, apenas DUAS novas
             arestas são criadas: `(u, v)` e `(v, w)`. Somente essas duas arestas são
             avaliadas e empilhadas para cada vértice livre restante `r`.
      - Contagem exata de operações: ao longo de toda a execução criam-se no máximo
        `h + 2(N - h) < 2N` arestas. Cada aresta gera no máximo `N` inserções no heap,
        totalizando O(N²) operações de `heappush`/`heappop` de custo O(log N) cada.

    Complexidade garantida de pior caso: O(N² log N) em tempo e O(N²) em memória no heap.
    """
    n = dist.shape[0]
    h = len(subtour)
    if h >= n:
        return list(subtour)

    # Lista nativa de listas de floats para acesso escalar rápido no laço do heap.
    dist_rows: List[List[float]] = dist.tolist()

    # Representa o ciclo corrente por ponteiros de sucessor `next_node[u] = w` (O(1) para inserir).
    next_node = [-1] * n
    in_tour = [False] * n
    for idx in range(h):
        u = subtour[idx]
        w = subtour[(idx + 1) % h]
        next_node[u] = w
        in_tour[u] = True

    free_vertices = [v for v in range(n) if not in_tour[v]]

    # 1. Popula o min-heap com todas as combinações (vértice livre, aresta inicial do fecho).
    # Tupla: (delta, v, u, w) — `u` e `w` definem a aresta onde `v` seria inserido.
    heap: List[Tuple[float, int, int, int]] = []
    for idx in range(h):
        u = subtour[idx]
        w = subtour[(idx + 1) % h]
        d_uw = dist_rows[u][w]
        dist_u = dist_rows[u]
        dist_w = dist_rows[w]
        for v in free_vertices:
            delta = dist_u[v] + dist_w[v] - d_uw
            heap.append((delta, v, u, w))
    heapq.heapify(heap)

    # 2. Consome o heap inserindo um vértice por vez até fechar os N vértices.
    tour_size = h
    remaining = set(free_vertices)

    while tour_size < n:
        delta, v, u, w = heapq.heappop(heap)
        # Validação preguiçosa O(1): ignora se `v` já foi inserido ou se (u, w) foi quebrada.
        if in_tour[v] or next_node[u] != w:
            continue

        # Insere `v` entre `u` e `w` em Θ(1), substituindo (u, w) por (u, v) e (v, w).
        next_node[u] = v
        next_node[v] = w
        in_tour[v] = True
        remaining.remove(v)
        tour_size += 1

        if not remaining:
            break

        # Avalia apenas as duas novas arestas criadas ((u, v) e (v, w)) para os vértices livres.
        d_uv = dist_rows[u][v]
        d_vw = dist_rows[v][w]
        dist_u = dist_rows[u]
        dist_v = dist_rows[v]
        dist_w = dist_rows[w]

        for r in remaining:
            heapq.heappush(heap, (dist_u[r] + dist_v[r] - d_uv, r, u, v))
            heapq.heappush(heap, (dist_v[r] + dist_w[r] - d_vw, r, v, w))

    # 3. Reconstrói a sequência linear do ciclo seguindo os ponteiros `next_node` a partir de 0.
    tour: List[int] = []
    cur = 0
    for _ in range(n):
        tour.append(cur)
        cur = next_node[cur]
    return tour


def convex_hull_insertion(
    instance: TSPInstance, profiler: Optional[ProfilerSession] = None
) -> TourResult:
    """Constrói a rota inicial pelo fecho convexo (Graham Scan) e insere os demais vértices."""
    prof = profiler or ProfilerSession(track_memory=False)
    t0 = time.perf_counter()

    with prof.profile("graham_scan"):
        hull = graham_scan(instance.coords)
    with prof.profile("cheapest_insertion"):
        tour = cheapest_insertion(instance.dist_matrix, hull)

    # Normaliza a rota para iniciar no vértice 0.
    if tour and tour[0] != 0:
        i0 = tour.index(0)
        tour = tour[i0:] + tour[:i0]

    return _build_result(
        "convex_hull_insertion", instance, tour, t0, prof, extra={"hull_size": len(hull)}
    )


# --------------------------------------------------------------------------- 4 e 5. Aproximativos
def twice_around_warm(
    instance: TSPInstance, profiler: Optional[ProfilerSession] = None
) -> TourResult:
    """Wrapper de warm-start para o Twice-around-the-tree."""
    res = twice_around.solve(instance, profiler)
    res.algorithm = "twice_around"
    return res


def christofides_warm(
    instance: TSPInstance, profiler: Optional[ProfilerSession] = None
) -> TourResult:
    """Wrapper de warm-start utilizando o Christofides otimizado (`v1`)."""
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


def run_warm_start(
    name: str, instance: TSPInstance, profiler: Optional[ProfilerSession] = None
) -> Optional[TourResult]:
    """Despacha a estratégia de warm-start pelo nome; retorna None se ainda não implementada."""
    try:
        return WARM_STARTS[name](instance, profiler)
    except NotImplementedError:
        return None
