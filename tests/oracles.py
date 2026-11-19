"""Oráculos independentes usados SOMENTE nos testes (não fazem parte dos algoritmos avaliados)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def held_karp(dist: np.ndarray) -> Tuple[float, List[int]]:
    """TSP exato por programação dinâmica (Held-Karp), O(2^N · N²). Usar só com N ≤ ~13."""
    n = dist.shape[0]
    full = 1 << (n - 1)  # subconjuntos de {1..n-1}
    INF = np.inf
    dp = np.full((full, n), INF)
    parent = np.full((full, n), -1, dtype=np.int64)
    for k in range(1, n):
        dp[1 << (k - 1), k] = dist[0, k]
    for mask in range(1, full):
        for k in range(1, n):
            bit = 1 << (k - 1)
            if not (mask & bit) or dp[mask, k] == INF:
                continue
            base = dp[mask, k]
            for j in range(1, n):
                jb = 1 << (j - 1)
                if mask & jb:
                    continue
                nm = mask | jb
                val = base + dist[k, j]
                if val < dp[nm, j]:
                    dp[nm, j] = val
                    parent[nm, j] = k
    last_mask = full - 1
    costs = dp[last_mask, 1:] + dist[1:, 0]
    k = int(np.argmin(costs)) + 1
    best = float(costs[k - 1])
    tour = []
    mask = last_mask
    while k != -1 and k != 0:
        tour.append(k)
        pk = int(parent[mask, k])
        mask ^= 1 << (k - 1)
        k = pk
    tour.append(0)
    tour.reverse()
    return best, tour
