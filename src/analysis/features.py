"""Extração de métricas geométricas e topológicas para instâncias 2D do TSP.

Calcula, para cada instância carregada em `TSPInstance`, um conjunto determinístico de
features espaciais e estruturais sobre as coordenadas `(x, y)` e a matriz `EUC_2D`:
  - `aspect_ratio`      : razão max(Δx, Δy) / min(Δx, Δy) do bounding box mínimo.
  - `convex_hull_size`  : número de vértices na envoltória convexa (via `graham_scan`).
  - `convex_hull_ratio` : proporção |CH(P)| / N de pontos na borda convexa.
  - `spatial_entropy`   : entropia de Shannon normalizada em grade adaptativa g x g
                          (g = ceil(sqrt(N))), variando em [0, 1].
  - `clark_evans_r`     : razão de Clark-Evans `mean(d_NN) / (0.5 * sqrt(Area / N))`;
                          valores < 1 indicam agrupamento (clustering), ~1 aleatoriedade
                          e > 1 dispersão regular.
  - `mst_max_degree`    : grau máximo de um vértice na AGM (presença de hubs).
  - `mst_degree_std`    : desvio padrão dos graus na AGM.
  - `mst_odd_ratio`     : proporção |I| / N de vértices com grau ímpar na AGM.
  - `mst_edge_cv`       : coeficiente de variação (std / mean) dos pesos das arestas da AGM.
  - `nn_dist_cv`        : coeficiente de variação das distâncias ao vizinho mais próximo
                          (heterogeneidade local de densidade).
  - `lb_root`           : limite inferior de grau 2 na raiz da busca Branch-and-Bound.
  - `opt_euc2d`         : custo ótimo de referência EUC_2D (quando disponível).
  - `lb_root_ratio`     : aperto inicial `lb_root / opt_euc2d` em (0, 1].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.algorithms.bnb.v0_naive import compute_bound_naive
from src.algorithms.common import prim_dense
from src.algorithms.warm_start import graham_scan
from src.core.instance import TSPInstance, load_benchmark_suite, load_opt_references


def compute_spatial_entropy(coords: np.ndarray) -> float:
    """Computa a entropia de Shannon normalizada da distribuição 2D em grade g x g.

    Define `g = max(2, ceil(sqrt(N)))` células por eixo cobrindo o bounding box `[min, max]`,
    calcula a fração `p_c` de pontos em cada célula não vazia e normaliza `H = -sum(p ln p)`
    por `ln(g^2)`, produzindo um índice adimensional em [0, 1].
    """
    n = coords.shape[0]
    if n <= 1:
        return 0.0

    g = max(2, int(math.ceil(math.sqrt(n))))
    x = coords[:, 0]
    y = coords[:, 1]
    dx = float(x.max() - x.min())
    dy = float(y.max() - y.min())

    # Mapeia cada ponto para índices inteiros de célula em {0, ..., g-1}.
    if dx > 0.0:
        ix = np.clip(((x - x.min()) / dx * g).astype(np.int64), 0, g - 1)
    else:
        ix = np.zeros(n, dtype=np.int64)
    if dy > 0.0:
        iy = np.clip(((y - y.min()) / dy * g).astype(np.int64), 0, g - 1)
    else:
        iy = np.zeros(n, dtype=np.int64)

    cell_ids = ix * g + iy
    _, counts = np.unique(cell_ids, return_counts=True)
    probs = counts.astype(np.float64) / float(n)
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = math.log(float(g * g))
    return entropy / max_entropy if max_entropy > 0.0 else 0.0


def extract_instance_features(
    instance: TSPInstance,
    opt_ref_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Extrai o dicionário completo de features geométricas e topológicas de uma instância."""
    n = instance.dimension
    coords = instance.coords
    dist = instance.dist_matrix

    # 1. Dimensões do bounding box e razão de aspecto (>= 1.0).
    dx = float(coords[:, 0].max() - coords[:, 0].min())
    dy = float(coords[:, 1].max() - coords[:, 1].min())
    min_side = min(dx, dy)
    max_side = max(dx, dy)
    aspect_ratio = (max_side / min_side) if min_side > 1e-12 else float("inf")
    bbox_area = max(dx * dy, 1e-12)

    # 2. Envoltória convexa (Varredura de Graham).
    hull = graham_scan(coords)
    hull_size = len(hull)
    convex_hull_ratio = float(hull_size) / float(n)

    # 3. Distâncias ao vizinho mais próximo (diagonal mascarada com +inf).
    dist_no_diag = np.array(dist, dtype=np.float64, copy=True)
    np.fill_diagonal(dist_no_diag, np.inf)
    nn_dists = np.min(dist_no_diag, axis=1)
    nn_mean = float(np.mean(nn_dists))
    nn_std = float(np.std(nn_dists, ddof=0))
    nn_dist_cv = (nn_std / nn_mean) if nn_mean > 1e-12 else 0.0

    # 4. Entropia espacial em grade adaptativa e índice de clustering de Clark-Evans.
    spatial_entropy = compute_spatial_entropy(coords)
    expected_random_nn = 0.5 * math.sqrt(bbox_area / float(n))
    clark_evans_r = (nn_mean / expected_random_nn) if expected_random_nn > 1e-12 else 0.0

    # 5. Estatísticas da Árvore Geradora Mínima (Prim denso compartilhado).
    mst_edges, degree = prim_dense(dist, root=0)
    mst_weights = np.asarray([dist[u, v] for u, v in mst_edges], dtype=np.float64)
    mst_max_degree = int(np.max(degree)) if n > 0 else 0
    mst_degree_std = float(np.std(degree, ddof=0)) if n > 0 else 0.0
    mst_odd_count = int(np.sum(degree % 2 == 1))
    mst_odd_ratio = float(mst_odd_count) / float(n)
    mst_edge_mean = float(np.mean(mst_weights)) if len(mst_weights) > 0 else 0.0
    mst_edge_std = float(np.std(mst_weights, ddof=0)) if len(mst_weights) > 0 else 0.0
    mst_edge_cv = (mst_edge_std / mst_edge_mean) if mst_edge_mean > 1e-12 else 0.0
    mst_cost = float(np.sum(mst_weights))

    # 6. Aperto do limite inferior na raiz relativo ao ótimo EUC_2D de referência.
    lb_root = compute_bound_naive(dist, [0], n, is_integer_metric=instance.round_to_int)
    lb_root_ratio = (
        float(lb_root / opt_ref_cost)
        if (opt_ref_cost is not None and opt_ref_cost > 1e-12)
        else float("nan")
    )

    return {
        "instance": instance.name,
        "n": n,
        "band": instance.band,
        "aspect_ratio": aspect_ratio,
        "convex_hull_size": hull_size,
        "convex_hull_ratio": convex_hull_ratio,
        "spatial_entropy": spatial_entropy,
        "clark_evans_r": clark_evans_r,
        "mst_cost": mst_cost,
        "mst_max_degree": mst_max_degree,
        "mst_degree_std": mst_degree_std,
        "mst_odd_count": mst_odd_count,
        "mst_odd_ratio": mst_odd_ratio,
        "mst_edge_cv": mst_edge_cv,
        "nn_dist_mean": nn_mean,
        "nn_dist_cv": nn_dist_cv,
        "lb_root": lb_root,
        "opt_euc2d": opt_ref_cost if opt_ref_cost is not None else float("nan"),
        "lb_root_ratio": lb_root_ratio,
    }


def extract_all_features(
    output_csv: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Extrai as features de todas as 15 instâncias e grava `data/results/instance_features.csv`."""
    if output_csv is None:
        output_csv = Path(__file__).resolve().parents[2] / "data" / "results" / "instance_features.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    refs = load_opt_references(data_dir=data_dir)
    suite = load_benchmark_suite(data_dir=data_dir)
    rows: List[Dict[str, Any]] = []
    for inst in suite:
        opt_cost = refs[inst.name]["euc2d_cost"] if inst.name in refs else None
        rows.append(extract_instance_features(inst, opt_ref_cost=opt_cost))

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df
