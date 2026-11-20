"""Construtor offline de gabarito exato EUC_2D (`data/tsplib/opt_references.json`).

Finalidade operacional:
  - Algumas instâncias TSPLIB (ex.: `bays29`, `ulysses16`, `ulysses22`, `att48`) tiveram
    seus arquivos `.opt.tour` originais calculados sob métricas distintas de EUC_2D
    (`EXPLICIT`, `GEO`, `ATT`), e os recortes prefixados (`eil51_n25`, `berlin52_n33`, etc.)
    não possuem `.opt.tour` publicado.
  - Para que o cálculo do gap relativo `(c(H) - OPT) / OPT` na análise de sensibilidade
    seja exato e nunca produza valores negativos em nenhuma das 15 instâncias (mesmo naquelas
    em que o Branch-and-Bound atinge o limite de 30 minutos), este módulo resolve o TSP
    simétrico EUC_2D de forma exata via formulação inteira DFJ (Dantzig-Fulkerson-Johnson)
    com eliminação iterativa de subciclos sobre `scipy.optimize.milp` (solver HiGHS).

Nota de escopo:
  - Este módulo é usado EXCLUSIVAMENTE para gerar o arquivo de gabarito estático
    `data/tsplib/opt_references.json`; nunca substitui o Branch-and-Bound nas medições de
    tempo, memória, nós expandidos ou timeout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from src.core.instance import TSPInstance, load_benchmark_suite
from src.core.prepare_dataset import OFFICIAL_OPT_TOURS


def _find_cycles(n: int, selected_edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Decompõe um subgrafo 2-regular (grau 2 em todo vértice) em seus ciclos disjuntos."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in selected_edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * n
    cycles: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        cyc: List[int] = []
        prev = -1
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cyc.append(cur)
            # Avança para o vizinho diferente do vértice anterior no ciclo.
            nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
            prev, cur = cur, nxt
        cycles.append(cyc)
    return cycles


def solve_exact_dfj_oracle(instance: TSPInstance) -> Tuple[float, List[int]]:
    """Resolve o TSP simétrico exato via MILP (DFJ) com cortes iterativos de subciclos.

    Variáveis binárias: uma variável `x_e in {0, 1}` para cada par não ordenado `(i, j)` com
    `0 <= i < j < N`, totalizando `E = N(N-1)/2` variáveis.
    Restrições iniciais: grau exatamente 2 em cada vértice (`sum_{e incidente a v} x_e == 2`).
    Cortes iterativos: enquanto a solução inteira contiver mais de um componente conexo,
    adiciona-se a desigualdade de corte de subciclo `sum_{i,j in S} x_{ij} <= |S| - 1`
    para cada ciclo desconexo `S`, repetindo até restar um único ciclo hamiltoniano.
    """
    n = instance.dimension
    dist = instance.dist_matrix

    # Mapeia cada par não ordenado (i, j) para seu índice de coluna no vetor de variáveis.
    edges: List[Tuple[int, int]] = []
    edge_idx: Dict[Tuple[int, int], int] = {}
    costs: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = len(edges)
            edges.append((i, j))
            costs.append(float(dist[i, j]))

    num_vars = len(edges)
    c = np.asarray(costs, dtype=np.float64)
    integrality = np.ones(num_vars, dtype=np.int8)
    bounds = Bounds(lb=0.0, ub=1.0)

    # Restrições iniciais de grau 2 para todos os N vértices.
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    lbs: List[float] = []
    ubs: List[float] = []

    for v in range(n):
        for w in range(n):
            if v == w:
                continue
            key = (v, w) if v < w else (w, v)
            rows.append(v)
            cols.append(edge_idx[key])
            vals.append(1.0)
        lbs.append(2.0)
        ubs.append(2.0)

    next_row = n
    while True:
        A = csc_matrix((vals, (rows, cols)), shape=(next_row, num_vars))
        constraint = LinearConstraint(A, lb=np.asarray(lbs), ub=np.asarray(ubs))
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraint)
        if not res.success or res.x is None:
            raise RuntimeError(f"Oráculo MILP falhou em {instance.name}: {res.message}")

        selected = [edges[idx] for idx, val in enumerate(res.x) if val > 0.5]
        cycles = _find_cycles(n, selected)
        if len(cycles) == 1:
            tour = cycles[0]
            # Rotaciona o ciclo hamiltoniano ótimo para iniciar no vértice 0.
            i0 = tour.index(0)
            tour = tour[i0:] + tour[:i0]
            return instance.tour_cost(tour), tour

        # Adiciona um corte DFJ `sum_{u,v in S} x_{uv} <= |S| - 1` para cada subciclo desconexo.
        for cyc in cycles:
            s_len = len(cyc)
            for a_idx in range(s_len):
                u = cyc[a_idx]
                for b_idx in range(a_idx + 1, s_len):
                    v = cyc[b_idx]
                    key = (u, v) if u < v else (v, u)
                    rows.append(next_row)
                    cols.append(edge_idx[key])
                    vals.append(1.0)
            lbs.append(-np.inf)
            ubs.append(float(s_len - 1))
            next_row += 1


def build_all_opt_references(data_dir: Path | None = None) -> Path:
    """Computa e salva o gabarito exato EUC_2D de todas as 15 instâncias em `opt_references.json`."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "tsplib"

    suite = load_benchmark_suite(data_dir=data_dir)
    refs: Dict[str, dict] = {}

    for inst in suite:
        opt_cost, opt_tour = solve_exact_dfj_oracle(inst)
        # Verifica se coincide com o .opt.tour oficial da TSPLIB (ex.: berlin52, st70).
        source = "exact_euc2d_oracle"
        if inst.name in OFFICIAL_OPT_TOURS:
            tsplib_cost = inst.tour_cost(OFFICIAL_OPT_TOURS[inst.name])
            if abs(tsplib_cost - opt_cost) <= 1e-4:
                source = "tsplib_opt_tour"
        refs[inst.name] = {
            "dimension": inst.dimension,
            "band": inst.band,
            "euc2d_cost": round(opt_cost, 6),
            "source": source,
            "tour": opt_tour,
        }

    out_path = data_dir / "opt_references.json"
    out_path.write_text(json.dumps(refs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = build_all_opt_references()
    print(f"Gabarito exato EUC_2D gravado em: {path}")
