"""Checkpoint 2: Christofides (v0, v1) e heurísticas de warm-start."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import ConvexHull

from src.algorithms.christofides import solve_v0, solve_v1
from src.algorithms.warm_start import (
    WARM_STARTS,
    graham_scan,
    run_warm_start,
)
from src.core.instance import load_benchmark_suite, load_tsplib_instance
from src.core.profiling import ProfilerSession
from tests.oracles import held_karp

SMALL = ["eil51_n10", "berlin52_n12"]
EPS = 1e-9


class TestOracle(unittest.TestCase):
    def test_held_karp_matches_bruteforce_tiny(self) -> None:
        """Sanidade do próprio oráculo contra força bruta em N=7."""
        from itertools import permutations

        inst = load_tsplib_instance("eil51_n10")
        d = inst.dist_matrix[:7, :7]
        brute = min(
            d[0, p[0]] + sum(d[p[i], p[i + 1]] for i in range(5)) + d[p[-1], 0]
            for p in permutations(range(1, 7))
        )
        hk, tour = held_karp(d)
        self.assertAlmostEqual(hk, brute, places=6)
        self.assertEqual(sorted(tour), list(range(7)))


class TestChristofides(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_benchmark_suite()
        cls.opt = {}
        for name in SMALL:
            inst = load_tsplib_instance(name)
            cls.opt[name], _ = held_karp(inst.dist_matrix)

    def test_valid_tours_and_structural_bounds_all_instances(self) -> None:
        """Ciclo hamiltoniano válido e c(H) ≤ c(T) + c(M) (shortcutting não aumenta custo)."""
        for inst in self.suite:
            for solve in (solve_v0, solve_v1):
                res = solve(inst)
                with self.subTest(inst=inst.name, alg=res.algorithm):
                    self.assertTrue(inst.validate_tour(res.tour))
                    self.assertEqual(res.tour[0], 0)
                    self.assertEqual(res.extra["n_odd"] % 2, 0)
                    self.assertLessEqual(
                        res.cost, res.extra["mst_cost"] + res.extra["matching_cost"] + EPS
                    )

    def test_v0_v1_same_mst_and_matching_cost(self) -> None:
        """Refatoração não muda a matemática.

        - Custo da AGM é único (mesmo com empates), logo v0 (Kruskal) == v1 (Prim).
        - Com arestas empatadas, as AGMs podem diferir (ex.: eil51_n38 tem duas arestas
          de peso sqrt(82)), mudando o conjunto ímpar I. Por isso o matching é comparado
          sobre o MESMO conjunto I (o do v0).
        """
        import networkx as nx

        from src.algorithms.christofides.v1_optimized import min_weight_perfect_matching

        for inst in self.suite:
            r0, r1 = solve_v0(inst), solve_v1(inst)
            with self.subTest(inst=inst.name):
                self.assertAlmostEqual(r0.extra["mst_cost"], r1.extra["mst_cost"], places=6)
                d = inst.dist_matrix
                G = nx.complete_graph(inst.dimension)
                for u, v in G.edges:
                    G[u][v]["weight"] = d[u, v]
                odd0 = [v for v, k in nx.minimum_spanning_tree(G).degree() if k % 2]
                m1 = min_weight_perfect_matching(d, odd0)
                self.assertAlmostEqual(
                    r0.extra["matching_cost"], float(sum(d[u, v] for u, v in m1)), places=6
                )

    def test_approximation_factor_against_exact_oracle(self) -> None:
        """c(H) ≤ 1.5·OPT e c(T) ≤ OPT, com OPT dado pelo oráculo Held-Karp."""
        for name in SMALL:
            inst = load_tsplib_instance(name)
            opt = self.opt[name]
            for solve in (solve_v0, solve_v1):
                res = solve(inst)
                with self.subTest(inst=name, alg=res.algorithm):
                    self.assertGreaterEqual(res.cost, opt - EPS)
                    self.assertLessEqual(res.cost, 1.5 * opt + EPS)
                    self.assertLessEqual(res.extra["mst_cost"], opt + EPS)

    def test_subroutine_profiling_present(self) -> None:
        """Gargalos exigidos pelo enunciado (AGM e matching) aparecem no profiling."""
        inst = load_tsplib_instance("berlin52")
        prof = ProfilerSession(track_memory=True)
        res = solve_v1(inst, prof)
        subs = res.profile["subroutines"]
        for key in ("mst", "odd_vertices", "min_weight_matching", "eulerian_circuit", "shortcut"):
            self.assertIn(key, subs)
            self.assertGreater(subs[key]["elapsed_ns"], 0)
        self.assertGreater(subs["min_weight_matching"]["peak_memory_bytes"], 0)


class TestWarmStarts(unittest.TestCase):
    def test_graham_scan_matches_scipy(self) -> None:
        for inst in load_benchmark_suite():
            with self.subTest(inst=inst.name):
                mine = graham_scan(inst.coords)
                ref = ConvexHull(inst.coords).vertices
                self.assertEqual(set(mine), set(int(v) for v in ref))

    def test_graham_scan_collinear_and_duplicates(self) -> None:
        pts = np.array([[0, 0], [1, 0], [2, 0], [2, 2], [1, 1], [0, 2], [1, 2], [0, 0]], float)
        self.assertEqual(set(graham_scan(pts)), {0, 2, 3, 5})

    def test_all_warm_starts_valid(self) -> None:
        for inst in load_benchmark_suite():
            for name in WARM_STARTS:
                res = run_warm_start(name, inst)
                if res is None:  # twice_around ainda não implementado pelo grupo
                    self.assertEqual(name, "twice_around")
                    continue
                with self.subTest(inst=inst.name, ws=name):
                    self.assertTrue(inst.validate_tour(res.tour))
                    self.assertEqual(res.tour[0], 0)
                    self.assertAlmostEqual(res.cost, inst.tour_cost(res.tour))

    def test_warm_starts_not_below_optimum(self) -> None:
        for name in SMALL:
            inst = load_tsplib_instance(name)
            opt, _ = held_karp(inst.dist_matrix)
            for ws in WARM_STARTS:
                res = run_warm_start(ws, inst)
                if res is not None:
                    with self.subTest(inst=name, ws=ws):
                        self.assertGreaterEqual(res.cost, opt - EPS)


if __name__ == "__main__":
    unittest.main()
