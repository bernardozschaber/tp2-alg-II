"""Testes unitários da Etapa 2: Christofides, Twice-around-the-tree (`v0`, `v1`) e warm-starts."""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

from src.algorithms.christofides import solve_v0, solve_v1
from src.algorithms.christofides.v1_optimized import min_weight_perfect_matching
from src.algorithms.twice_around import (
    solve as twice_solve,
    solve_v0 as twice_solve_v0,
    solve_v1 as twice_solve_v1,
)
from src.algorithms.twice_around.v1_optimized import preorder_dfs_iterative
from src.algorithms.warm_start import (
    WARM_STARTS,
    cheapest_insertion,
    graham_scan,
    run_warm_start,
)
from src.core.instance import load_benchmark_suite, load_tsplib_instance
from src.core.profiling import ProfilerSession
from tests.oracles import held_karp

SMALL = ["eil51_n10", "berlin52_n12"]
EPS = 1e-9


def _cheapest_insertion_reference_n3(dist: np.ndarray, subtour: list[int]) -> list[int]:
    """Implementação matricial de referência O(N³) para conferir a versão O(N² log N)."""
    n = dist.shape[0]
    tour = list(subtour)
    in_tour = np.zeros(n, dtype=bool)
    in_tour[tour] = True
    while len(tour) < n:
        free = np.flatnonzero(~in_tour)
        a = np.asarray(tour)
        b = np.roll(a, -1)
        delta = dist[np.ix_(free, a)] + dist[np.ix_(free, b)] - dist[a, b][None, :]
        r, e = np.unravel_index(int(np.argmin(delta)), delta.shape)
        v = int(free[r])
        tour.insert(int(e) + 1, v)
        in_tour[v] = True
    return tour


class TestOracle(unittest.TestCase):
    def test_held_karp_matches_bruteforce_tiny(self) -> None:
        """Valida o oráculo Held-Karp contra força bruta exaustiva em N=7."""
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
        """Confirma ciclo hamiltoniano válido e c(H) <= c(T) + c(M) nas 15 instâncias."""
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
        """Confirma que v0 e v1 obtêm o mesmo custo de AGM e o mesmo matching sobre igual conjunto I."""
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
        """Confirma fator de aproximação c(H) <= 1.5 * OPT e c(T) <= OPT."""
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
        """Confirma registro de tempo e memória das 5 subrotinas do Christofides v1."""
        inst = load_tsplib_instance("berlin52")
        prof = ProfilerSession(track_memory=True)
        res = solve_v1(inst, prof)
        subs = res.profile["subroutines"]
        for key in ("mst", "odd_vertices", "min_weight_matching", "eulerian_circuit", "shortcut"):
            self.assertIn(key, subs)
            self.assertGreater(subs[key]["elapsed_ns"], 0)
        self.assertGreater(subs["min_weight_matching"]["peak_memory_bytes"], 0)
        self.assertGreater(res.to_row()["process_max_rss_kb"], 0.0)


class TestTwiceAround(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_benchmark_suite()
        cls.opt = {}
        for name in SMALL:
            inst = load_tsplib_instance(name)
            cls.opt[name], _ = held_karp(inst.dist_matrix)

    def test_valid_tours_and_structural_bounds_all_instances(self) -> None:
        """Confirma ciclo hamiltoniano válido e c(H) <= 2 * c(T) nas 15 instâncias."""
        for inst in self.suite:
            for solver in (twice_solve_v0, twice_solve_v1):
                res = solver(inst)
                with self.subTest(inst=inst.name, alg=res.algorithm):
                    self.assertTrue(inst.validate_tour(res.tour))
                    self.assertEqual(res.tour[0], 0)
                    self.assertAlmostEqual(res.cost, inst.tour_cost(res.tour), places=6)
                    self.assertGreaterEqual(res.extra["mst_max_degree"], 1)
                    self.assertLessEqual(res.cost, 2.0 * res.extra["mst_cost"] + EPS)

    def test_v0_v1_and_christofides_same_mst_cost(self) -> None:
        """Confirma igualdade exata do custo da AGM entre Twice-around v0/v1 e Christofides v1."""
        for inst in self.suite:
            r0 = twice_solve_v0(inst)
            r1 = twice_solve_v1(inst)
            rc = solve_v1(inst)
            with self.subTest(inst=inst.name):
                self.assertAlmostEqual(r0.extra["mst_cost"], r1.extra["mst_cost"], places=6)
                self.assertAlmostEqual(r1.extra["mst_cost"], rc.extra["mst_cost"], places=6)

    def test_approximation_factor_against_exact_oracle(self) -> None:
        """Confirma fator de aproximação OPT <= c(H) <= 2 * OPT e c(T) <= OPT."""
        for name in SMALL:
            inst = load_tsplib_instance(name)
            opt = self.opt[name]
            for solver in (twice_solve_v0, twice_solve_v1):
                res = solver(inst)
                with self.subTest(inst=name, alg=res.algorithm):
                    self.assertGreaterEqual(res.cost, opt - EPS)
                    self.assertLessEqual(res.cost, 2.0 * opt + EPS)
                    self.assertLessEqual(res.extra["mst_cost"], opt + EPS)

    def test_subroutine_profiling_present(self) -> None:
        """Confirma registro de tempo e memória das subrotinas em Twice-around v0 e v1."""
        inst = load_tsplib_instance("berlin52")

        prof_v1 = ProfilerSession(track_memory=True)
        res_v1 = twice_solve(inst, prof_v1)
        subs_v1 = res_v1.profile["subroutines"]
        for key in ("mst", "preorder"):
            self.assertIn(key, subs_v1)
            self.assertGreater(subs_v1[key]["elapsed_ns"], 0)
        self.assertGreater(subs_v1["mst"]["peak_memory_bytes"], 0)
        self.assertGreater(res_v1.to_row()["process_max_rss_kb"], 0.0)

        prof_v0 = ProfilerSession(track_memory=True)
        res_v0 = twice_solve_v0(inst, prof_v0)
        subs_v0 = res_v0.profile["subroutines"]
        for key in ("build_complete_graph", "mst", "eulerian_circuit", "shortcut"):
            self.assertIn(key, subs_v0)
            self.assertGreater(subs_v0[key]["elapsed_ns"], 0)
        self.assertGreater(subs_v0["build_complete_graph"]["peak_memory_bytes"], 0)

    def test_iterative_preorder_handles_deep_path_without_recursion_error(self) -> None:
        """Confirma ausência de RecursionError em árvore-caminho profunda (N=2500 > limite padrão)."""
        n_deep = 2500
        edges = [(i, i + 1) for i in range(n_deep - 1)]
        order = preorder_dfs_iterative(n_deep, edges, start=0)
        self.assertEqual(order, list(range(n_deep)))

    def test_round_to_int_consistency(self) -> None:
        """Confirma consistência entre v0 e v1 quando a métrica EUC_2D é arredondada para inteiros."""
        inst = load_tsplib_instance("berlin52_n12", round_to_int=True)
        r0 = twice_solve_v0(inst)
        r1 = twice_solve_v1(inst)
        self.assertTrue(inst.validate_tour(r0.tour))
        self.assertTrue(inst.validate_tour(r1.tour))
        self.assertAlmostEqual(r0.extra["mst_cost"], r1.extra["mst_cost"], places=6)
        self.assertLessEqual(r0.cost, 2.0 * r0.extra["mst_cost"] + EPS)
        self.assertLessEqual(r1.cost, 2.0 * r1.extra["mst_cost"] + EPS)


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

    def test_cheapest_insertion_heap_matches_reference_n3(self) -> None:
        """Confirma que a versão O(N² log N) com heap preguiçoso coincide com a referência O(N³)."""
        for name in ("eil51_n10", "berlin52_n12", "burma14", "bays29", "st70"):
            inst = load_tsplib_instance(name)
            hull = graham_scan(inst.coords)
            t_fast = cheapest_insertion(inst.dist_matrix, hull)
            t_ref = _cheapest_insertion_reference_n3(inst.dist_matrix, hull)
            with self.subTest(inst=name):
                self.assertTrue(inst.validate_tour(t_fast))
                self.assertAlmostEqual(inst.tour_cost(t_fast), inst.tour_cost(t_ref), places=5)

    def test_all_warm_starts_valid(self) -> None:
        for inst in load_benchmark_suite():
            for name in WARM_STARTS:
                res = run_warm_start(name, inst)
                self.assertIsNotNone(res, f"Warm-start '{name}' retornou None em {inst.name}")
                assert res is not None
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
                self.assertIsNotNone(res)
                assert res is not None
                with self.subTest(inst=name, ws=ws):
                    self.assertGreaterEqual(res.cost, opt - EPS)


if __name__ == "__main__":
    unittest.main()
