"""Testes unitários da Etapa 3: Branch-and-Bound nas versões graduais v0, v1, v2 e v3."""

from __future__ import annotations

import unittest

import numpy as np

from src.algorithms.bnb import solve_v0, solve_v1, solve_v2, solve_v3
from src.algorithms.bnb.v0_naive import compute_bound_naive
from src.core.instance import TSPInstance, load_tsplib_instance
from src.core.profiling import ProfilerSession
from tests.oracles import held_karp

EPS = 1e-6


class TestStage3BranchAndBound(unittest.TestCase):
    """Valida corretude exata, uso de memória, impacto de warm-start e timeout nas versões v0..v3."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.inst10 = load_tsplib_instance("eil51_n10")
        cls.opt10, _ = held_karp(cls.inst10.dist_matrix)
        cls.inst12 = load_tsplib_instance("berlin52_n12")
        cls.opt12, _ = held_karp(cls.inst12.dist_matrix)

    def test_small_integer_graph_exact_optimum(self) -> None:
        """Valida `compute_bound_naive` e as 4 versões em grafo simétrico inteiro de 5 vértices."""
        adj = np.array(
            [
                [0.0, 3.0, 1.0, 5.0, 8.0],
                [3.0, 0.0, 6.0, 7.0, 9.0],
                [1.0, 6.0, 0.0, 4.0, 2.0],
                [5.0, 7.0, 4.0, 0.0, 3.0],
                [8.0, 9.0, 2.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        )
        inst_small = TSPInstance(
            name="small_int_5",
            dimension=5,
            coords=np.zeros((5, 2), dtype=np.float64),
            dist_matrix=adj,
            round_to_int=True,
        )

        self.assertEqual(compute_bound_naive(adj, [0], 5, is_integer_metric=True), 14.0)
        self.assertEqual(compute_bound_naive(adj, [0, 3], 5, is_integer_metric=True), 16.0)

        hk_opt, _ = held_karp(adj)
        self.assertEqual(hk_opt, 16.0)

        for solver in (solve_v0, solve_v1, solve_v2, solve_v3):
            res = solver(inst_small, warm_start=None, use_gpu=False)
            self.assertAlmostEqual(res.cost, 16.0, places=6)
            self.assertTrue(inst_small.validate_tour(res.tour))
            self.assertEqual(res.extra["status"], "OPTIMAL")

    def test_all_four_versions_match_held_karp_on_eil51_n10(self) -> None:
        """Confirma convergência ao mesmo ótimo exato em v0, v1, v2 e v3 (CPU e flag GPU)."""
        for solver in (solve_v0, solve_v1, solve_v2, solve_v3):
            for use_gpu in (False, True):
                if solver is solve_v0 and use_gpu:
                    continue
                res = solver(self.inst10, warm_start="cold_start", use_gpu=use_gpu)
                with self.subTest(alg=res.algorithm, use_gpu=use_gpu):
                    self.assertTrue(self.inst10.validate_tour(res.tour))
                    self.assertAlmostEqual(res.cost, self.opt10, places=5)
                    self.assertEqual(res.extra["status"], "OPTIMAL")

    def test_memory_bound_and_speed_progression_on_berlin52_n12(self) -> None:
        """Confirma que v2 e v3 limitam a pilha em <= N e mantêm otimalidade exata em N=12."""
        r0 = solve_v0(self.inst12, warm_start="cold_start")
        r1 = solve_v1(self.inst12, warm_start="cold_start", use_gpu=True)
        r2 = solve_v2(self.inst12, warm_start="cold_start", use_gpu=False)
        r3 = solve_v3(self.inst12, warm_start="cold_start", use_gpu=False)

        for res in (r0, r1, r2, r3):
            self.assertTrue(self.inst12.validate_tour(res.tour))
            self.assertAlmostEqual(res.cost, self.opt12, places=4)

        self.assertLessEqual(r2.extra["max_queue_size"], self.inst12.dimension)
        self.assertLessEqual(r3.extra["max_queue_size"], self.inst12.dimension)
        self.assertGreater(r0.extra["max_queue_size"], self.inst12.dimension)
        self.assertLess(r3.extra["nodes_expanded"], r2.extra["nodes_expanded"])

    def test_warm_start_pruning_impact(self) -> None:
        """Confirma redução de nós expandidos com as 5 estratégias de warm-start mantendo o ótimo."""
        r_cold = solve_v3(self.inst12, warm_start="cold_start")
        r_nn = solve_v3(self.inst12, warm_start="nearest_neighbor")
        r_ch = solve_v3(self.inst12, warm_start="convex_hull_insertion")
        r_ta = solve_v3(self.inst12, warm_start="twice_around")
        r_chr = solve_v3(self.inst12, warm_start="christofides")

        for res in (r_cold, r_nn, r_ch, r_ta, r_chr):
            self.assertAlmostEqual(res.cost, self.opt12, places=4)
            self.assertTrue(self.inst12.validate_tour(res.tour))

        self.assertLess(r_ta.extra["initial_ub"], r_cold.extra["initial_ub"])
        self.assertLessEqual(r_ta.extra["nodes_expanded"], r_cold.extra["nodes_expanded"])
        self.assertLessEqual(r_nn.extra["nodes_expanded"], r_cold.extra["nodes_expanded"])
        self.assertLess(r_chr.extra["nodes_expanded"], r_cold.extra["nodes_expanded"])
        self.assertLess(r_ch.extra["nodes_expanded"], r_cold.extra["nodes_expanded"])

    def test_timeout_graceful_na_status_and_partial_metrics(self) -> None:
        """Confirma encerramento limpo no timeout com status='NA' e preservação de métricas parciais."""
        inst_large = load_tsplib_instance("berlin52")
        prof = ProfilerSession(track_memory=False)
        res = solve_v3(inst_large, warm_start="nearest_neighbor", timeout_s=0.005, profiler=prof)

        self.assertTrue(res.extra["timed_out"])
        self.assertEqual(res.extra["status"], "NA")
        self.assertGreater(res.extra["nodes_expanded"], 0)
        self.assertGreater(res.profile["process_max_rss_kb"], 0.0)
        self.assertTrue(inst_large.validate_tour(res.tour))


if __name__ == "__main__":
    unittest.main()
