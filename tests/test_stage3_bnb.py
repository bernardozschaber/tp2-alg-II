"""Checkpoint 3: Testes unitários das 4 versões graduais do Branch-and-Bound (v0 -> v1 -> v2 -> v3).

Valida:
  1. Equivalência exata de custo ótimo entre v0_naive, v1_gpu, v2_dfs_memory, v3_fast_bound
     e o oráculo independente Held-Karp.
  2. Exemplo didático de 5 vértices da Aula 12 (slides 14-16).
  3. Redução drástica de memória (max_queue_size <= N) nas versões v2 e v3 em relação a v0/v1.
  4. Impacto das estratégias de Warm-Start na poda da árvore (redução de nodes_expanded).
  5. Tratamento gracioso de timeout (status='NA' preservando contadores parciais).
"""

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
    """Suite de testes do Checkpoint 3 (Branch-and-Bound v0, v1, v2, v3)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.inst10 = load_tsplib_instance("eil51_n10")
        cls.opt10, _ = held_karp(cls.inst10.dist_matrix)
        cls.inst12 = load_tsplib_instance("berlin52_n12")
        cls.opt12, _ = held_karp(cls.inst12.dist_matrix)

    def test_aula12_slide14_5vertex_exact_optimum(self) -> None:
        """Valida as 4 versões do B&B no grafo de 5 vértices do slide 14 da Aula 12."""
        inf = np.inf
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
        dummy_coords = np.zeros((5, 2), dtype=np.float64)
        inst_slide = TSPInstance(
            name="aula12_slide14",
            dimension=5,
            coords=dummy_coords,
            dist_matrix=adj,
            round_to_int=True,
        )

        # Verifica os bounds do slide 14 na função naive:
        self.assertEqual(compute_bound_naive(adj, [0], 5, is_integer_metric=True), 14.0)
        self.assertEqual(compute_bound_naive(adj, [0, 3], 5, is_integer_metric=True), 16.0)

        hk_opt, _ = held_karp(adj)
        self.assertEqual(hk_opt, 16.0)

        for solver in (solve_v0, solve_v1, solve_v2, solve_v3):
            res = solver(inst_slide, warm_start=None, use_gpu=False)
            self.assertAlmostEqual(res.cost, 16.0, places=6)
            self.assertTrue(inst_slide.validate_tour(res.tour))
            self.assertEqual(res.extra["status"], "OPTIMAL")

    def test_all_four_versions_match_held_karp_on_eil51_n10(self) -> None:
        """Garante que v0, v1, v2 e v3 encontram exatamente o mesmo ótimo em eil51_n10."""
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
        """Verifica que v0, v1, v2, v3 encontram OPT em berlin52_n12 e que v2/v3 limitam a fila em <= N."""
        r0 = solve_v0(self.inst12, warm_start="cold_start")
        r1 = solve_v1(self.inst12, warm_start="cold_start", use_gpu=True)
        r2 = solve_v2(self.inst12, warm_start="cold_start", use_gpu=False)
        r3 = solve_v3(self.inst12, warm_start="cold_start", use_gpu=False)

        for res in (r0, r1, r2, r3):
            self.assertTrue(self.inst12.validate_tour(res.tour))
            self.assertAlmostEqual(res.cost, self.opt12, places=4)

        # Em v2 e v3 (DFS in-place), a profundidade máxima da pilha é estritamente <= N
        self.assertLessEqual(r2.extra["max_queue_size"], self.inst12.dimension)
        self.assertLessEqual(r3.extra["max_queue_size"], self.inst12.dimension)
        # Já no Best-First (v0), a fila acumula muito mais que N nós simultâneos em cold_start
        self.assertGreater(r0.extra["max_queue_size"], self.inst12.dimension)
        # E a quebra de simetria do v3 expande menos nós que o v2 com a mesma heurística
        self.assertLess(r3.extra["nodes_expanded"], r2.extra["nodes_expanded"])

    def test_warm_start_pruning_impact(self) -> None:
        """Verifica que warm-starts mais fortes reduzem drasticamente os nós expandidos no B&B."""
        r_cold = solve_v3(self.inst12, warm_start="cold_start")
        r_nn = solve_v3(self.inst12, warm_start="nearest_neighbor")
        r_ch = solve_v3(self.inst12, warm_start="convex_hull_insertion")
        r_chr = solve_v3(self.inst12, warm_start="christofides")

        for res in (r_cold, r_nn, r_ch, r_chr):
            self.assertAlmostEqual(res.cost, self.opt12, places=4)

        # Warm-start de Christofides / Fecho Convexo deve expandir muito menos nós que Cold Start
        self.assertLess(r_chr.extra["nodes_expanded"], r_cold.extra["nodes_expanded"])
        self.assertLess(r_ch.extra["nodes_expanded"], r_cold.extra["nodes_expanded"])

    def test_timeout_graceful_na_status_and_partial_metrics(self) -> None:
        """Verifica que ao atingir timeout, o B&B aborta registrando status='NA' e preservando métricas."""
        inst_large = load_tsplib_instance("berlin52")
        prof = ProfilerSession(track_memory=True)
        res = solve_v3(inst_large, warm_start="nearest_neighbor", timeout_s=0.005, profiler=prof)

        self.assertTrue(res.extra["timed_out"])
        self.assertEqual(res.extra["status"], "NA")
        self.assertGreater(res.extra["nodes_expanded"], 0)
        self.assertTrue(inst_large.validate_tour(res.tour))


if __name__ == "__main__":
    unittest.main()
