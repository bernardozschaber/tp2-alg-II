"""Testes unitários de validação da Etapa 1:
- Tratamento da base TSPLIB com coordenadas 2D nativas (EUC_2D)
- Cálculo da matriz de distâncias e validação de propriedades métricas (Aula 13, p. 13)
- Profiling de tempo e memória em nível de subrotina
- Abstração da flag --use-gpu e validação algébrica do bound (Aula 12, p. 14)
"""

from __future__ import annotations

import unittest
import numpy as np

from src.core.gpu_utils import get_backend
from src.core.instance import load_benchmark_suite, load_tsplib_instance
from src.core.profiling import ProfilerSession


class TestStage1Core(unittest.TestCase):
    """Suite de testes do Checkpoint 1."""

    def test_benchmark_suite_proportions_and_native_coords(self) -> None:
        """Verifica se as 15 instâncias EUC_2D respeitam a divisão 80% (N<=42) / 20% (N in [45,75])."""
        instances = load_benchmark_suite()
        self.assertEqual(len(instances), 15)

        main_band = [inst for inst in instances if inst.band == "main"]
        stress_band = [inst for inst in instances if inst.band == "stress"]
        self.assertEqual(len(main_band), 12)  # 80% de 15 = 12
        self.assertEqual(len(stress_band), 3)  # 20% de 15 = 3

        for inst in main_band:
            self.assertGreaterEqual(inst.dimension, 10)
            self.assertLessEqual(inst.dimension, 42)
            self.assertEqual(inst.coords.shape, (inst.dimension, 2))
            self.assertEqual(inst.dist_matrix.shape, (inst.dimension, inst.dimension))

        for inst in stress_band:
            self.assertGreaterEqual(inst.dimension, 45)
            self.assertLessEqual(inst.dimension, 75)
            self.assertEqual(inst.coords.shape, (inst.dimension, 2))

    def test_euclidean_metric_properties_aula13(self) -> None:
        """Verifica as 4 propriedades métricas do TSP Euclidiano (Aula 13, slide 13) em ulysses16."""
        inst = load_tsplib_instance("ulysses16")
        d = inst.dist_matrix
        n = inst.dimension

        # 1. c(u, v) >= 0
        self.assertTrue(np.all(d >= 0.0))
        # 2. c(u, u) == 0
        self.assertTrue(np.allclose(np.diag(d), 0.0))
        # 3. Simetria: c(u, v) == c(v, u)
        self.assertTrue(np.allclose(d, d.T))
        # 4. Desigualdade triangular: c(u, v) <= c(u, w) + c(w, v)
        for u in range(n):
            for v in range(n):
                self.assertTrue(np.all(d[u, v] <= d[u, :] + d[:, v] + 1e-9))

        # Validação de rota trivial
        trivial_tour = list(range(n))
        self.assertTrue(inst.validate_tour(trivial_tour))
        self.assertGreater(inst.tour_cost(trivial_tour), 0.0)

    def test_subroutine_profiler_time_and_memory(self) -> None:
        """Verifica se o ProfilerSession registra tempo (ns/ms) e memória (bytes/KB) por subrotina."""
        profiler = ProfilerSession(track_memory=True)
        inst = load_tsplib_instance("berlin52", profiler=profiler)

        with profiler.profile("dummy_allocation"):
            arr = np.ones((256, 256), dtype=np.float64)
            _ = float(np.sum(arr))

        summary = profiler.summary_dict()
        self.assertIn("dist_matrix_euc2d", summary["subroutines"])
        self.assertIn("dummy_allocation", summary["subroutines"])
        self.assertGreater(summary["subroutines"]["dist_matrix_euc2d"]["elapsed_ns"], 0)
        self.assertGreater(summary["subroutines"]["dummy_allocation"]["peak_memory_bytes"], 0)
        self.assertEqual(inst.dimension, 52)

    def test_gpu_backend_and_aula12_slide14_bound_example(self) -> None:
        """Valida o cálculo em lote na GPU/vetorizado contra o exemplo exato da Aula 12 (slide 14).

        No slide 14 da Aula 12:
        Grafo de 5 vértices (a=0, b=1, c=2, d=3, e=4) com pesos:
        (a,b)=3, (a,c)=1, (a,d)=5, (a,e)=8
        (b,c)=6, (b,d)=7, (b,e)=9
        (c,d)=4, (c,e)=2
        (d,e)=3
        - Sem arestas incluídas: bound = teto([(1+3)+(3+6)+(1+2)+(4+3)+(2+3)]/2) = teto(28/2) = 14
        - Incluindo a aresta (a,d): bound = teto([(1+5)+(3+6)+(1+2)+(3+5)+(2+3)]/2) = teto(31/2) = 16
        """
        inf = np.inf
        adj = np.array(
            [
                [inf, 3.0, 1.0, 5.0, 8.0],  # a: menores 1, 3 -> soma 4
                [3.0, inf, 6.0, 7.0, 9.0],  # b: menores 3, 6 -> soma 9
                [1.0, 6.0, inf, 4.0, 2.0],  # c: menores 1, 2 -> soma 3
                [5.0, 7.0, 4.0, inf, 3.0],  # d: menores 3, 4 -> soma 7
                [8.0, 9.0, 2.0, 3.0, inf],  # e: menores 2, 3 -> soma 5
            ],
            dtype=np.float64,
        )

        for use_gpu in (False, True):
            backend = get_backend(use_gpu=use_gpu)
            dist_dev, min1_val, min1_idx, min2_val = backend.precompute_min_edges_device(adj)

            # Soma na raiz = 4 + 9 + 3 + 7 + 5 = 28 -> bound raiz = teto(28/2) = 14
            pair_sums = backend.to_cpu(min1_val + min2_val)
            root_sum = float(np.sum(pair_sums))
            self.assertEqual(root_sum, 28.0)
            self.assertEqual(np.ceil(root_sum / 2.0), 14.0)

            # Avalia em lote os 4 filhos da raiz a=0: candidatos k in [1 (b), 2 (c), 3 (d), 4 (e)]
            # Quando na raiz s=[0], soma dos não-visitados antes de escolher k é root_sum - pair_sums[0] = 28 - 4 = 24
            candidates = np.array([1, 2, 3, 4], dtype=np.int64)
            first_v = np.array([-1, -1, -1, -1], dtype=np.int64)
            last_v = np.array([0, 0, 0, 0], dtype=np.int64)
            p_costs = np.zeros(4, dtype=np.float64)
            unvis_sums = np.full(4, root_sum - pair_sums[0], dtype=np.float64)

            new_costs, bounds = backend.batch_evaluate_children_bounds(
                dist_dev=dist_dev,
                min1_val=min1_val,
                min1_idx=min1_idx,
                min2_val=min2_val,
                first_vertices=first_v,
                last_vertices=last_v,
                candidates=candidates,
                parent_costs=p_costs,
                unvisited_sums=unvis_sums,
                is_integer_metric=True,
            )

            # Para k = 3 (vértice d, índice 2 no lote [1, 2, 3, 4]):
            # Custo de (a, d) = 5.0 e bound = 16.0 (idêntico ao Slide 14 da Aula 12!)
            self.assertEqual(new_costs[2], 5.0)
            self.assertEqual(bounds[2], 16.0)


if __name__ == "__main__":
    unittest.main()
