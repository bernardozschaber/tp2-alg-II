"""Testes unitários da Etapa 1: base TSPLIB EUC_2D, profiling (tracemalloc + RSS) e backend GPU."""

from __future__ import annotations

import unittest

import numpy as np

from src.core.gpu_utils import get_backend
from src.core.instance import (
    load_benchmark_suite,
    load_opt_references,
    load_tsplib_instance,
)
from src.core.profiling import ProfilerSession


class TestStage1Core(unittest.TestCase):
    """Validações da camada de dados, instrumentação e backend numérico."""

    def test_benchmark_suite_proportions_and_native_coords(self) -> None:
        """Confirma 15 instâncias EUC_2D divididas em 12 (N<=40) e 3 (45<=N<=75)."""
        instances = load_benchmark_suite()
        self.assertEqual(len(instances), 15)

        main_band = [inst for inst in instances if inst.band == "main"]
        stress_band = [inst for inst in instances if inst.band == "stress"]
        self.assertEqual(len(main_band), 12)
        self.assertEqual(len(stress_band), 3)

        for inst in main_band:
            self.assertGreaterEqual(inst.dimension, 10)
            self.assertLessEqual(inst.dimension, 40)
            self.assertEqual(inst.coords.shape, (inst.dimension, 2))
            self.assertEqual(inst.dist_matrix.shape, (inst.dimension, inst.dimension))

        for inst in stress_band:
            self.assertGreaterEqual(inst.dimension, 45)
            self.assertLessEqual(inst.dimension, 75)
            self.assertEqual(inst.coords.shape, (inst.dimension, 2))

        # Verifica inferência automática de faixa ao carregar instância isoladamente pelo nome.
        self.assertEqual(load_tsplib_instance("berlin52").band, "stress")
        self.assertEqual(load_tsplib_instance("burma14").band, "main")

    def test_opt_references_valid_euc2d_tours(self) -> None:
        """Verifica que todas as rotas em `opt_references.json` são válidas e têm custo consistente."""
        refs = load_opt_references()
        for inst in load_benchmark_suite():
            name = inst.name
            self.assertIn(name, refs)
            tour = refs[name]["tour"]
            self.assertTrue(inst.validate_tour(tour))
            self.assertAlmostEqual(inst.tour_cost(tour), refs[name]["euc2d_cost"], places=4)
            self.assertIn(refs[name]["source"], ("tsplib_opt_tour", "exact_euc2d_oracle"))

    def test_euclidean_metric_properties(self) -> None:
        """Verifica não-negatividade, diagonal nula, simetria e desigualdade triangular em ulysses16."""
        inst = load_tsplib_instance("ulysses16")
        d = inst.dist_matrix
        n = inst.dimension

        self.assertTrue(np.all(d >= 0.0))
        self.assertTrue(np.allclose(np.diag(d), 0.0))
        self.assertTrue(np.allclose(d, d.T))
        for u in range(n):
            for v in range(n):
                self.assertTrue(np.all(d[u, v] <= d[u, :] + d[:, v] + 1e-9))

        trivial_tour = list(range(n))
        self.assertTrue(inst.validate_tour(trivial_tour))
        self.assertGreater(inst.tour_cost(trivial_tour), 0.0)

    def test_subroutine_profiler_tracemalloc_and_rss(self) -> None:
        """Valida captura de tempo/memória (tracemalloc) e coleta de Max RSS mesmo sem tracemalloc."""
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
        self.assertGreater(summary["process_max_rss_kb"], 0.0)
        self.assertEqual(inst.dimension, 52)

        # Modo leve (sem tracemalloc): ainda deve registrar Max RSS positivo via SO.
        fast_prof = ProfilerSession(track_memory=False)
        with fast_prof.profile("noop"):
            _ = 1 + 1
        fast_summary = fast_prof.summary_dict()
        self.assertEqual(fast_summary["total_peak_memory_bytes"], 0)
        self.assertGreater(fast_summary["process_max_rss_kb"], 0.0)

    def test_gpu_backend_batch_lower_bound_exact_values(self) -> None:
        """Valida o cálculo em lote de custos e limites inferiores em grafo simétrico de 5 vértices.

        Matriz de pesos inteiros (0=a, 1=b, 2=c, 3=d, 4=e):
          - Sem arestas fixadas: soma das 2 menores por vértice = 4+9+3+7+5 = 28 -> bound = 14.
          - Com aresta (0, 3) de peso 5 fixada: soma = (1+5)+(3+6)+(1+2)+(3+5)+(2+3) = 31 -> bound = 16.
        """
        inf = np.inf
        adj = np.array(
            [
                [inf, 3.0, 1.0, 5.0, 8.0],
                [3.0, inf, 6.0, 7.0, 9.0],
                [1.0, 6.0, inf, 4.0, 2.0],
                [5.0, 7.0, 4.0, inf, 3.0],
                [8.0, 9.0, 2.0, 3.0, inf],
            ],
            dtype=np.float64,
        )

        for use_gpu in (False, True):
            backend = get_backend(use_gpu=use_gpu)
            dist_dev, min1_val, min1_idx, min2_val = backend.precompute_min_edges_device(adj)

            pair_sums = backend.to_cpu(min1_val + min2_val)
            root_sum = float(np.sum(pair_sums))
            self.assertEqual(root_sum, 28.0)
            self.assertEqual(np.ceil(root_sum / 2.0 - 1e-9), 14.0)

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

            # Candidato k = 3 (índice 2 do lote): aresta (0, 3) de custo 5 e bound 16.
            self.assertEqual(new_costs[2], 5.0)
            self.assertEqual(bounds[2], 16.0)


if __name__ == "__main__":
    unittest.main()
