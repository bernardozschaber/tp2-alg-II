"""Testes unitários da Etapa 4: gabarito DFJ, features topológicas, worker, regressões e gráficos."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analysis.build_opt_reference import solve_exact_dfj_oracle
from src.analysis.complexity_fit import fit_all_models
from src.analysis.features import extract_all_features
from src.analysis.plots import generate_all_plots
from src.analysis.sensitivity import run_sensitivity_analysis
from src.core.instance import load_tsplib_instance
from src.experiments.runner import (
    run_approx_battery,
    run_warmstart_battery,
)
from tests.oracles import held_karp


class TestStage4Analysis(unittest.TestCase):
    """Valida todos os componentes do pipeline experimental e analítico."""

    def test_dfj_oracle_matches_held_karp(self) -> None:
        """Confirma que o oráculo MILP (DFJ) produz o mesmo ótimo exato do Held-Karp."""
        for name in ("eil51_n10", "berlin52_n12"):
            inst = load_tsplib_instance(name)
            hk_cost, _ = held_karp(inst.dist_matrix)
            dfj_cost, dfj_tour = solve_exact_dfj_oracle(inst)
            self.assertTrue(inst.validate_tour(dfj_tour))
            self.assertAlmostEqual(dfj_cost, hk_cost, places=5)

    def test_feature_extraction_bounds_all_15_instances(self) -> None:
        """Confirma extração determinística e limites válidos das features nas 15 instâncias."""
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "features.csv"
            df = extract_all_features(output_csv=out_csv)
            self.assertTrue(out_csv.exists())
            self.assertEqual(len(df), 15)

            self.assertTrue((df["aspect_ratio"] >= 1.0).all())
            self.assertTrue(((df["convex_hull_ratio"] > 0.0) & (df["convex_hull_ratio"] <= 1.0)).all())
            self.assertTrue(((df["spatial_entropy"] >= 0.0) & (df["spatial_entropy"] <= 1.0 + 1e-9)).all())
            self.assertTrue((df["clark_evans_r"] > 0.0).all())
            self.assertTrue((df["mst_max_degree"] >= 1).all())
            self.assertTrue(((df["lb_root_ratio"] > 0.0) & (df["lb_root_ratio"] <= 1.0 + 1e-6)).all())

    def test_end_to_end_pipeline_approx_fits_sensitivity_and_plots(self) -> None:
        """Valida de ponta a ponta: Bateria A + worker B&B + regressões + sensibilidade + gráficos."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            res_dir = tmp_dir / "results"
            fig_dir = tmp_dir / "figures"
            tab_dir = tmp_dir / "tables"

            # 1. Features + Bateria A em todas as 15 instâncias (1 repetição rápida no teste).
            feat_csv = res_dir / "instance_features.csv"
            approx_csv = res_dir / "approx_profiling.csv"
            ws_csv = res_dir / "bnb_warmstart.csv"

            extract_all_features(output_csv=feat_csv)
            df_approx = run_approx_battery(repeats=1, output_csv=approx_csv)

            # Nenhum algoritmo aproximado ou heurística pode ter gap negativo contra o ótimo EUC_2D.
            self.assertTrue((df_approx["gap_pct"] >= -1e-4).all())
            self.assertEqual(len(df_approx), 15 * 7)

            # 2. Worker em subprocesso + checkpoint/resume em instância pequena.
            df_ws1 = run_warmstart_battery(
                instances=["eil51_n10"],
                warm_starts=["cold_start", "christofides"],
                version="v3",
                timeout_s=10.0,
                workers=2,
                resume=True,
                output_csv=ws_csv,
            )
            self.assertEqual(len(df_ws1), 2)
            self.assertTrue((df_ws1["status"] == "OPTIMAL").all())
            self.assertTrue((df_ws1["process_max_rss_kb"] > 0.0).all())

            # Nova chamada com resume=True não deve duplicar linhas já concluídas.
            df_ws2 = run_warmstart_battery(
                instances=["eil51_n10"],
                warm_starts=["cold_start", "christofides"],
                version="v3",
                timeout_s=10.0,
                workers=1,
                resume=True,
                output_csv=ws_csv,
            )
            self.assertEqual(len(df_ws2), 2)

            # 3. Modelagem de complexidade empírica (R² e RMSE).
            df_fits = fit_all_models(
                approx_csv=approx_csv,
                bnb_warmstart_csv=ws_csv,
                output_dir=res_dir,
            )
            self.assertGreaterEqual(len(df_fits), 10)
            self.assertIn("r2", df_fits.columns)
            self.assertIn("rmse", df_fits.columns)

            # 4. Análise de sensibilidade (Ridge + Árvore de Decisão + Random Forest).
            sens = run_sensitivity_analysis(
                features_csv=feat_csv,
                approx_csv=approx_csv,
                bnb_warmstart_csv=ws_csv,
                output_dir=res_dir,
            )
            self.assertIn("gap_pct_christofides", sens["models"])
            self.assertIn("gap_pct_twice_around", sens["models"])
            self.assertIn("lb_root_gap_pct", sens["models"])

            # 5. Geração de figuras vetoriais e tabelas LaTeX.
            gen = generate_all_plots(results_dir=res_dir, fig_dir=fig_dir, tables_dir=tab_dir)
            self.assertIn("fig1", gen)
            self.assertIn("fig2", gen)
            self.assertIn("fig5", gen)
            self.assertIn("tables", gen)
            for p_list in gen.values():
                for p_str in p_list:
                    self.assertTrue(Path(p_str).exists())


if __name__ == "__main__":
    unittest.main()
