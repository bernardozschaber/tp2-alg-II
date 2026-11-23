"""Modelagem estatística de complexidade empírica (tempo e memória) via regressão.

Ajusta séries empíricas coletadas nas baterias experimentais (`approx_profiling.csv`,
`bnb_warmstart.csv` e `bnb_ablation.csv`) às famílias funcionais de complexidade teórica
usando `scipy.optimize.curve_fit` e avalia a qualidade de ajuste por `R²` e `RMSE`
(`sklearn.metrics`).

Famílias de curvas ajustadas:
  - `linear`     : f(x) = a·x + b                     [Θ(N): preorder, shortcut, hierholzer]
  - `nlogn`      : f(x) = a·x·log2(x) + b             [O(N log N): graham_scan]
  - `quadratic`  : f(x) = a·x² + b·x + c              [Θ(N²): prim_dense, twice_v1, NN, grafo]
  - `n2logn`     : f(x) = a·x²·log2(x) + b·x² + c     [O(N² log N): twice_v0, cheapest_insertion]
  - `cubic`      : f(x) = a·x³ + b·x² + c             [O(N³) / O(|I|³): Blossom / christofides]
  - `exponential`: log10(y) = a·x + b  (y = 10^(ax+b))[O(b^N): Branch-and-Bound tempo e nós]
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


# --- Famílias funcionais de regressão ---
def _f_linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def _f_nlogn(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x * np.log2(np.maximum(x, 2.0)) + b


def _f_quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * (x ** 2) + b * x + c


def _f_n2logn(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * (x ** 2) * np.log2(np.maximum(x, 2.0)) + b * (x ** 2) + c


def _f_cubic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * (x ** 3) + b * (x ** 2) + c


def _f_log10_exp(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Ajuste linear em escala log10: log10(y) = a*x + b <=> y = 10^b * (10^a)^x."""
    return a * x + b


MODEL_SPECS: Dict[str, Tuple[Callable[..., np.ndarray], Tuple[str, ...], str]] = {
    "linear": (_f_linear, ("a", "b"), "a*x + b"),
    "nlogn": (_f_nlogn, ("a", "b"), "a*x*log2(x) + b"),
    "quadratic": (_f_quadratic, ("a", "b", "c"), "a*x^2 + b*x + c"),
    "n2logn": (_f_n2logn, ("a", "b", "c"), "a*x^2*log2(x) + b*x^2 + c"),
    "cubic": (_f_cubic, ("a", "b", "c"), "a*x^3 + b*x^2 + c"),
    "exponential_log10": (_f_log10_exp, ("a", "b"), "log10(y) = a*x + b"),
}


@dataclass
class FitResult:
    """Resultado consolidado de um ajuste de regressão empírica."""

    series_name: str
    algorithm: str
    metric: str
    x_var: str
    model_name: str
    formula: str
    n_points: int
    r2: float
    rmse: float
    params: Dict[str, float]
    growth_base: Optional[float] = None  # Base b = 10^a para ajustes exponenciais

    def to_flat_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["params_json"] = json.dumps(d.pop("params"))
        return d


def fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    series_name: str,
    algorithm: str,
    metric: str,
    x_var: str = "n",
) -> Optional[FitResult]:
    """Ajusta uma curva teórica `(x, y)` via mínimos quadrados não lineares (`curve_fit`).

    Filtra valores NaN/infinitos (e valores `<= 0` quando o modelo é `exponential_log10`)
    e devolve `None` caso restem menos pontos do que parâmetros do modelo.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if model_name == "exponential_log10":
        valid &= y_arr > 0.0

    x_clean = x_arr[valid]
    y_clean = y_arr[valid]
    fn, param_names, formula = MODEL_SPECS[model_name]
    if len(x_clean) <= len(param_names):
        return None

    y_target = np.log10(y_clean) if model_name == "exponential_log10" else y_clean

    # Nos modelos polinomiais, os termos de crescimento (todos menos o intercepto, que é o
    # último parâmetro) ficam restritos a >= 0: um custo computacional não decresce com N.
    # Sem a restrição, ruído em tempos sub-milissegundo produzia "quadráticas" côncavas.
    if model_name == "exponential_log10":
        bounds = (-np.inf, np.inf)
    else:
        k = len(param_names)
        bounds = ([0.0] * (k - 1) + [-np.inf], [np.inf] * k)

    try:
        popt, _ = curve_fit(fn, x_clean, y_target, bounds=bounds, maxfev=10000)
    except Exception:
        return None

    y_pred = fn(x_clean, *popt)
    r2 = float(r2_score(y_target, y_pred))
    rmse = float(root_mean_squared_error(y_target, y_pred))
    params = {name: float(val) for name, val in zip(param_names, popt)}
    growth_base = float(10.0 ** params["a"]) if model_name == "exponential_log10" else None

    return FitResult(
        series_name=series_name,
        algorithm=algorithm,
        metric=metric,
        x_var=x_var,
        model_name=model_name,
        formula=formula,
        n_points=int(len(x_clean)),
        r2=r2,
        rmse=rmse,
        params=params,
        growth_base=growth_base,
    )


def fit_all_models(
    approx_csv: Optional[Path] = None,
    bnb_warmstart_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Executa todos os ajustes de complexidade temporal e espacial disponíveis nos CSVs."""
    if approx_csv is None:
        approx_csv = RESULTS_DIR / "approx_profiling.csv"
    if bnb_warmstart_csv is None:
        bnb_warmstart_csv = RESULTS_DIR / "bnb_warmstart.csv"
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[FitResult] = []

    # 1. Ajustes dos algoritmos aproximados, heurísticas e suas subrotinas críticas.
    if approx_csv.exists():
        df_a = pd.read_csv(approx_csv)

        approx_specs = [
            # Twice-around-the-tree v0 vs v1 (tempo e memória)
            ("twice_v0_total_time", "twice_around_v0_naive", "elapsed_ms", "n", "n2logn"),
            ("twice_v0_total_time_quad", "twice_around_v0_naive", "elapsed_ms", "n", "quadratic"),
            ("twice_v0_graph_mem", "twice_around_v0_naive", "build_complete_graph__peak_kb", "n", "quadratic"),
            ("twice_v1_total_time", "twice_around_v1_optimized", "elapsed_ms", "n", "quadratic"),
            ("twice_v1_mst_time", "twice_around_v1_optimized", "mst__ms", "n", "quadratic"),
            ("twice_v1_preorder_time", "twice_around_v1_optimized", "preorder__ms", "n", "linear"),
            ("twice_v1_total_mem", "twice_around_v1_optimized", "total_peak_memory_kb", "n", "linear"),
            # Christofides v0 vs v1 (tempo total, AGM, matching em função de N e de |I|, memória)
            ("chris_v0_total_time", "christofides_v0_naive", "elapsed_ms", "n", "cubic"),
            ("chris_v0_graph_mem", "christofides_v0_naive", "build_complete_graph__peak_kb", "n", "quadratic"),
            ("chris_v1_total_time", "christofides_v1_optimized", "elapsed_ms", "n", "cubic"),
            ("chris_v1_mst_time", "christofides_v1_optimized", "mst__ms", "n", "quadratic"),
            ("chris_v1_matching_vs_n", "christofides_v1_optimized", "min_weight_matching__ms", "n", "cubic"),
            ("chris_v1_matching_vs_odd", "christofides_v1_optimized", "min_weight_matching__ms", "n_odd", "cubic"),
            ("chris_v1_matching_mem_vs_odd", "christofides_v1_optimized", "min_weight_matching__peak_kb", "n_odd", "quadratic"),
            ("chris_v1_euler_time", "christofides_v1_optimized", "eulerian_circuit__ms", "n", "linear"),
            # Warm-starts construtivos
            ("nn_total_time", "nearest_neighbor", "elapsed_ms", "n", "quadratic"),
            ("ch_total_time", "convex_hull_insertion", "elapsed_ms", "n", "n2logn"),
            ("ch_graham_time", "convex_hull_insertion", "graham_scan__ms", "n", "nlogn"),
            ("ch_insert_time", "convex_hull_insertion", "cheapest_insertion__ms", "n", "n2logn"),
        ]

        for label, alg, col_y, col_x, model in approx_specs:
            sub = df_a[df_a["algorithm"] == alg]
            if sub.empty or col_y not in sub.columns or col_x not in sub.columns:
                continue
            fit = fit_curve(
                x=sub[col_x].to_numpy(dtype=np.float64),
                y=sub[col_y].to_numpy(dtype=np.float64),
                model_name=model,
                series_name=label,
                algorithm=alg,
                metric=col_y,
                x_var=col_x,
            )
            if fit is not None:
                results.append(fit)

    # 2. Ajustes exponenciais para o Branch-and-Bound (sobre instâncias resolvidas sem timeout).
    if bnb_warmstart_csv.exists():
        df_b = pd.read_csv(bnb_warmstart_csv)
        solved = df_b[df_b["status"] == "OPTIMAL"].copy()
        if not solved.empty:
            solved["elapsed_num"] = pd.to_numeric(solved["elapsed_s"], errors="coerce")
            for ws, grp in solved.groupby("warm_start"):
                fit_t = fit_curve(
                    x=grp["n"].to_numpy(dtype=np.float64),
                    y=grp["elapsed_num"].to_numpy(dtype=np.float64),
                    model_name="exponential_log10",
                    series_name=f"bnb_time_{ws}",
                    algorithm=f"bnb_v3_{ws}",
                    metric="elapsed_s",
                    x_var="n",
                )
                if fit_t is not None:
                    results.append(fit_t)

                fit_n = fit_curve(
                    x=grp["n"].to_numpy(dtype=np.float64),
                    y=grp["nodes_expanded"].to_numpy(dtype=np.float64),
                    model_name="exponential_log10",
                    series_name=f"bnb_nodes_{ws}",
                    algorithm=f"bnb_v3_{ws}",
                    metric="nodes_expanded",
                    x_var="n",
                )
                if fit_n is not None:
                    results.append(fit_n)

    df_out = pd.DataFrame([r.to_flat_dict() for r in results])
    df_out.to_csv(output_dir / "complexity_fits.csv", index=False)
    (output_dir / "complexity_fits.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return df_out
