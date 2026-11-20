"""Análise de sensibilidade topológica: peso das features geométricas no gap e na poda.

Correlaciona as métricas geométricas/topológicas extraídas em `instance_features.csv` com:
  - O erro relativo percentual (`gap_pct = 100 * (cost - OPT) / OPT`) dos algoritmos
    aproximados (`twice_around_v1`, `christofides_v1`) e heurísticas (`nearest_neighbor`,
    `convex_hull_insertion`).
  - Quando disponível `bnb_warmstart.csv`, o esforço de busca (`log10(nodes_expanded)`) e a
    taxa de poda (`prune_rate`) do Branch-and-Bound.

Métodos estatísticos e preditivos aplicados para cada variável-alvo:
  1. Correlações bivariadas de Spearman (rank) e Pearson com p-valores (`scipy.stats`).
  2. Regressão Linear Múltipla Padronizada + RidgeCV (`StandardScaler` + `RidgeCV` com
     validação cruzada Leave-One-Out adequada para amostras de N=15 instâncias), fornecendo
     coeficientes padronizados β (direção e peso relativo de cada feature), R² e RMSE.
  3. Árvore de Decisão rasa (`DecisionTreeRegressor(max_depth=3)`) para extração de regras
     explícitas interpretáveis (`export_text`) e importância por redução de variância,
     complementada por `RandomForestRegressor` para estabilidade do ranking de importâncias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# Features geométricas e topológicas principais usadas como preditoras nos modelos.
DEFAULT_FEATURES: Tuple[str, ...] = (
    "aspect_ratio",
    "convex_hull_ratio",
    "spatial_entropy",
    "clark_evans_r",
    "mst_max_degree",
    "mst_odd_ratio",
    "mst_edge_cv",
    "nn_dist_cv",
)


def analyze_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
) -> Optional[Dict[str, Any]]:
    """Ajusta correlações, regressão múltipla/Ridge e Árvore de Decisão para uma coluna alvo."""
    cols = [c for c in feature_cols if c in df.columns]
    sub = df.dropna(subset=list(cols) + [target_col]).copy()
    n_samples = len(sub)
    if n_samples < 5 or not cols:
        return None

    X = sub[cols].to_numpy(dtype=np.float64)
    y = sub[target_col].to_numpy(dtype=np.float64)
    if float(np.std(y)) < 1e-12:
        return None

    # 1. Correlações bivariadas (Spearman e Pearson) por feature.
    correlations: Dict[str, Dict[str, float]] = {}
    for idx, feat in enumerate(cols):
        xf = X[:, idx]
        if float(np.std(xf)) < 1e-12:
            correlations[feat] = {
                "spearman_r": 0.0,
                "spearman_p": 1.0,
                "pearson_r": 0.0,
                "pearson_p": 1.0,
            }
            continue
        sr, sp = spearmanr(xf, y)
        pr, pp = pearsonr(xf, y)
        correlations[feat] = {
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
        }

    # 2. Regressão Múltipla Padronizada (OLS) e RidgeCV (regularização L2 contra colinearidade).
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    ols = LinearRegression().fit(X_std, y)
    y_pred_ols = ols.predict(X_std)

    alphas = np.logspace(-2, 3, 40)
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=alphas))])
    ridge_pipe.fit(X, y)
    y_pred_ridge_in = ridge_pipe.predict(X)
    ridge_model: RidgeCV = ridge_pipe.named_steps["ridge"]

    # Validação cruzada Leave-One-Out (LOO) para medir generalização fora da amostra em N=15.
    loo = LeaveOneOut()
    y_pred_ridge_loo = cross_val_predict(
        Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=alphas))]),
        X,
        y,
        cv=loo,
    )

    # 3. Árvore de Decisão rasa interpretável + Random Forest para estabilidade das importâncias.
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42)
    tree.fit(X, y)
    y_pred_tree = tree.predict(X)
    tree_rules = export_text(tree, feature_names=list(cols), decimals=3)

    rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
    rf.fit(X, y)

    # Identifica as 3 instâncias mais adversariais (maior valor do alvo) e as 3 mais favoráveis.
    ranked = sub[["instance", "n", target_col]].sort_values(target_col, ascending=False)
    top_adversarial = ranked.head(3).to_dict(orient="records")
    top_favorable = ranked.tail(3).to_dict(orient="records")

    return {
        "target": target_col,
        "n_samples": n_samples,
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "correlations": correlations,
        "ols_standardized": {
            "r2_train": float(r2_score(y, y_pred_ols)),
            "rmse_train": float(root_mean_squared_error(y, y_pred_ols)),
            "betas": {c: float(b) for c, b in zip(cols, ols.coef_)},
        },
        "ridge_standardized": {
            "alpha": float(ridge_model.alpha_),
            "r2_train": float(r2_score(y, y_pred_ridge_in)),
            "rmse_train": float(root_mean_squared_error(y, y_pred_ridge_in)),
            "r2_loo": float(r2_score(y, y_pred_ridge_loo)),
            "rmse_loo": float(root_mean_squared_error(y, y_pred_ridge_loo)),
            "betas": {c: float(b) for c, b in zip(cols, ridge_model.coef_)},
        },
        "decision_tree": {
            "r2_train": float(r2_score(y, y_pred_tree)),
            "rmse_train": float(root_mean_squared_error(y, y_pred_tree)),
            "importances": {c: float(imp) for c, imp in zip(cols, tree.feature_importances_)},
            "rules_text": tree_rules,
        },
        "random_forest": {
            "importances": {c: float(imp) for c, imp in zip(cols, rf.feature_importances_)},
        },
        "top_adversarial": top_adversarial,
        "top_favorable": top_favorable,
    }


def run_sensitivity_analysis(
    features_csv: Optional[Path] = None,
    approx_csv: Optional[Path] = None,
    bnb_warmstart_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executa a análise de sensibilidade completa e exporta JSON + tabela consolidada CSV."""
    if features_csv is None:
        features_csv = RESULTS_DIR / "instance_features.csv"
    if approx_csv is None:
        approx_csv = RESULTS_DIR / "approx_profiling.csv"
    if bnb_warmstart_csv is None:
        bnb_warmstart_csv = RESULTS_DIR / "bnb_warmstart.csv"
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df_feat = pd.read_csv(features_csv)
    merged = df_feat.copy()
    targets: List[str] = []

    # Anexa os gaps percentuais de cada algoritmo aproximado/heurística.
    if approx_csv.exists():
        df_a = pd.read_csv(approx_csv)
        alg_target_map = {
            "twice_around_v1_optimized": "gap_pct_twice_around",
            "christofides_v1_optimized": "gap_pct_christofides",
            "nearest_neighbor": "gap_pct_nearest_neighbor",
            "convex_hull_insertion": "gap_pct_convex_hull",
        }
        for alg_name, col_out in alg_target_map.items():
            sub = df_a[df_a["algorithm"] == alg_name][["instance", "gap_pct"]].rename(
                columns={"gap_pct": col_out}
            )
            if not sub.empty:
                merged = merged.merge(sub, on="instance", how="left")
                targets.append(col_out)

    # Anexa o aperto inicial lb_root_ratio (sempre disponível) como alvo de vulnerabilidade do B&B.
    if "lb_root_ratio" in merged.columns:
        merged["lb_root_gap_pct"] = 100.0 * (1.0 - merged["lb_root_ratio"])
        targets.append("lb_root_gap_pct")

    # Anexa métricas reais do B&B (nós expandidos em log10 e taxa de poda) quando presentes.
    if bnb_warmstart_csv.exists():
        df_b = pd.read_csv(bnb_warmstart_csv)
        sub_b = df_b[df_b["warm_start"] == "christofides"].copy()
        if not sub_b.empty:
            sub_b["bnb_log10_nodes"] = np.log10(np.maximum(sub_b["nodes_expanded"].astype(float), 1.0))
            merged = merged.merge(
                sub_b[["instance", "bnb_log10_nodes", "prune_rate"]],
                on="instance",
                how="left",
            )
            targets.extend(["bnb_log10_nodes", "prune_rate"])

    summary: Dict[str, Any] = {"features": list(DEFAULT_FEATURES), "models": {}}
    table_rows: List[Dict[str, Any]] = []

    for tgt in targets:
        res = analyze_target(merged, target_col=tgt, feature_cols=DEFAULT_FEATURES)
        if res is None:
            continue
        summary["models"][tgt] = res
        for feat in DEFAULT_FEATURES:
            table_rows.append(
                {
                    "target": tgt,
                    "feature": feat,
                    "spearman_r": res["correlations"][feat]["spearman_r"],
                    "spearman_p": res["correlations"][feat]["spearman_p"],
                    "pearson_r": res["correlations"][feat]["pearson_r"],
                    "ridge_beta": res["ridge_standardized"]["betas"][feat],
                    "tree_importance": res["decision_tree"]["importances"][feat],
                    "rf_importance": res["random_forest"]["importances"][feat],
                    "ridge_r2_train": res["ridge_standardized"]["r2_train"],
                    "ridge_r2_loo": res["ridge_standardized"]["r2_loo"],
                }
            )

    (output_dir / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(table_rows).to_csv(output_dir / "sensitivity_table.csv", index=False)
    merged.to_csv(output_dir / "merged_instance_metrics.csv", index=False)
    return summary
