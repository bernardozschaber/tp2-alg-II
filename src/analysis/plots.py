"""Geração automatizada de gráficos vetoriais (.pdf/.png) e tabelas LaTeX (.tex).

Lê os artefatos produzidos pelas baterias experimentais e pelos módulos analíticos em
`data/results/` e materializa:
  - Figuras em `report/figures/` (PDF vetorial + PNG 300 DPI):
      1. `fig1_subroutine_profiling`: tempo e memória por subrotina (Twice-around e Christofides).
      2. `fig2_complexity_regression`: dispersão empírica e curvas ajustadas com R² e RMSE.
      3. `fig3_bnb_gradual_optimization`: evolução gradual v0 -> v1 -> v2 -> v3 (tempo e memória).
      4. `fig4_warmstart_tradeoff`: nós expandidos, taxa de poda e custo/benefício dos 5 warm-starts.
      5. `fig5_sensitivity_analysis`: pesos padronizados/importâncias + mapa espacial regular vs adversarial.
      6. `fig6_gpu_validation`: comparação CPU vs GPU (`--use-gpu`) no Branch-and-Bound.
  - Tabelas em `report/tables/` (`.tex` prontas para inclusão direta via `\\input`).

Cada figura/tabela verifica a presença de seus CSVs de entrada e é gerada apenas quando os
dados correspondentes existem, permitindo uso incremental durante as execuções.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Garante diretório gravável de cache para o Matplotlib em ambientes restritos.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.complexity_fit import MODEL_SPECS
from src.core.instance import load_tsplib_instance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
TABLES_DIR = PROJECT_ROOT / "report" / "tables"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    }
)


def _save_fig(fig: plt.Figure, name: str, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def plot_subroutine_profiling(approx_csv: Path, fig_dir: Path) -> List[Path]:
    """Figura 1: decomposição de tempo e memória por subrotina em Twice-around e Christofides."""
    if not approx_csv.exists():
        return []
    df = pd.read_csv(approx_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Painel A: Christofides v1 (subrotinas em função de N)
    chr1 = df[df["algorithm"] == "christofides_v1_optimized"].sort_values("n")
    x = np.arange(len(chr1))
    labels = [f"{r['instance']}\n(N={int(r['n'])})" for _, r in chr1.iterrows()]

    mst_ms = chr1["mst__ms"].fillna(0.0).to_numpy()
    match_ms = chr1["min_weight_matching__ms"].fillna(0.0).to_numpy()
    other_ms = (
        chr1.get("odd_vertices__ms", pd.Series(0.0, index=chr1.index)).fillna(0.0).to_numpy()
        + chr1.get("eulerian_circuit__ms", pd.Series(0.0, index=chr1.index)).fillna(0.0).to_numpy()
        + chr1.get("shortcut__ms", pd.Series(0.0, index=chr1.index)).fillna(0.0).to_numpy()
    )

    axes[0].bar(x, mst_ms, label="AGM (Prim Θ(N²))", color="#2b8cbe")
    axes[0].bar(x, match_ms, bottom=mst_ms, label="Matching (Blossom O(|I|³))", color="#e34a33")
    axes[0].bar(x, other_ms, bottom=mst_ms + match_ms, label="Ímpares + Hierholzer + Shortcut", color="#74c476")
    axes[0].set_title("Christofides v1: Tempo por Subrotina")
    axes[0].set_ylabel("Tempo mediano (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", alpha=0.3)

    # Painel B: Comparação de pico de memória v0 vs v1 em Twice-around e Christofides
    for alg, lbl, style, col in [
        ("christofides_v0_naive", "Christofides v0 (NetworkX)", "--o", "#b30000"),
        ("christofides_v1_optimized", "Christofides v1 (Otimizado)", "-o", "#e34a33"),
        ("twice_around_v0_naive", "Twice-around v0 (NetworkX)", "--s", "#08519c"),
        ("twice_around_v1_optimized", "Twice-around v1 (Otimizado)", "-s", "#3182bd"),
    ]:
        sub = df[df["algorithm"] == alg].sort_values("n")
        if not sub.empty:
            axes[1].plot(
                sub["n"],
                sub["total_peak_memory_kb"],
                style,
                color=col,
                label=lbl,
                linewidth=1.6,
                markersize=4,
            )
    axes[1].set_title("Auditoria de Memória: Baseline v0 vs Refatorado v1")
    axes[1].set_xlabel("Número de vértices (N)")
    axes[1].set_ylabel("Pico de memória tracemalloc (KiB)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "fig1_subroutine_profiling", fig_dir)


def plot_complexity_regression(approx_csv: Path, fits_json: Path, fig_dir: Path) -> List[Path]:
    """Figura 2: dispersão empírica e curvas teóricas ajustadas (R² e RMSE)."""
    if not approx_csv.exists() or not fits_json.exists():
        return []
    df = pd.read_csv(approx_csv)
    fits = {item["series_name"]: item for item in json.loads(fits_json.read_text(encoding="utf-8"))}

    panels = [
        ("twice_v1_mst_time", "twice_around_v1_optimized", "n", "mst__ms", "AGM Prim Denso — Θ(N²)", "N", "Tempo (ms)"),
        ("chris_v1_matching_vs_odd", "christofides_v1_optimized", "n_odd", "min_weight_matching__ms", "Matching Blossom — O(|I|³)", "|I| (vértices ímpares)", "Tempo (ms)"),
        ("twice_v0_graph_mem", "twice_around_v0_naive", "n", "build_complete_graph__peak_kb", "Grafo Completo v0 (Memória) — Θ(N²)", "N", "Memória (KiB)"),
        ("ch_insert_time", "convex_hull_insertion", "n", "cheapest_insertion__ms", "Inserção Fecho Convexo — O(N² log N)", "N", "Tempo (ms)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (series_key, alg, col_x, col_y, title, xlabel, ylabel) in zip(axes.flat, panels):
        sub = df[df["algorithm"] == alg].dropna(subset=[col_x, col_y]).sort_values(col_x)
        if sub.empty:
            ax.set_visible(False)
            continue
        x = sub[col_x].to_numpy(dtype=np.float64)
        y = sub[col_y].to_numpy(dtype=np.float64)
        ax.scatter(x, y, color="#1f78b4", s=26, zorder=3, label="Empírico")

        fit = fits.get(series_key)
        if fit is not None:
            fn, pnames, _ = MODEL_SPECS[fit["model_name"]]
            x_grid = np.linspace(float(x.min()), float(x.max()), 150)
            popt = [fit["params"][p] for p in pnames]
            y_grid = fn(x_grid, *popt)
            ax.plot(
                x_grid,
                y_grid,
                color="#e31a1c",
                linewidth=1.8,
                label=f"Ajuste (R²={fit['r2']:.4f}, RMSE={fit['rmse']:.3f})",
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "fig2_complexity_regression", fig_dir)


def plot_bnb_ablation(ablation_csv: Path, fig_dir: Path) -> List[Path]:
    """Figura 3: evolução gradual do Branch-and-Bound (v0 -> v1 -> v2 -> v3) em tempo e memória."""
    if not ablation_csv.exists():
        return []
    df = pd.read_csv(ablation_csv)
    sub = df[df["warm_start"] == "cold_start"].copy()
    if sub.empty:
        sub = df.copy()

    sub["time_ms"] = pd.to_numeric(sub["elapsed_until_abort_s"], errors="coerce") * 1000.0
    versions_order = ["v0_naive", "v1_gpu", "v2_dfs_memory", "v3_fast_bound"]
    present_insts = list(sub.sort_values("n")["instance"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    width = 0.18
    x = np.arange(len(present_insts))
    colors = ["#b30000", "#e34a33", "#2b8cbe", "#238b45"]

    for idx, (ver, col) in enumerate(zip(versions_order, colors)):
        v_df = sub[sub["version"] == ver].set_index("instance")
        vals_t = [float(v_df.loc[i, "time_ms"]) if i in v_df.index else np.nan for i in present_insts]
        vals_q = [float(v_df.loc[i, "max_queue_size"]) if i in v_df.index else np.nan for i in present_insts]
        offset = (idx - 1.5) * width
        axes[0].bar(x + offset, vals_t, width=width, label=ver, color=col)
        axes[1].bar(x + offset, vals_q, width=width, label=ver, color=col)

    axes[0].set_yscale("log")
    axes[0].set_title("Evolução Gradual B&B: Tempo Real Puro (escala log)")
    axes[0].set_ylabel("Tempo (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(present_insts, rotation=25, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_yscale("log")
    axes[1].set_title("Pico de Nós Ativos em Memória (max_queue_size, escala log)")
    axes[1].set_ylabel("Máximo de nós simultâneos na fila/pilha")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(present_insts, rotation=25, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "fig3_bnb_gradual_optimization", fig_dir)


def plot_warmstart_tradeoff(warmstart_csv: Path, fig_dir: Path) -> List[Path]:
    """Figura 4: impacto dos 5 warm-starts na expansão de nós e no tempo total do B&B."""
    if not warmstart_csv.exists():
        return []
    df = pd.read_csv(warmstart_csv)
    if df.empty:
        return []

    ws_order = ["cold_start", "nearest_neighbor", "twice_around", "christofides", "convex_hull_insertion"]
    ws_labels = ["Cold Start", "Nearest\nNeighbor", "Twice-\nAround", "Christofides", "Convex Hull\nInsertion"]
    colors = ["#636363", "#3182bd", "#756bb1", "#e6550d", "#31a354"]

    # Seleciona instâncias onde ao menos 1 estratégia convergiu ou todas rodaram.
    inst_order = list(df.sort_values("n")["instance"].unique())[:8]
    sub = df[df["instance"].isin(inst_order)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # Painel A: redução percentual média de nós expandidos em relação ao cold_start (nas resolvidas)
    solved = df[df["status"] == "OPTIMAL"].copy()
    if not solved.empty:
        cold_map = solved[solved["warm_start"] == "cold_start"].set_index("instance")["nodes_expanded"]
        reductions = []
        for ws in ws_order:
            w_df = solved[solved["warm_start"] == ws].set_index("instance")["nodes_expanded"]
            common = cold_map.index.intersection(w_df.index)
            if len(common) > 0:
                rel = 100.0 * (1.0 - (w_df.loc[common] / np.maximum(cold_map.loc[common], 1.0)))
                reductions.append(float(rel.mean()))
            else:
                reductions.append(0.0)
        axes[0].bar(ws_labels, reductions, color=colors)
        axes[0].set_title("Redução Média de Nós Expandidos vs. Cold Start")
        axes[0].set_ylabel("Redução de nós expandidos (%)")
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].grid(axis="y", alpha=0.3)

    # Painel B: nós expandidos (escala log) por instância nas 5 estratégias
    x = np.arange(len(inst_order))
    width = 0.15
    for idx, (ws, lbl, col) in enumerate(zip(ws_order, ws_labels, colors)):
        w_df = sub[sub["warm_start"] == ws].set_index("instance")
        vals = [float(w_df.loc[i, "nodes_expanded"]) if i in w_df.index else np.nan for i in inst_order]
        axes[1].bar(x + (idx - 2) * width, vals, width=width, label=lbl.replace("\n", " "), color=col)
    axes[1].set_yscale("log")
    axes[1].set_title("Nós Expandidos na Árvore B&B por Estratégia de Warm-Start")
    axes[1].set_ylabel("Nós expandidos (escala log)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(inst_order, rotation=35, ha="right", fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "fig4_warmstart_tradeoff", fig_dir)


def plot_sensitivity_analysis(sensitivity_json: Path, fig_dir: Path) -> List[Path]:
    """Figura 5: pesos das features geométricas no gap + contraste visual regular vs adversarial."""
    if not sensitivity_json.exists():
        return []
    data = json.loads(sensitivity_json.read_text(encoding="utf-8"))
    models = data.get("models", {})
    if not models:
        return []

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    # Painel A: coeficientes padronizados Ridge para gap do Christofides e Twice-around
    feats = data["features"]
    y_pos = np.arange(len(feats))
    h = 0.36
    chr_m = models.get("gap_pct_christofides")
    ta_m = models.get("gap_pct_twice_around")
    if chr_m:
        betas_c = [chr_m["ridge_standardized"]["betas"][f] for f in feats]
        axes[0].barh(y_pos - h / 2, betas_c, height=h, label="Christofides (β)", color="#e34a33")
    if ta_m:
        betas_t = [ta_m["ridge_standardized"]["betas"][f] for f in feats]
        axes[0].barh(y_pos + h / 2, betas_t, height=h, label="Twice-around (β)", color="#2b8cbe")
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(feats)
    axes[0].set_title("Sensibilidade do Gap (Coeficientes Ridge β)")
    axes[0].set_xlabel("Peso padronizado β (>0 piora gap)")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="x", alpha=0.3)

    # Painéis B e C: geometria espacial de uma instância regular (eil51_n18) vs adversarial (ulysses16)
    for ax, iname, subtitle in [
        (axes[1], "eil51_n18", "Regular: eil51_n18 (N=18)\nLB/OPT=0.909 -> B&B 0.36s"),
        (axes[2], "ulysses16", "Adversarial: ulysses16 (N=16)\nLB/OPT=0.662 -> Clusters + Outlier"),
    ]:
        inst = load_tsplib_instance(iname)
        ax.scatter(inst.coords[:, 0], inst.coords[:, 1], s=32, color="#1f78b4", edgecolors="black", linewidth=0.5)
        ax.set_title(subtitle)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "fig5_sensitivity_analysis", fig_dir)


def plot_gpu_validation(gpu_csv: Path, fig_dir: Path) -> List[Path]:
    """Figura 6: comparação direta CPU vs GPU (`--use-gpu`) no Branch-and-Bound."""
    if not gpu_csv.exists():
        return []
    df = pd.read_csv(gpu_csv)
    if df.empty:
        return []

    df["time_ms"] = pd.to_numeric(df["elapsed_until_abort_s"], errors="coerce") * 1000.0
    labels = [f"{r['instance']}\n{r['version']} (B={int(r.get('batch_size', 128))})" for _, r in df.drop_duplicates(["instance", "version", "batch_size"]).iterrows()]
    keys = list(df.drop_duplicates(["instance", "version", "batch_size"])[["instance", "version", "batch_size"]].itertuples(index=False, name=None))

    cpu_vals = []
    gpu_vals = []
    for inst_n, ver, bs in keys:
        c = df[(df["instance"] == inst_n) & (df["version"] == ver) & (df["batch_size"] == bs) & (~df["use_gpu"])]
        g = df[(df["instance"] == inst_n) & (df["version"] == ver) & (df["batch_size"] == bs) & (df["use_gpu"])]
        cpu_vals.append(float(c["time_ms"].iloc[0]) if not c.empty else np.nan)
        gpu_vals.append(float(g["time_ms"].iloc[0]) if not g.empty else np.nan)

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w / 2, cpu_vals, width=w, label="CPU (use_gpu=False)", color="#2b8cbe")
    ax.bar(x + w / 2, gpu_vals, width=w, label="GPU / Flag --use-gpu", color="#31a354")
    ax.set_yscale("log")
    ax.set_title("Validação Experimental da Flag --use-gpu no Branch-and-Bound")
    ax.set_ylabel("Tempo (ms, escala log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "fig6_gpu_validation", fig_dir)


def _escape_latex(val: object, float_fmt: str = "%.4f") -> str:
    """Formata e escapa valores escalares para células de tabela LaTeX (sem depender de Jinja2)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NA"
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        formatted = float_fmt % fval
        if formatted.startswith("-") and set(formatted[1:]) <= {"0", "."}:
            formatted = formatted[1:]
        return formatted
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    text = str(val)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(old, new)
    return text


def _dataframe_to_latex(df: pd.DataFrame, float_fmt: str = "%.4f") -> str:
    """Serializa um DataFrame em tabela LaTeX limpa com réguas `booktabs`."""
    cols = list(df.columns)
    align = "".join(
        "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols
    )
    header = " & ".join(_escape_latex(c, float_fmt) for c in cols) + r" \\"
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = [_escape_latex(row[c], float_fmt) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def export_latex_tables(results_dir: Path, tables_dir: Path) -> List[Path]:
    """Exporta tabelas resumo em formato LaTeX prontas para inclusão direta via `\\input`."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    fits_csv = results_dir / "complexity_fits.csv"
    if fits_csv.exists():
        df_f = pd.read_csv(fits_csv)
        cols = ["series_name", "x_var", "model_name", "n_points", "r2", "rmse"]
        p = tables_dir / "tab_complexity_fits.tex"
        p.write_text(_dataframe_to_latex(df_f[cols], float_fmt="%.4f"), encoding="utf-8")
        written.append(p)

    approx_csv = results_dir / "approx_profiling.csv"
    if approx_csv.exists():
        df_a = pd.read_csv(approx_csv)
        piv = (
            df_a[
                df_a["algorithm"].isin(
                    [
                        "twice_around_v1_optimized",
                        "christofides_v1_optimized",
                        "nearest_neighbor",
                        "convex_hull_insertion",
                    ]
                )
            ]
            .pivot(index=["instance", "n"], columns="algorithm", values="gap_pct")
            .reset_index()
            .sort_values("n")
        )
        p = tables_dir / "tab_approx_gaps.tex"
        p.write_text(_dataframe_to_latex(piv, float_fmt="%.2f"), encoding="utf-8")
        written.append(p)

    sens_csv = results_dir / "sensitivity_table.csv"
    if sens_csv.exists():
        df_s = pd.read_csv(sens_csv)
        cols_s = ["target", "feature", "spearman_r", "ridge_beta", "tree_importance", "rf_importance"]
        p = tables_dir / "tab_sensitivity.tex"
        p.write_text(_dataframe_to_latex(df_s[cols_s], float_fmt="%.3f"), encoding="utf-8")
        written.append(p)

    ablation_csv = results_dir / "bnb_ablation.csv"
    if ablation_csv.exists():
        df_ab = pd.read_csv(ablation_csv, keep_default_na=False)
        cols_ab = [
            c
            for c in [
                "instance",
                "n",
                "version",
                "warm_start",
                "status",
                "elapsed_s",
                "nodes_expanded",
                "max_queue_size",
                "process_max_rss_kb",
            ]
            if c in df_ab.columns
        ]
        p = tables_dir / "tab_bnb_ablation.tex"
        p.write_text(_dataframe_to_latex(df_ab[cols_ab].sort_values(["n", "version"]), float_fmt="%.2f"), encoding="utf-8")
        written.append(p)

    ws_csv = results_dir / "bnb_warmstart.csv"
    if ws_csv.exists():
        df_w = pd.read_csv(ws_csv, keep_default_na=False)
        cols_w = [
            c
            for c in [
                "instance",
                "n",
                "warm_start",
                "status",
                "elapsed_s",
                "initial_ub",
                "nodes_expanded",
                "nodes_pruned",
                "prune_rate",
            ]
            if c in df_w.columns
        ]
        p = tables_dir / "tab_bnb_warmstart.tex"
        p.write_text(_dataframe_to_latex(df_w[cols_w].sort_values(["n", "warm_start"]), float_fmt="%.3f"), encoding="utf-8")
        written.append(p)

    return written


def generate_all_plots(
    results_dir: Optional[Path] = None,
    fig_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Gera todas as figuras e tabelas cujos dados de entrada existam em `results_dir`."""
    res_dir = results_dir or RESULTS_DIR
    f_dir = fig_dir or FIGURES_DIR
    t_dir = tables_dir or TABLES_DIR

    generated: Dict[str, List[str]] = {}
    for key, paths in [
        ("fig1", plot_subroutine_profiling(res_dir / "approx_profiling.csv", f_dir)),
        ("fig2", plot_complexity_regression(res_dir / "approx_profiling.csv", res_dir / "complexity_fits.json", f_dir)),
        ("fig3", plot_bnb_ablation(res_dir / "bnb_ablation.csv", f_dir)),
        ("fig4", plot_warmstart_tradeoff(res_dir / "bnb_warmstart.csv", f_dir)),
        ("fig5", plot_sensitivity_analysis(res_dir / "sensitivity_summary.json", f_dir)),
        ("fig6", plot_gpu_validation(res_dir / "gpu_comparison.csv", f_dir)),
        ("tables", export_latex_tables(res_dir, t_dir)),
    ]:
        if paths:
            generated[key] = [str(p) for p in paths]
    return generated
