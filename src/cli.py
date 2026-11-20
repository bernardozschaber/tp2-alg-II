"""Interface de linha de comando (CLI) para execução experimental e geração de análises.

Subcomandos disponíveis (`python -m src.cli <comando> [opções]`):
  - `prepare-opt-refs` : certifica o gabarito exato EUC_2D das 15 instâncias (`opt_references.json`).
  - `extract-features` : extrai todas as métricas geométricas/topológicas (`instance_features.csv`).
  - `run-approx`       : Bateria A (Twice-around v0/v1, Christofides v0/v1 e warm-starts — 15 instâncias).
  - `run-ablation`     : Bateria B (ablação gradual v0 -> v1 -> v2 -> v3 — 15 instâncias por padrão).
  - `run-warmstart`    : Bateria C (experimento oficial de 30 min: 15 instâncias x 5 warm-starts).
  - `run-gpu-bench`    : Bateria D (validação direta CPU vs `--use-gpu` — 15 instâncias por padrão).
  - `analyze`          : executa features + regressões de complexidade + sensibilidade + gráficos/tabelas.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from src.analysis.build_opt_reference import build_all_opt_references
from src.analysis.complexity_fit import fit_all_models
from src.analysis.features import extract_all_features
from src.analysis.plots import generate_all_plots
from src.analysis.sensitivity import run_sensitivity_analysis
from src.experiments.runner import (
    run_ablation_battery,
    run_approx_battery,
    run_gpu_battery,
    run_warmstart_battery,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Orquestrador experimental e analítico do TSP Geométrico (EUC_2D).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. Gabarito exato EUC_2D
    sub.add_parser("prepare-opt-refs", help="Gera data/tsplib/opt_references.json exato (EUC_2D).")

    # 2. Features geométricas/topológicas
    sub.add_parser("extract-features", help="Extrai features das 15 instâncias em CSV.")

    # 3. Bateria A (aproximativos + warm-starts)
    p_a = sub.add_parser("run-approx", help="Bateria A: Twice-around, Christofides e warm-starts (15 instâncias).")
    p_a.add_argument("--repeats", type=int, default=5, help="Repetições para mediana de tempo puro.")
    p_a.add_argument("--instances", nargs="*", default=None, help="Filtro opcional de instâncias (padrão: todas as 15).")

    # 4. Bateria B (ablação v0 -> v3)
    p_b = sub.add_parser("run-ablation", help="Bateria B: ablação gradual B&B v0 -> v3 (padrão: todas as 15 instâncias).")
    p_b.add_argument("--instances", nargs="*", default=None, help="Filtro opcional de instâncias (padrão: todas as 15).")
    p_b.add_argument("--versions", nargs="+", default=["v0", "v1", "v2", "v3"])
    p_b.add_argument("--warm-starts", nargs="+", default=["cold_start", "christofides"])
    p_b.add_argument("--timeout", type=float, default=120.0, help="Limite de tempo em segundos por caso.")
    p_b.add_argument("--max-memory-mb", type=int, default=4096, help="Teto de RAM (MiB) por subprocesso para captura segura de OOM.")
    p_b.add_argument("--workers", type=int, default=1, help="Número de subprocessos paralelos.")
    p_b.add_argument("--use-gpu", action="store_true")
    p_b.add_argument("--no-resume", action="store_true")

    # 5. Bateria C (warm-start oficial 30 min)
    p_c = sub.add_parser("run-warmstart", help="Bateria C: 15 instâncias x 5 warm-starts (timeout 30m).")
    p_c.add_argument("--instances", nargs="*", default=None, help="Filtro opcional de instâncias (padrão: todas as 15).")
    p_c.add_argument(
        "--warm-starts",
        nargs="+",
        default=["cold_start", "nearest_neighbor", "convex_hull_insertion", "twice_around", "christofides"],
    )
    p_c.add_argument("--version", default="v3", choices=["v0", "v1", "v2", "v3"])
    p_c.add_argument("--timeout", type=float, default=1800.0, help="Limite de tempo em segundos (padrão: 1800s = 30 min).")
    p_c.add_argument("--max-memory-mb", type=int, default=4096, help="Teto de RAM (MiB) por subprocesso.")
    p_c.add_argument("--workers", type=int, default=1, help="Número de subprocessos paralelos.")
    p_c.add_argument("--use-gpu", action="store_true")
    p_c.add_argument("--no-resume", action="store_true")

    # 6. Bateria D (validação GPU)
    p_d = sub.add_parser("run-gpu-bench", help="Bateria D: comparativo direto CPU vs --use-gpu (padrão: todas as 15 instâncias).")
    p_d.add_argument("--instances", nargs="*", default=None, help="Filtro opcional de instâncias (padrão: todas as 15).")
    p_d.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    p_d.add_argument("--warm-start", default="christofides")
    p_d.add_argument("--timeout", type=float, default=300.0)
    p_d.add_argument("--max-memory-mb", type=int, default=4096)
    p_d.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 1024])
    p_d.add_argument("--no-resume", action="store_true")

    # 7. Pipeline analítico completo
    sub.add_parser("analyze", help="Executa modelagem de complexidade, sensibilidade e geração de figuras/tabelas.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-opt-refs":
        path = build_all_opt_references()
        print(f"[OK] Gabarito exato EUC_2D gravado em: {path}")
        return 0

    if args.command == "extract-features":
        df = extract_all_features()
        print(f"[OK] Features extraídas para {len(df)} instâncias.")
        return 0

    if args.command == "run-approx":
        df = run_approx_battery(repeats=args.repeats, instance_names=args.instances)
        print(f"[OK] Bateria A concluída: {len(df)} linhas gravadas em data/results/approx_profiling.csv")
        return 0

    if args.command == "run-ablation":
        df = run_ablation_battery(
            instances=args.instances,
            versions=args.versions,
            warm_starts=args.warm_starts,
            timeout_s=args.timeout,
            max_memory_mb=args.max_memory_mb,
            use_gpu=args.use_gpu,
            workers=args.workers,
            resume=not args.no_resume,
        )
        print(f"[OK] Bateria B concluída: {len(df)} linhas em data/results/bnb_ablation.csv")
        return 0

    if args.command == "run-warmstart":
        df = run_warmstart_battery(
            instances=args.instances,
            warm_starts=args.warm_starts,
            version=args.version,
            timeout_s=args.timeout,
            max_memory_mb=args.max_memory_mb,
            use_gpu=args.use_gpu,
            workers=args.workers,
            resume=not args.no_resume,
        )
        print(f"[OK] Bateria C concluída: {len(df)} linhas em data/results/bnb_warmstart.csv")
        return 0

    if args.command == "run-gpu-bench":
        df = run_gpu_battery(
            instances=args.instances,
            versions=args.versions,
            warm_start=args.warm_start,
            timeout_s=args.timeout,
            max_memory_mb=args.max_memory_mb,
            batch_sizes=args.batch_sizes,
            resume=not args.no_resume,
        )
        print(f"[OK] Bateria D concluída: {len(df)} linhas em data/results/gpu_comparison.csv")
        return 0

    if args.command == "analyze":
        df_feat = extract_all_features()
        df_fits = fit_all_models()
        sens = run_sensitivity_analysis()
        gen = generate_all_plots()
        print(
            json.dumps(
                {
                    "features_rows": len(df_feat),
                    "complexity_fits_rows": len(df_fits),
                    "sensitivity_targets": list(sens.get("models", {}).keys()),
                    "generated_artifacts": gen,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
