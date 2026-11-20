"""Worker isolado em subprocesso para executar um único caso do Branch-and-Bound.

Motivação operacional:
  - Como `resource.getrusage(RUSAGE_SELF).ru_maxrss` acumula o pico máximo de memória de
    toda a vida do processo no kernel do SO, executar cada combinação `(instância, versão,
    warm_start, use_gpu)` em um subprocesso limpo garante medição fiel de Max RSS por caso,
    ausência de contaminação de Garbage Collection entre execuções e proteção controlada
    contra estouro de memória (OOM) via `RLIMIT_AS` (`--max-memory-mb`).

Entrada via linha de comando:
  `python -m src.experiments.worker --instance NAME --version v3 --warm-start christofides
   --timeout 1800 [--max-memory-mb 4096] [--use-gpu] [--track-memory] [--batch-size 128]`

Saída:
  Imprime em stdout exatamente uma linha JSON contendo todas as métricas planas do caso.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys

from src.algorithms.bnb import VERSIONS as BNB_VERSIONS
from src.core.instance import load_opt_references, load_tsplib_instance
from src.core.profiling import ProfilerSession


def _apply_memory_limit(max_memory_mb: int) -> None:
    """Impõe teto opcional de memória virtual (`RLIMIT_AS`) para capturar OOM com segurança.

    Quando `--use-gpu` está ativo, `RLIMIT_AS` não deve restringir o mapeamento de endereço
    virtual do driver CUDA (que reserva dezenas de GB virtuais sem consumir RAM física); por
    isso o teto `RLIMIT_AS` só é aplicado quando `max_memory_mb > 0` em execuções CPU.
    """
    if max_memory_mb <= 0:
        return
    limit_bytes = int(max_memory_mb) * 1024 * 1024
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY and limit_bytes > hard:
            limit_bytes = hard
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
    except (ValueError, OSError):
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Worker isolado para uma execução de B&B.")
    parser.add_argument("--instance", required=True, help="Nome da instância TSPLIB.")
    parser.add_argument("--version", required=True, choices=list(BNB_VERSIONS.keys()))
    parser.add_argument("--warm-start", default="cold_start", help="Estratégia de warm-start.")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Limite de tempo em segundos.")
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=4096,
        help="Teto de memória virtual em MiB para captura segura de OOM (0 desativa).",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Ativa backend GPU (CuPy).")
    parser.add_argument("--track-memory", action="store_true", help="Ativa tracemalloc por subrotina.")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamanho do lote na v1_gpu.")
    args = parser.parse_args(argv)

    inst = load_tsplib_instance(args.instance, use_gpu=False)
    ws = None if args.warm_start.lower() == "none" else args.warm_start
    solver = BNB_VERSIONS[args.version]

    # Aplica teto de memória somente após carregar os módulos e apenas fora do contexto CUDA.
    if not args.use_gpu and args.max_memory_mb > 0:
        _apply_memory_limit(args.max_memory_mb)

    prof = ProfilerSession(track_memory=args.track_memory)
    kwargs = {
        "warm_start": ws,
        "timeout_s": args.timeout,
        "use_gpu": args.use_gpu,
        "profiler": prof,
    }
    if args.version == "v1":
        kwargs["batch_size"] = args.batch_size

    res = solver(inst, **kwargs)
    row = res.to_row()
    row["band"] = inst.band

    # Anexa o ótimo de referência EUC_2D para cálculo direto de gap do warm-start e do incumbente.
    refs = load_opt_references()
    opt_ref = refs[inst.name]["euc2d_cost"] if inst.name in refs else float("nan")
    row["opt_euc2d_ref"] = opt_ref
    row["opt_source"] = (
        "bnb_exact"
        if row.get("status") == "OPTIMAL"
        else (refs[inst.name]["source"] if inst.name in refs else "best_known_ub")
    )
    if opt_ref > 0:
        row["initial_ub_gap_pct"] = (
            100.0 * (float(row.get("initial_ub", float("nan"))) - opt_ref) / opt_ref
        )
        row["final_cost_gap_pct"] = 100.0 * (float(row["cost"]) - opt_ref) / opt_ref

    # Em caso de timeout ou OOM, registra tempo/ótimo como "NA" preservando contadores parciais.
    row["elapsed_until_abort_s"] = row["elapsed_s"]
    aborted = bool(row.get("timed_out")) or bool(row.get("oom_aborted")) or row.get("status") != "OPTIMAL"
    if aborted:
        row["elapsed_s"] = "NA"
        row["optimal_cost"] = "NA"
    else:
        row["optimal_cost"] = row["cost"]

    expanded = int(row.get("nodes_expanded", 0))
    pruned = int(row.get("nodes_pruned", 0))
    row["prune_rate"] = (float(pruned) / float(expanded + pruned)) if (expanded + pruned) > 0 else 0.0

    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
