"""Executor das 4 baterias experimentais com checkpointing em CSV, OOM seguro e paralelismo.

Todas as 4 baterias operam por padrão sobre TODAS as 15 instâncias da base (permitindo
filtrar subconjuntos via `--instances` para depuração):
  - `run_approx_battery`    : Bateria A — Twice-around (v0/v1), Christofides (v0/v1) e
                              warm-starts (5 repetições de tempo puro + 1 passada `tracemalloc`).
  - `run_ablation_battery`  : Bateria B — Evolução gradual do B&B (v0 -> v1 -> v2 -> v3)
                              nas 15 instâncias, registrando explicitamente onde `v0`/`v1`
                              colapsam por estouro de memória (`OOM`) ou tempo (`NA`).
  - `run_warmstart_battery` : Bateria C — Experimento oficial das 15 instâncias x 5
                              estratégias de warm-start no B&B `v3` (timeout padrão 1800 s).
  - `run_gpu_battery`       : Bateria D — Comparação direta CPU vs `--use-gpu` nas versões
                              `v1`, `v2` e `v3` do B&B sobre as 15 instâncias.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.algorithms.christofides import solve_v0 as chr_v0, solve_v1 as chr_v1
from src.algorithms.twice_around import solve_v0 as ta_v0, solve_v1 as ta_v1
from src.algorithms.warm_start import (
    WARM_STARTS,
    cold_start,
    convex_hull_insertion,
    nearest_neighbor,
)
from src.core.instance import load_benchmark_suite, load_opt_references, load_tsplib_instance
from src.core.profiling import ProfilerSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def _version_suffix(ver: str) -> str:
    return {"v0": "naive", "v1": "gpu", "v2": "dfs_memory", "v3": "fast_bound"}[ver]


def _run_bnb_subprocess(
    instance_name: str,
    version: str,
    warm_start: str,
    timeout_s: float,
    use_gpu: bool = False,
    track_memory: bool = False,
    batch_size: int = 128,
    max_memory_mb: int = 4096,
) -> Dict[str, Any]:
    """Dispara `src.experiments.worker` em subprocesso limpo com proteção contra OOM/SIGKILL.

    Se o subprocesso encerrar normalmente, retorna o JSON impresso em stdout. Caso o kernel
    encerre o processo abruptamente por OOM (`SIGKILL`, exit code -9/137) ou erro fatal,
    devolve uma linha padronizada com `status="OOM_KILLED"` e `elapsed_s="NA"` sem derrubar
    o restante da bateria experimental.
    """
    cmd = [
        sys.executable,
        "-m",
        "src.experiments.worker",
        "--instance",
        instance_name,
        "--version",
        version,
        "--warm-start",
        warm_start,
        "--timeout",
        str(timeout_s),
        "--batch-size",
        str(batch_size),
        "--max-memory-mb",
        str(max_memory_mb),
    ]
    if use_gpu:
        cmd.append("--use-gpu")
    if track_memory:
        cmd.append("--track-memory")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s + 60.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        inst = load_tsplib_instance(instance_name)
        return {
            "algorithm": f"bnb_{version}_{_version_suffix(version)}",
            "version": f"{version}_{_version_suffix(version)}",
            "instance": instance_name,
            "n": inst.dimension,
            "band": inst.band,
            "warm_start": warm_start,
            "use_gpu": use_gpu,
            "status": "NA",
            "timed_out": True,
            "oom_aborted": False,
            "elapsed_s": "NA",
            "optimal_cost": "NA",
            "elapsed_until_abort_s": timeout_s,
            "nodes_expanded": 0,
            "nodes_pruned": 0,
            "max_queue_size": 0,
            "process_max_rss_kb": float("nan"),
        }

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode == 0 and lines:
        return json.loads(lines[-1])

    # Captura de rede de segurança caso o SO mate o subprocesso por OOM (-9 / 137) ou MemoryError não tratado.
    inst = load_tsplib_instance(instance_name)
    is_oom = proc.returncode in (-9, 137) or "MemoryError" in proc.stderr or "OutOfMemory" in proc.stderr
    return {
        "algorithm": f"bnb_{version}_{_version_suffix(version)}",
        "version": f"{version}_{_version_suffix(version)}",
        "instance": instance_name,
        "n": inst.dimension,
        "band": inst.band,
        "warm_start": warm_start,
        "use_gpu": use_gpu,
        "status": "OOM" if is_oom else f"ERR_{proc.returncode}",
        "timed_out": False,
        "oom_aborted": is_oom,
        "elapsed_s": "NA",
        "optimal_cost": "NA",
        "elapsed_until_abort_s": float("nan"),
        "nodes_expanded": 0,
        "nodes_pruned": 0,
        "max_queue_size": 0,
        "process_max_rss_kb": float(max_memory_mb * 1024) if is_oom and max_memory_mb > 0 else float("nan"),
    }


def _load_existing_keys(csv_path: Path, key_cols: Sequence[str]) -> Tuple[List[Dict[str, Any]], set]:
    """Lê linhas já gravadas em um CSV de checkpoint e devolve o conjunto de chaves concluídas."""
    if not csv_path.exists():
        return [], set()
    df = pd.read_csv(csv_path, keep_default_na=False)
    rows = df.to_dict(orient="records")
    done = {tuple(str(r.get(c, "")) for c in key_cols) for r in rows}
    return rows, done


def _save_checkpoint(csv_path: Path, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Persiste a lista completa de linhas em disco de forma atômica via arquivo temporário."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    return df


def _resolve_instances(instance_names: Optional[Sequence[str]]) -> List[str]:
    """Retorna todas as 15 instâncias do benchmark quando `instance_names` é None."""
    if instance_names is not None and len(instance_names) > 0:
        return list(instance_names)
    return [inst.name for inst in load_benchmark_suite()]


def run_approx_battery(
    repeats: int = 5,
    output_csv: Optional[Path] = None,
    instance_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Bateria A: profiling de tempo (mediana de `repeats` execuções) e memória (`tracemalloc`)."""
    if output_csv is None:
        output_csv = RESULTS_DIR / "approx_profiling.csv"

    refs = load_opt_references()
    suite = load_benchmark_suite()
    if instance_names is not None and len(instance_names) > 0:
        allowed = set(instance_names)
        suite = [inst for inst in suite if inst.name in allowed]

    solvers = [
        ("twice_around_v0_naive", ta_v0),
        ("twice_around_v1_optimized", ta_v1),
        ("christofides_v0_naive", chr_v0),
        ("christofides_v1_optimized", chr_v1),
        ("cold_start", cold_start),
        ("nearest_neighbor", nearest_neighbor),
        ("convex_hull_insertion", convex_hull_insertion),
    ]

    rows: List[Dict[str, Any]] = []
    for inst in suite:
        opt_ref = refs[inst.name]["euc2d_cost"] if inst.name in refs else float("nan")
        opt_src = refs[inst.name]["source"] if inst.name in refs else "unknown"

        for alg_label, fn in solvers:
            timed_runs = []
            for _ in range(max(1, repeats)):
                prof_t = ProfilerSession(track_memory=False)
                timed_runs.append(fn(inst, prof_t))

            prof_m = ProfilerSession(track_memory=True)
            mem_run = fn(inst, prof_m)

            base = mem_run.to_row()
            base["algorithm"] = alg_label
            base["band"] = inst.band
            base["elapsed_s"] = statistics.median(r.elapsed_s for r in timed_runs)
            base["elapsed_ms"] = base["elapsed_s"] * 1000.0
            base["elapsed_std_ms"] = (
                statistics.stdev(r.elapsed_s * 1000.0 for r in timed_runs)
                if len(timed_runs) > 1
                else 0.0
            )
            base["repeats"] = max(1, repeats)

            sub_names = list(timed_runs[0].profile.get("subroutines", {}).keys())
            for sub in sub_names:
                base[f"{sub}__ms"] = statistics.median(
                    r.profile["subroutines"][sub]["elapsed_ms"] for r in timed_runs
                )

            base["opt_euc2d_ref"] = opt_ref
            base["opt_source"] = opt_src
            base["approx_ratio"] = (base["cost"] / opt_ref) if opt_ref > 0 else float("nan")
            base["gap_pct"] = (
                100.0 * (base["cost"] - opt_ref) / opt_ref if opt_ref > 0 else float("nan")
            )
            rows.append(base)

    return _save_checkpoint(output_csv, rows)


def run_ablation_battery(
    instances: Optional[Sequence[str]] = None,
    versions: Sequence[str] = ("v0", "v1", "v2", "v3"),
    warm_starts: Sequence[str] = ("cold_start", "christofides"),
    timeout_s: float = 120.0,
    max_memory_mb: int = 4096,
    use_gpu: bool = False,
    workers: int = 1,
    tracemalloc_max_n: int = 12,
    resume: bool = True,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Bateria B: comparação gradual `v0 -> v1 -> v2 -> v3` sobre todas as 15 instâncias.

    Registra empiricamente o ponto exato de colapso por estouro de memória (`OOM`) ou
    tempo (`NA`) nas versões iniciais (`v0`/`v1`) frente às versões refatoradas (`v2`/`v3`).
    """
    if output_csv is None:
        output_csv = RESULTS_DIR / "bnb_ablation.csv"

    target_names = _resolve_instances(instances)
    key_cols = ("instance", "version", "warm_start", "use_gpu")
    rows, done = _load_existing_keys(output_csv, key_cols) if resume else ([], set())

    pending: List[Tuple[str, str, str, bool]] = []
    for inst_name in target_names:
        for ver in versions:
            ver_gpu = use_gpu if ver != "v0" else False
            ver_full = f"{ver}_{_version_suffix(ver)}"
            for ws in warm_starts:
                key = (inst_name, ver_full, ws, str(ver_gpu))
                if key not in done:
                    pending.append((inst_name, ver, ws, ver_gpu))

    if not pending:
        return pd.DataFrame(rows)

    def _exec_one(item: Tuple[str, str, str, bool]) -> Dict[str, Any]:
        inst_name, ver, ws, ver_gpu = item
        row = _run_bnb_subprocess(
            instance_name=inst_name,
            version=ver,
            warm_start=ws,
            timeout_s=timeout_s,
            use_gpu=ver_gpu,
            track_memory=False,
            max_memory_mb=max_memory_mb,
        )
        if int(row["n"]) <= tracemalloc_max_n and row.get("status") == "OPTIMAL":
            mem_row = _run_bnb_subprocess(
                instance_name=inst_name,
                version=ver,
                warm_start=ws,
                timeout_s=timeout_s,
                use_gpu=ver_gpu,
                track_memory=True,
                max_memory_mb=max_memory_mb,
            )
            row["tracemalloc_peak_kb"] = mem_row.get("total_peak_memory_kb", 0.0)
            row["bnb_tree_search__peak_kb"] = mem_row.get("bnb_tree_search__peak_kb", 0.0)
        else:
            row["tracemalloc_peak_kb"] = float("nan")
        return row

    if workers <= 1:
        for item in pending:
            rows.append(_exec_one(item))
            _save_checkpoint(output_csv, rows)
    else:
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_exec_one, item) for item in pending]
            for fut in as_completed(futs):
                row = fut.result()
                with lock:
                    rows.append(row)
                    _save_checkpoint(output_csv, rows)

    return pd.DataFrame(rows)


def run_warmstart_battery(
    instances: Optional[Sequence[str]] = None,
    warm_starts: Sequence[str] = tuple(WARM_STARTS.keys()),
    version: str = "v3",
    timeout_s: float = 1800.0,
    max_memory_mb: int = 4096,
    use_gpu: bool = False,
    workers: int = 1,
    resume: bool = True,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Bateria C: experimento oficial das 15 instâncias x 5 estratégias de warm-start (30 min)."""
    if output_csv is None:
        output_csv = RESULTS_DIR / "bnb_warmstart.csv"

    target_names = _resolve_instances(instances)
    key_cols = ("instance", "warm_start")
    rows, done = _load_existing_keys(output_csv, key_cols) if resume else ([], set())

    pending: List[Tuple[str, str]] = []
    for inst_name in target_names:
        for ws in warm_starts:
            if (inst_name, ws) not in done:
                pending.append((inst_name, ws))

    if not pending:
        return pd.DataFrame(rows)

    if workers <= 1:
        for inst_name, ws in pending:
            row = _run_bnb_subprocess(
                instance_name=inst_name,
                version=version,
                warm_start=ws,
                timeout_s=timeout_s,
                use_gpu=use_gpu,
                track_memory=False,
                max_memory_mb=max_memory_mb,
            )
            rows.append(row)
            _save_checkpoint(output_csv, rows)
    else:
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(
                    _run_bnb_subprocess,
                    inst_name,
                    version,
                    ws,
                    timeout_s,
                    use_gpu,
                    False,
                    128,
                    max_memory_mb,
                ): (inst_name, ws)
                for inst_name, ws in pending
            }
            for fut in as_completed(future_map):
                row = fut.result()
                with lock:
                    rows.append(row)
                    _save_checkpoint(output_csv, rows)

    return pd.DataFrame(rows)


def run_gpu_battery(
    instances: Optional[Sequence[str]] = None,
    versions: Sequence[str] = ("v1", "v2", "v3"),
    warm_start: str = "christofides",
    timeout_s: float = 300.0,
    max_memory_mb: int = 4096,
    batch_sizes: Sequence[int] = (128, 1024),
    resume: bool = True,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Bateria D: validação empírica CPU (`use_gpu=False`) vs GPU (`use_gpu=True`) nas 15 instâncias."""
    if output_csv is None:
        output_csv = RESULTS_DIR / "gpu_comparison.csv"

    target_names = _resolve_instances(instances)
    key_cols = ("instance", "version", "use_gpu", "batch_size")
    rows, done = _load_existing_keys(output_csv, key_cols) if resume else ([], set())

    for inst_name in target_names:
        for ver in versions:
            bs_list: Iterable[int] = batch_sizes if ver == "v1" else (128,)
            for bs in bs_list:
                for flag_gpu in (False, True):
                    ver_full = f"{ver}_{_version_suffix(ver)}"
                    key = (inst_name, ver_full, str(flag_gpu), str(bs))
                    if key in done:
                        continue
                    row = _run_bnb_subprocess(
                        instance_name=inst_name,
                        version=ver,
                        warm_start=warm_start,
                        timeout_s=timeout_s,
                        use_gpu=flag_gpu,
                        track_memory=False,
                        batch_size=bs,
                        max_memory_mb=max_memory_mb,
                    )
                    row["batch_size"] = bs
                    rows.append(row)
                    _save_checkpoint(output_csv, rows)
                    done.add(key)

    return pd.DataFrame(rows)
