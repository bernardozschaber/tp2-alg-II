"""Instrumentação de tempo e memória em nível de subrotina e processo.

Fornece duas camadas complementares de medição:
  1. Tempo de alta resolução por subrotina (`time.perf_counter_ns()`): custo de CPU
     praticamente nulo, adequado para todas as execuções.
  2. Memória em dois níveis:
     - `tracemalloc` (`track_memory=True`): rastreia alocações exatas do heap Python
       (pico e saldo líquido por subrotina). Impõe overhead de tempo (~5x em laços
       recursivos intensivos), devendo ser usado em passadas dedicadas de memória.
     - Max RSS do processo (`resource.getrusage(RUSAGE_SELF).ru_maxrss`): lido direto
       do kernel do sistema operacional com custo zero de tempo, permitindo coletar
       memória física real mesmo durante execuções longas com `track_memory=False`.
"""

from __future__ import annotations

import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List


def get_process_max_rss_kb() -> float:
    """Retorna o pico de memória física (Max RSS) do processo atual em KiB.

    No Linux, `ru_maxrss` já é devolvido pelo kernel em KiB; no macOS (Darwin),
    é devolvido em bytes e convertido dividindo por 1024.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / 1024.0
    return float(usage)


@dataclass
class SubroutineMetrics:
    """Contadores acumulados de tempo e memória de uma única subrotina instrumentada."""

    name: str
    call_count: int = 0
    elapsed_ns: int = 0
    peak_memory_bytes: int = 0  # Maior pico relativo (delta) observado em uma chamada
    net_memory_bytes: int = 0   # Saldo acumulado de memória que permaneceu alocada ao sair

    @property
    def elapsed_ms(self) -> float:
        """Tempo total acumulado em milissegundos."""
        return self.elapsed_ns / 1_000_000.0

    @property
    def elapsed_s(self) -> float:
        """Tempo total acumulado em segundos."""
        return self.elapsed_ns / 1_000_000_000.0

    @property
    def peak_memory_kb(self) -> float:
        """Pico de memória alocada durante a subrotina em KiB."""
        return self.peak_memory_bytes / 1024.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializa todas as métricas (brutas e convertidas) para exportação em JSON/CSV."""
        data = asdict(self)
        data["elapsed_ms"] = self.elapsed_ms
        data["elapsed_s"] = self.elapsed_s
        data["peak_memory_kb"] = self.peak_memory_kb
        return data


@dataclass
class ProfilerSession:
    """Sessão que agrega medições de tempo e memória de múltiplas subrotinas de um algoritmo."""

    track_memory: bool = True
    subroutines: Dict[str, SubroutineMetrics] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)
    total_elapsed_ns: int = 0
    total_peak_memory_bytes: int = 0
    process_max_rss_kb: float = 0.0

    @contextmanager
    def profile(self, subroutine_name: str) -> Iterator[SubroutineMetrics]:
        """Context manager que mede duração (ns) e memória (tracemalloc + RSS) de um bloco."""
        if subroutine_name not in self.subroutines:
            self.subroutines[subroutine_name] = SubroutineMetrics(name=subroutine_name)
            self._order.append(subroutine_name)
        record = self.subroutines[subroutine_name]

        started_here = False
        if self.track_memory:
            # Inicia o rastreamento apenas se ainda não estiver ativo para suportar chamadas aninhadas.
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                started_here = True
            mem_before, _ = tracemalloc.get_traced_memory()
            # Zera o marcador de pico para isolar o pico gerado exclusivamente por esta subrotina.
            tracemalloc.reset_peak()
        else:
            mem_before = 0

        t0 = time.perf_counter_ns()
        try:
            yield record
        finally:
            dt = time.perf_counter_ns() - t0
            record.call_count += 1
            record.elapsed_ns += dt

            # Atualiza o Max RSS do processo (leitura direta do SO, custo desprezível).
            rss_now = get_process_max_rss_kb()
            if rss_now > self.process_max_rss_kb:
                self.process_max_rss_kb = rss_now

            if self.track_memory and tracemalloc.is_tracing():
                mem_after, mem_peak = tracemalloc.get_traced_memory()
                delta_peak = max(0, mem_peak - mem_before)
                delta_net = max(0, mem_after - mem_before)
                if delta_peak > record.peak_memory_bytes:
                    record.peak_memory_bytes = delta_peak
                record.net_memory_bytes += delta_net
                if mem_peak > self.total_peak_memory_bytes:
                    self.total_peak_memory_bytes = mem_peak
                if started_here:
                    tracemalloc.stop()

    def get(self, subroutine_name: str) -> SubroutineMetrics:
        """Retorna o registro de métricas da subrotina especificada."""
        return self.subroutines[subroutine_name]

    def summary_dict(self) -> Dict[str, Any]:
        """Consolida tempos, picos de tracemalloc e Max RSS do processo em dicionário serializável."""
        total_sub_ns = sum(m.elapsed_ns for m in self.subroutines.values())
        effective_total_ns = max(self.total_elapsed_ns, total_sub_ns)
        effective_peak_bytes = max(
            [self.total_peak_memory_bytes]
            + [m.peak_memory_bytes for m in self.subroutines.values()]
        )
        rss_kb = max(self.process_max_rss_kb, get_process_max_rss_kb())
        return {
            "total_elapsed_ns": effective_total_ns,
            "total_elapsed_ms": effective_total_ns / 1_000_000.0,
            "total_elapsed_s": effective_total_ns / 1_000_000_000.0,
            "total_peak_memory_bytes": effective_peak_bytes,
            "total_peak_memory_kb": effective_peak_bytes / 1024.0,
            "process_max_rss_kb": rss_kb,
            "subroutines": {
                name: self.subroutines[name].to_dict() for name in self._order
            },
        }
