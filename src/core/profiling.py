"""Instrumentação de tempo e memória em nível de subrotina para o TP2 (DCC207).

Utiliza `time.perf_counter_ns()` para medição temporal de alta resolução
e `tracemalloc` da biblioteca padrão para rastreamento de alocações de memória
(memória líquida e pico de memória dentro de cada subrotina).
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List


@dataclass
class SubroutineMetrics:
    """Métricas agregadas de tempo e memória de uma subrotina."""

    name: str
    call_count: int = 0
    elapsed_ns: int = 0
    peak_memory_bytes: int = 0
    net_memory_bytes: int = 0

    @property
    def elapsed_ms(self) -> float:
        """Tempo acumulado em milissegundos."""
        return self.elapsed_ns / 1_000_000.0

    @property
    def elapsed_s(self) -> float:
        """Tempo acumulado em segundos."""
        return self.elapsed_ns / 1_000_000_000.0

    @property
    def peak_memory_kb(self) -> float:
        """Pico de memória alocada durante a subrotina em KiB."""
        return self.peak_memory_bytes / 1024.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["elapsed_ms"] = self.elapsed_ms
        data["elapsed_s"] = self.elapsed_s
        data["peak_memory_kb"] = self.peak_memory_kb
        return data


@dataclass
class ProfilerSession:
    """Sessão de profiling para coletar métricas de múltiplas subrotinas de uma execução."""

    track_memory: bool = True
    subroutines: Dict[str, SubroutineMetrics] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)
    total_elapsed_ns: int = 0
    total_peak_memory_bytes: int = 0

    @contextmanager
    def profile(self, subroutine_name: str) -> Iterator[SubroutineMetrics]:
        """Context manager que mede tempo (ns) e memória (bytes) de uma subrotina."""
        if subroutine_name not in self.subroutines:
            self.subroutines[subroutine_name] = SubroutineMetrics(name=subroutine_name)
            self._order.append(subroutine_name)
        record = self.subroutines[subroutine_name]

        started_here = False
        if self.track_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                started_here = True
            mem_before, _ = tracemalloc.get_traced_memory()
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
        """Retorna as métricas de uma subrotina específica."""
        return self.subroutines[subroutine_name]

    def summary_dict(self) -> Dict[str, Any]:
        """Exporta todas as métricas da sessão em formato serializável (JSON/CSV)."""
        total_sub_ns = sum(m.elapsed_ns for m in self.subroutines.values())
        effective_total_ns = max(self.total_elapsed_ns, total_sub_ns)
        effective_peak_bytes = max(
            [self.total_peak_memory_bytes]
            + [m.peak_memory_bytes for m in self.subroutines.values()]
        )
        return {
            "total_elapsed_ns": effective_total_ns,
            "total_elapsed_ms": effective_total_ns / 1_000_000.0,
            "total_elapsed_s": effective_total_ns / 1_000_000_000.0,
            "total_peak_memory_bytes": effective_peak_bytes,
            "total_peak_memory_kb": effective_peak_bytes / 1024.0,
            "subroutines": {
                name: self.subroutines[name].to_dict() for name in self._order
            },
        }
