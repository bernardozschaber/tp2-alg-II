"""Estruturas de retorno compartilhadas e utilitário de shortcutting hamiltoniano."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TourResult:
    """Contrato unificado de saída para todos os algoritmos exatos, aproximados e heurísticas.

    Convenção de representação de rota:
      - `tour` armazena uma permutação de `{0, ..., N-1}` iniciada no vértice `0`, sem
        duplicar o vértice inicial ao final.
      - O fechamento do ciclo `(tour[-1], tour[0])` é sempre incluído no campo `cost`.
    """

    algorithm: str
    instance: str
    n: int
    tour: List[int]
    cost: float
    elapsed_s: float
    profile: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """Converte o resultado em dicionário plano pronto para exportação em CSV/DataFrame."""
        row: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "instance": self.instance,
            "n": self.n,
            "cost": self.cost,
            "elapsed_s": self.elapsed_s,
            "total_peak_memory_kb": self.profile.get("total_peak_memory_kb", 0.0),
            "process_max_rss_kb": self.profile.get("process_max_rss_kb", 0.0),
        }
        # Achata tempo (ms) e pico de memória (KiB) de cada subrotina instrumentada.
        for sub, metrics in self.profile.get("subroutines", {}).items():
            row[f"{sub}__ms"] = metrics["elapsed_ms"]
            row[f"{sub}__peak_kb"] = metrics["peak_memory_kb"]
        # Incorpora metadados escalares específicos do algoritmo (ex.: contadores de poda).
        row.update({k: v for k, v in self.extra.items() if not isinstance(v, (list, dict))})
        return row


def shortcut_walk(walk: List[int], start: Optional[int] = 0) -> List[int]:
    """Remove visitas repetidas de um passeio fechado preservando apenas a primeira ocorrência.

    Ao suprimir vértices já visitados, substitui subcaminhos `u -> w -> v` pela ligação
    direta `u -> v`. Em espaços métricos (desigualdade triangular), essa operação nunca
    aumenta o comprimento total do circuito.
    Complexidade: Θ(|walk|) em tempo e Θ(N) em espaço via tabela hash (`set`).
    """
    seen = set()
    tour: List[int] = []
    for v in walk:
        if v not in seen:
            seen.add(v)
            tour.append(v)

    # Rotaciona circularmente a sequência para iniciar no vértice `start`, quando solicitado.
    if start is not None and tour and tour[0] != start:
        idx = tour.index(start)
        tour = tour[idx:] + tour[:idx]
    return tour
