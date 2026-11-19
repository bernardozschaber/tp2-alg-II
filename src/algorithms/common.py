"""Estruturas comuns aos algoritmos (contrato de saída padronizado)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TourResult:
    """Resultado padronizado de qualquer algoritmo/heurística do TSP.

    `tour` é uma permutação de {0, ..., N-1} começando no vértice 0, SEM repetir
    o vértice inicial no fim (o retorno ao início é implícito no custo).
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
        """Linha achatada para CSV/DataFrame (sem a rota completa)."""
        row: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "instance": self.instance,
            "n": self.n,
            "cost": self.cost,
            "elapsed_s": self.elapsed_s,
        }
        for sub, metrics in self.profile.get("subroutines", {}).items():
            row[f"{sub}__ms"] = metrics["elapsed_ms"]
            row[f"{sub}__peak_kb"] = metrics["peak_memory_kb"]
        row.update({k: v for k, v in self.extra.items() if not isinstance(v, (list, dict))})
        return row


def shortcut_walk(walk: List[int], start: Optional[int] = 0) -> List[int]:
    """Elimina vértices repetidos de um passeio (u-w-v vira u-v), mantendo a 1ª ocorrência.

    Válido pela desigualdade triangular (Aula 13): nunca aumenta o custo.
    Complexidade Θ(len(walk)).
    """
    seen = set()
    tour: List[int] = []
    for v in walk:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    if start is not None and tour and tour[0] != start:
        i = tour.index(start)
        tour = tour[i:] + tour[:i]
    return tour
