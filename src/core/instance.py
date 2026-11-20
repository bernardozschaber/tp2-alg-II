"""Leitura de arquivos TSPLIB (`.tsp`), cálculo da matriz EUC_2D e validação de rotas.

Estruturas e funções principais:
  - `TSPInstance`: dataclass imutável em uso que encapsula as coordenadas `(N, 2)` e a
    matriz simétrica de distâncias `(N, N)` em `float64`, além de métodos de validação
    de permutação e cálculo vetorizado do custo total de um ciclo hamiltoniano.
  - `parse_tsp_coordinates`: lê cabeçalhos chave-valor e extrai a tabela `(x, y)` das
    seções `NODE_COORD_SECTION` ou `DISPLAY_DATA_SECTION`.
  - `load_tsplib_instance` e `load_benchmark_suite`: carregam instâncias individuais ou
    toda a bateria de testes, gerando os arquivos automaticamente caso não existam.
  - `load_opt_references`: carrega o dicionário de custos de referência `.opt.tour`
    (`data/tsplib/opt_references.json`) para cálculo de gap em caso de timeout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gpu_utils import GPUBackend, get_backend
from .prepare_dataset import prepare_all_instances
from .profiling import ProfilerSession


@dataclass
class TSPInstance:
    """Representação em memória de uma instância do TSP geométrico 2D."""

    name: str
    dimension: int
    coords: np.ndarray  # Matriz (N, 2) em float64 com (x_i, y_i) de cada vértice
    dist_matrix: np.ndarray  # Matriz simétrica (N, N) em float64; diagonal igual a 0.0
    comment: str = ""
    band: str = "main"  # Faixa de tamanho: "main" (N <= 40) ou "stress" (N >= 45)
    round_to_int: bool = False  # Se True, aplica arredondamento nint(d) = floor(d + 0.5)
    filepath: Optional[Path] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def tour_cost(self, tour: Sequence[int]) -> float:
        """Calcula o custo fechado de um ciclo hamiltoniano somando todas as N arestas.

        Aceita `tour` com N vértices (ciclo implícito do último para o primeiro) ou com
        N+1 vértices quando o vértice inicial é repetido no final.
        Complexidade: Θ(N) vetorizado em NumPy.
        """
        if len(tour) == 0:
            return float("inf")
        arr = np.asarray(tour, dtype=np.int64)
        # Normaliza removendo a repetição final do vértice inicial, se presente.
        if len(arr) == self.dimension + 1 and arr[0] == arr[-1]:
            arr = arr[:-1]
        if len(arr) != self.dimension:
            raise ValueError(
                f"Rota inválida: esperado {self.dimension} vértices, recebido {len(arr)}."
            )
        # Desloca o vetor em 1 posição circularmente para parear cada vértice com seu sucessor.
        nxt = np.roll(arr, -1)
        return float(np.sum(self.dist_matrix[arr, nxt]))

    def validate_tour(self, tour: Sequence[int]) -> bool:
        """Retorna True se e somente se `tour` visitar todos os vértices {0..N-1} uma única vez."""
        arr = list(tour)
        if len(arr) == self.dimension + 1 and arr[0] == arr[-1]:
            arr = arr[:-1]
        return len(arr) == self.dimension and set(arr) == set(range(self.dimension))


def parse_tsp_coordinates(filepath: Path) -> Tuple[Dict[str, str], np.ndarray]:
    """Extrai o dicionário de cabeçalhos e a matriz `(N, 2)` de coordenadas de um arquivo `.tsp`."""
    headers: Dict[str, str] = {}
    coords_list: List[List[float]] = []
    reading_coords = False

    for raw_line in filepath.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper in ("NODE_COORD_SECTION", "DISPLAY_DATA_SECTION"):
            reading_coords = True
            continue
        if not reading_coords:
            if ":" in line:
                key, val = line.split(":", 1)
                headers[key.strip().upper()] = val.strip()
        else:
            # Cada linha da seção de coordenadas segue o formato `<id> <x> <y>`.
            parts = line.split()
            if len(parts) >= 3:
                coords_list.append([float(parts[1]), float(parts[2])])

    if not coords_list:
        raise ValueError(f"Nenhuma coordenada 2D encontrada em {filepath}")

    return headers, np.asarray(coords_list, dtype=np.float64)


def load_tsplib_instance(
    filepath_or_name: str | Path,
    round_to_int: bool = False,
    use_gpu: bool = False,
    profiler: Optional[ProfilerSession] = None,
    band: Optional[str] = None,
) -> TSPInstance:
    """Carrega uma instância `.tsp` por caminho ou nome lógico e calcula sua matriz EUC_2D."""
    default_dir = Path(__file__).resolve().parents[2] / "data" / "tsplib"
    path = Path(filepath_or_name)
    if not path.exists():
        filename = (
            str(filepath_or_name)
            if str(filepath_or_name).endswith(".tsp")
            else f"{filepath_or_name}.tsp"
        )
        candidate = default_dir / filename
        if not candidate.exists():
            prepare_all_instances(default_dir)
        path = candidate

    headers, coords = parse_tsp_coordinates(path)
    name = headers.get("NAME", path.stem)
    comment = headers.get("COMMENT", "")
    dim = int(headers.get("DIMENSION", len(coords)))
    if len(coords) != dim:
        dim = len(coords)

    # Infere automaticamente a faixa de tamanho quando não fornecida explicitamente.
    if band is None:
        if "band=stress" in comment:
            band = "stress"
        elif "band=main" in comment:
            band = "main"
        else:
            band = "stress" if dim >= 45 else "main"

    backend: GPUBackend = get_backend(use_gpu=use_gpu)
    if profiler is not None:
        with profiler.profile("dist_matrix_euc2d"):
            dist_matrix = backend.compute_euc2d_matrix(coords, round_to_int=round_to_int)
    else:
        dist_matrix = backend.compute_euc2d_matrix(coords, round_to_int=round_to_int)

    return TSPInstance(
        name=name,
        dimension=dim,
        coords=coords,
        dist_matrix=dist_matrix,
        comment=comment,
        band=band,
        round_to_int=round_to_int,
        filepath=path,
        metadata=headers,
    )


def load_benchmark_suite(
    data_dir: Optional[Path] = None,
    round_to_int: bool = False,
    use_gpu: bool = False,
    band_filter: Optional[str] = None,
) -> List[TSPInstance]:
    """Carrega todas as instâncias registradas em `manifest.json`, com filtro opcional por faixa."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "tsplib"
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        prepare_all_instances(data_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances: List[TSPInstance] = []
    for item in manifest["instances"]:
        if band_filter is not None and item["band"] != band_filter:
            continue
        inst = load_tsplib_instance(
            data_dir / item["file"],
            round_to_int=round_to_int,
            use_gpu=use_gpu,
            band=item["band"],
        )
        instances.append(inst)
    return instances


def load_opt_references(data_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Carrega os custos de referência EUC_2D calculados sobre os `.opt.tour` oficiais da TSPLIB."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "tsplib"
    opt_path = data_dir / "opt_references.json"
    if not opt_path.exists():
        prepare_all_instances(data_dir)
    return json.loads(opt_path.read_text(encoding="utf-8"))
