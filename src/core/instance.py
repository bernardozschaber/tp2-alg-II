"""Representação e parser de instâncias TSPLIB com coordenadas 2D nativas (EUC_2D)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gpu_utils import GPUBackend, get_backend
from .prepare_dataset import prepare_all_instances
from .profiling import ProfilerSession


@dataclass
class TSPInstance:
    """Instância do Problema do Caixeiro Viajante Geométrico (2D Euclidiano)."""

    name: str
    dimension: int
    coords: np.ndarray  # shape (N, 2), dtype float64
    dist_matrix: np.ndarray  # shape (N, N), dtype float64
    comment: str = ""
    band: str = "main"
    round_to_int: bool = False
    filepath: Optional[Path] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def tour_cost(self, tour: Sequence[int]) -> float:
        """Calcula o custo total de um ciclo hamiltoniano [v_0, v_1, ..., v_{N-1}] (ou fechado com v_0 no fim)."""
        if len(tour) == 0:
            return float("inf")
        arr = np.asarray(tour, dtype=np.int64)
        if len(arr) == self.dimension + 1 and arr[0] == arr[-1]:
            arr = arr[:-1]
        if len(arr) != self.dimension:
            raise ValueError(
                f"Rota inválida: esperado {self.dimension} vértices, recebido {len(arr)}."
            )
        nxt = np.roll(arr, -1)
        return float(np.sum(self.dist_matrix[arr, nxt]))

    def validate_tour(self, tour: Sequence[int]) -> bool:
        """Verifica se `tour` é um ciclo hamiltoniano válido sobre {0, ..., N-1}."""
        arr = list(tour)
        if len(arr) == self.dimension + 1 and arr[0] == arr[-1]:
            arr = arr[:-1]
        return len(arr) == self.dimension and set(arr) == set(range(self.dimension))


def parse_tsp_coordinates(filepath: Path) -> Tuple[Dict[str, str], np.ndarray]:
    """Lê cabeçalhos e a seção NODE_COORD_SECTION (ou DISPLAY_DATA_SECTION) de um arquivo .tsp."""
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
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords_list.append([x, y])

    if not coords_list:
        raise ValueError(f"Nenhuma coordenada 2D encontrada em {filepath}")

    coords = np.asarray(coords_list, dtype=np.float64)
    return headers, coords


def load_tsplib_instance(
    filepath_or_name: str | Path,
    round_to_int: bool = False,
    use_gpu: bool = False,
    profiler: Optional[ProfilerSession] = None,
    band: Optional[str] = None,
) -> TSPInstance:
    """Carrega uma instância TSPLIB (.tsp) e constrói sua matriz de distâncias EUC_2D."""
    default_dir = Path(__file__).resolve().parents[2] / "data" / "tsplib"
    path = Path(filepath_or_name)
    if not path.exists():
        candidate = default_dir / (
            f"{filepath_or_name}" if str(filepath_or_name).endswith(".tsp") else f"{filepath_or_name}.tsp"
        )
        if not candidate.exists():
            prepare_all_instances(default_dir)
        path = candidate

    headers, coords = parse_tsp_coordinates(path)
    name = headers.get("NAME", path.stem)
    comment = headers.get("COMMENT", "")
    dim = int(headers.get("DIMENSION", len(coords)))
    if len(coords) != dim:
        dim = len(coords)

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
    """Carrega todas as 15 instâncias do conjunto de teste (ou filtradas por faixa 'main'/'stress')."""
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
