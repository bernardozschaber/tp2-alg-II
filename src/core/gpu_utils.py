"""Abstração e utilitários de aceleração via GPU (--use-gpu) para o TP2 (DCC207).

Quando `use_gpu=True`, tenta utilizar `cupy` (CUDA) para executar operações
matriciais e avaliações de limites inferiores (bounds) em lote na GPU.
Caso a máquina local não possua GPU CUDA ou `cupy` instalado, realiza fallback
transparente para operações vetorizadas em `numpy`, mantendo a mesma interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = bool(cp.cuda.is_available())
except Exception:
    cp = None  # type: ignore
    _HAS_CUPY = False


@dataclass
class GPUBackend:
    """Gerenciador de backend numérico (CPU NumPy vs GPU CuPy) controlado por --use-gpu."""

    use_gpu_requested: bool = False
    device_active: bool = False
    backend_name: str = "numpy_cpu"
    device_name: str = "CPU"
    xp: Any = np

    def __post_init__(self) -> None:
        if self.use_gpu_requested and _HAS_CUPY and cp is not None:
            self.device_active = True
            self.backend_name = "cupy_gpu"
            self.xp = cp
            try:
                dev_id = cp.cuda.Device().id
                props = cp.cuda.runtime.getDeviceProperties(dev_id)
                raw_name = props.get("name", b"CUDA GPU")
                self.device_name = (
                    raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                )
            except Exception:
                self.device_name = "NVIDIA CUDA GPU"
        else:
            self.device_active = False
            self.backend_name = "numpy_cpu_fallback" if self.use_gpu_requested else "numpy_cpu"
            self.device_name = "CPU (Vectorized NumPy)"
            self.xp = np

    def to_device(self, array: np.ndarray) -> Any:
        """Transfere um array NumPy para a GPU se ativa, ou retorna o próprio array."""
        if self.device_active and cp is not None:
            return cp.asarray(array)
        return np.asarray(array)

    def to_cpu(self, array: Any) -> np.ndarray:
        """Garante que o array retorne para a memória da CPU como `np.ndarray`."""
        if self.device_active and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)

    def sync(self) -> None:
        """Sincroniza o stream da GPU para medições precisas de profiling."""
        if self.device_active and cp is not None:
            cp.cuda.Stream.null.synchronize()

    def compute_euc2d_matrix(
        self, coords: np.ndarray, round_to_int: bool = False
    ) -> np.ndarray:
        """Calcula a matriz de distâncias euclidianas 2D (N x N) usando o backend ativo."""
        pts = self.to_device(np.asarray(coords, dtype=np.float64))
        diff = pts[:, None, :] - pts[None, :, :]
        dist = self.xp.sqrt(self.xp.sum(diff * diff, axis=-1))
        if round_to_int:
            dist = self.xp.floor(dist + 0.5)
        self.sync()
        return self.to_cpu(dist)

    def precompute_min_edges_device(self, dist_matrix: np.ndarray) -> Tuple[Any, Any, Any, Any]:
        """Pré-computa na GPU/vetorizado as duas menores arestas incidentes a cada vértice.

        Retorna (dist_dev, min1_val, min1_idx, min2_val) no dispositivo ativo.
        """
        n = dist_matrix.shape[0]
        mat = np.array(dist_matrix, dtype=np.float64, copy=True)
        np.fill_diagonal(mat, np.inf)
        mat_dev = self.to_device(mat)

        # Ordena cada linha para obter as 2 menores arestas incidentes em cada vértice
        order = self.xp.argsort(mat_dev, axis=1)
        rows = self.xp.arange(n)
        min1_idx = order[:, 0]
        min2_idx = order[:, 1]
        min1_val = mat_dev[rows, min1_idx]
        min2_val = mat_dev[rows, min2_idx]
        self.sync()
        return mat_dev, min1_val, min1_idx, min2_val

    def batch_evaluate_children_bounds(
        self,
        dist_dev: Any,
        min1_val: Any,
        min1_idx: Any,
        min2_val: Any,
        first_vertices: np.ndarray,
        last_vertices: np.ndarray,
        candidates: np.ndarray,
        parent_costs: np.ndarray,
        unvisited_sums: np.ndarray,
        is_integer_metric: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Avalia em lote na GPU (ou NumPy vetorizado) os custos e bounds de M filhos candidatos.

        Para cada transição candidata i (estendendo caminho v_0 -> v_1 -> ... -> u com novo vértice k):
        - novo custo: parent_costs[i] + dist[u, k]
        - soma das 2 arestas por vértice segundo Aula 12 (pp. 13-14):
          S = 2 * new_cost + min_other(v_0, v_1) + min_other(k, u) + sum_{w not in path U {k}} (min1[w] + min2[w])
        - bound = ceil(S / 2) (para métrica inteira) ou S / 2.0 (para métrica contínua float).
        """
        xp = self.xp
        v1_arr = self.to_device(np.asarray(first_vertices, dtype=np.int64))
        u_arr = self.to_device(np.asarray(last_vertices, dtype=np.int64))
        k_arr = self.to_device(np.asarray(candidates, dtype=np.int64))
        p_cost = self.to_device(np.asarray(parent_costs, dtype=np.float64))
        unvis_sum = self.to_device(np.asarray(unvisited_sums, dtype=np.float64))

        step_cost = dist_dev[u_arr, k_arr]
        new_costs = p_cost + step_cost

        # Contribuição residual da raiz (vértice 0), cuja primeira aresta fixada é (0, v1)
        # Se o caminho tinha apenas [0], então v1 é o próprio candidato k
        eff_v1 = xp.where(v1_arr >= 0, v1_arr, k_arr)
        root_min_other = xp.where(min1_idx[0] == eff_v1, min2_val[0], min1_val[0])

        # Contribuição residual da nova extremidade k, cuja aresta fixada é (u, k)
        k_min_other = xp.where(min1_idx[k_arr] == u_arr, min2_val[k_arr], min1_val[k_arr])

        # Soma de (min1 + min2) dos vértices ainda não visitados após incluir k
        k_pair_sum = min1_val[k_arr] + min2_val[k_arr]
        rem_unvisited_sum = unvis_sum - k_pair_sum

        total_double_sum = 2.0 * new_costs + root_min_other + k_min_other + rem_unvisited_sum
        if is_integer_metric:
            bounds = xp.ceil(total_double_sum / 2.0)
        else:
            bounds = total_double_sum / 2.0

        self.sync()
        return self.to_cpu(new_costs), self.to_cpu(bounds)


def get_backend(use_gpu: bool = False) -> GPUBackend:
    """Fábrica de backend para a flag --use-gpu."""
    return GPUBackend(use_gpu_requested=use_gpu)
