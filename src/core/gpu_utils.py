"""Abstração de execução vetorizada em CPU (NumPy) ou GPU CUDA (CuPy) via `--use-gpu`.

Operação empírica do módulo:
  - Na importação, tenta carregar `cupy` e consultar `cp.cuda.is_available()`.
  - Quando `use_gpu=True` e há dispositivo CUDA operacional, `GPUBackend.xp` aponta para
    `cupy` e todos os tensores intermediários são alocados e operados em VRAM.
  - Quando `use_gpu=False` ou não há GPU/CuPy no ambiente atual, `GPUBackend.xp` aponta
    para `numpy`, executando exatamente as mesmas operações vetorizadas em CPU.
  - Oferece três primitivas numéricas compartilhadas pelos algoritmos:
      1. `compute_euc2d_matrix`: matriz completa de distâncias euclidianas `(N, N)`.
      2. `precompute_min_edges_device`: extração vetorizada das duas menores arestas
         incidentes a cada vértice (`min1_val`, `min1_idx`, `min2_val`).
      3. `batch_evaluate_children_bounds`: cálculo paralelo de novos custos acumulados e
         limites inferiores (bounds) para um lote de `M` transições candidatas de uma vez.
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
    """Encapsula a seleção de dispositivo (NumPy/CPU vs CuPy/GPU) e transferências de memória."""

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
        """Move um array da RAM (host) para a VRAM da GPU, ou retorna visão NumPy na CPU."""
        if self.device_active and cp is not None:
            return cp.asarray(array)
        return np.asarray(array)

    def to_cpu(self, array: Any) -> np.ndarray:
        """Garante que o resultado esteja na memória principal (host) como `np.ndarray`."""
        if self.device_active and cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)

    def sync(self) -> None:
        """Bloqueia até a conclusão dos kernels CUDA pendentes para cronometragem fiel."""
        if self.device_active and cp is not None:
            cp.cuda.Stream.null.synchronize()

    def compute_euc2d_matrix(
        self, coords: np.ndarray, round_to_int: bool = False
    ) -> np.ndarray:
        """Computa a matriz simétrica `(N, N)` de distâncias euclidianas por broadcasting 3D.

        Expande `coords` de `(N, 2)` para `(N, 1, 2)` e `(1, N, 2)`, calcula a diferença
        par a par `diff` de shape `(N, N, 2)` e aplica norma L2 no último eixo.
        Complexidade: Θ(N²) em tempo e espaço.
        """
        pts = self.to_device(np.asarray(coords, dtype=np.float64))
        diff = pts[:, None, :] - pts[None, :, :]
        dist = self.xp.sqrt(self.xp.sum(diff * diff, axis=-1))
        if round_to_int:
            # Padrão de arredondamento inteiro: nint(d) = floor(d + 0.5).
            dist = self.xp.floor(dist + 0.5)
        self.sync()
        return self.to_cpu(dist)

    def precompute_min_edges_device(self, dist_matrix: np.ndarray) -> Tuple[Any, Any, Any, Any]:
        """Pré-computa no dispositivo ativo as duas menores arestas incidentes de cada vértice.

        Substitui a diagonal principal por +inf para impedir auto-arestas `(v, v)`, ordena
        os índices de cada linha via `argsort` e extrai as colunas 0 e 1:
          - `dist_dev` : matriz `(N, N)` no dispositivo com diagonal +inf.
          - `min1_val` : vetor `(N,)` com o menor peso incidente em cada vértice `v`.
          - `min1_idx` : vetor `(N,)` com o vizinho que realiza `min1_val[v]`.
          - `min2_val` : vetor `(N,)` com o segundo menor peso incidente em `v`.
        Complexidade: O(N² log N) no pré-processamento único da instância.
        """
        n = dist_matrix.shape[0]
        mat = np.array(dist_matrix, dtype=np.float64, copy=True)
        np.fill_diagonal(mat, np.inf)
        mat_dev = self.to_device(mat)

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
        """Avalia em lote vetorizado os custos e limites inferiores de `M` transições candidatas.

        Cada índice `i in 0..M-1` dos vetores de entrada representa estender um caminho
        parcial `0 -> v1 -> ... -> u` anexando o vértice não visitado `k`:
          - `first_vertices[i]`: `v1` (ou `-1` se o caminho pai continha apenas a raiz `[0]`).
          - `last_vertices[i]` : extremidade atual `u`.
          - `candidates[i]`    : próximo vértice candidato `k`.
          - `parent_costs[i]`  : soma das arestas já percorridas no caminho pai.
          - `unvisited_sums[i]`: soma de `(min1[w] + min2[w])` sobre todos os vértices `w`
                                 ainda não visitados no estado pai (incluindo `k`).

        Cálculo executado em paralelo para todo o lote:
          1. `new_costs = parent_costs + dist[u, k]`.
          2. Cada aresta já fixada no caminho contribui duas vezes na soma de graus 2
             (`2 * new_costs`), cobrindo integralmente todos os vértices internos e uma
             aresta de cada extremidade (`0` e `k`).
          3. A segunda aresta livre da raiz `0` é `min2[0]` se a primeira aresta `(0, v1)` já
             usar seu vizinho mais próximo `min1_idx[0]`, ou `min1[0]` caso contrário.
          4. Analogamente, a segunda aresta livre da nova extremidade `k` é `min2[k]` se
             `min1_idx[k] == u`, ou `min1[k]` caso contrário.
          5. Retira-se `min1[k] + min2[k]` de `unvisited_sums`, restando exatamente a soma
             das duas menores arestas dos vértices ainda fora do caminho.
          6. Divide-se a soma dobrada por 2 (aplicando `ceil(x - 1e-9)` em métricas inteiras
             para evitar saltos artificiais por erro de ponto flutuante).

        Complexidade: Θ(M) operações vetoriais (paralelizadas na GPU quando ativa).
        """
        xp = self.xp
        v1_arr = self.to_device(np.asarray(first_vertices, dtype=np.int64))
        u_arr = self.to_device(np.asarray(last_vertices, dtype=np.int64))
        k_arr = self.to_device(np.asarray(candidates, dtype=np.int64))
        p_cost = self.to_device(np.asarray(parent_costs, dtype=np.float64))
        unvis_sum = self.to_device(np.asarray(unvisited_sums, dtype=np.float64))

        # 1. Custo acumulado após adicionar a aresta (u, k).
        step_cost = dist_dev[u_arr, k_arr]
        new_costs = p_cost + step_cost

        # 2. Menor aresta restante na raiz 0 (se o pai era [0], o primeiro salto v1 é o próprio k).
        eff_v1 = xp.where(v1_arr >= 0, v1_arr, k_arr)
        root_min_other = xp.where(min1_idx[0] == eff_v1, min2_val[0], min1_val[0])

        # 3. Menor aresta restante na nova extremidade k distinta da aresta de chegada (u, k).
        k_min_other = xp.where(min1_idx[k_arr] == u_arr, min2_val[k_arr], min1_val[k_arr])

        # 4. Remove a contribuição de vértice livre do candidato k, que acaba de entrar no caminho.
        k_pair_sum = min1_val[k_arr] + min2_val[k_arr]
        rem_unvisited_sum = unvis_sum - k_pair_sum

        # 5. Limite inferior final: metade da soma dobrada de graus 2 sobre todos os N vértices.
        total_double_sum = 2.0 * new_costs + root_min_other + k_min_other + rem_unvisited_sum
        if is_integer_metric:
            bounds = xp.ceil(total_double_sum / 2.0 - 1e-9)
        else:
            bounds = total_double_sum / 2.0

        self.sync()
        return self.to_cpu(new_costs), self.to_cpu(bounds)


def get_backend(use_gpu: bool = False) -> GPUBackend:
    """Instancia um `GPUBackend` configurado para CPU ou GPU conforme `use_gpu`."""
    return GPUBackend(use_gpu_requested=use_gpu)
