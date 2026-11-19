"""Módulos centrais: carregamento TSPLIB (EUC_2D), profiling de subrotinas e suporte a GPU."""

from .instance import TSPInstance, load_tsplib_instance, load_benchmark_suite
from .profiling import ProfilerSession, SubroutineMetrics
from .gpu_utils import GPUBackend, get_backend

__all__ = [
    "TSPInstance",
    "load_tsplib_instance",
    "load_benchmark_suite",
    "ProfilerSession",
    "SubroutineMetrics",
    "GPUBackend",
    "get_backend",
]
