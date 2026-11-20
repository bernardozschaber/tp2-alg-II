"""Infraestrutura central do projeto: base TSPLIB EUC_2D, profiling e backend GPU."""

from .gpu_utils import GPUBackend, get_backend
from .instance import (
    TSPInstance,
    load_benchmark_suite,
    load_opt_references,
    load_tsplib_instance,
)
from .profiling import ProfilerSession, SubroutineMetrics, get_process_max_rss_kb

__all__ = [
    "TSPInstance",
    "load_tsplib_instance",
    "load_benchmark_suite",
    "load_opt_references",
    "ProfilerSession",
    "SubroutineMetrics",
    "get_process_max_rss_kb",
    "GPUBackend",
    "get_backend",
]
