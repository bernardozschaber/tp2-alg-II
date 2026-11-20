"""Geração determinística dos arquivos `.tsp` (EUC_2D), manifesto e referências ótimas.

Responsabilidades do módulo:
  1. Armazenar tabelas imutáveis com coordenadas 2D nativas extraídas diretamente dos
     arquivos oficiais `.tsp.gz` da TSPLIB95.
  2. Gerar os 15 arquivos `.tsp` padronizados no diretório `data/tsplib/` utilizando
     distância euclidiana 2D (`EDGE_WEIGHT_TYPE : EUC_2D`):
       - 12 instâncias na faixa principal (10 <= N <= 40, correspondendo a 80% da base).
       - 3 instâncias na faixa limite/estresse (48 <= N <= 70, correspondendo a 20% da base).
  3. Gerar `data/tsplib/manifest.json` com os metadados de cada instância.
  4. Gerar `data/tsplib/opt_references.json` com as rotas oficiais `.opt.tour` da TSPLIB95
     (indexadas em 0) e seus respectivos custos avaliados sob a distância euclidiana 2D,
     servindo de referência de gap quando o Branch-and-Bound atingir timeout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

# Coordenadas 2D (x, y) nativas extraídas dos arquivos oficiais da TSPLIB95.
# Cada entrada mapeia o identificador da instância para a lista ordenada de vértices 0..N-1.
RAW_TSPLIB_COORDS: Dict[str, List[Tuple[float, float]]] = {
    "burma14": [
        (16.47, 96.10), (16.47, 94.44), (20.09, 92.54), (22.39, 93.37),
        (25.23, 97.24), (22.00, 96.05), (20.47, 97.02), (17.20, 96.29),
        (16.30, 97.38), (14.05, 98.12), (16.53, 97.38), (21.52, 95.59),
        (19.41, 97.13), (20.09, 94.55),
    ],
    "ulysses16": [
        (38.24, 20.42), (39.57, 26.15), (40.56, 25.32), (36.26, 23.12),
        (33.48, 10.54), (37.56, 12.19), (38.42, 13.11), (37.52, 20.44),
        (41.23, 9.10),  (41.17, 13.05), (36.08, -5.21), (38.47, 15.13),
        (38.15, 15.35), (37.51, 15.17), (35.49, 14.32), (39.36, 19.56),
    ],
    "ulysses22": [
        (38.24, 20.42), (39.57, 26.15), (40.56, 25.32), (36.26, 23.12),
        (33.48, 10.54), (37.56, 12.19), (38.42, 13.11), (37.52, 20.44),
        (41.23, 9.10),  (41.17, 13.05), (36.08, -5.21), (38.47, 15.13),
        (38.15, 15.35), (37.51, 15.17), (35.49, 14.32), (39.36, 19.56),
        (38.09, 24.36), (36.09, 23.00), (40.44, 13.57), (40.33, 14.15),
        (40.37, 14.23), (37.57, 22.56),
    ],
    "bays29": [
        (1150.0, 1760.0), (630.0, 1660.0),  (40.0, 2090.0),   (750.0, 1100.0),
        (750.0, 2030.0),  (1030.0, 2070.0), (1650.0, 650.0),  (1490.0, 1630.0),
        (790.0, 2260.0),  (710.0, 1310.0),  (840.0, 550.0),   (1170.0, 2300.0),
        (970.0, 1340.0),  (510.0, 700.0),   (750.0, 900.0),   (1280.0, 1200.0),
        (230.0, 590.0),   (460.0, 860.0),   (1040.0, 950.0),  (590.0, 1390.0),
        (830.0, 1770.0),  (490.0, 500.0),   (1840.0, 1240.0), (1260.0, 1500.0),
        (1280.0, 790.0),  (490.0, 2130.0),  (1460.0, 1420.0), (1260.0, 1910.0),
        (360.0, 1980.0),
    ],
    "att48": [
        (6734.0, 1453.0), (2233.0, 10.0),   (5530.0, 1424.0), (401.0, 841.0),
        (3082.0, 1644.0), (7608.0, 4458.0), (7573.0, 3716.0), (7265.0, 1268.0),
        (6898.0, 1885.0), (1112.0, 2049.0), (5468.0, 2606.0), (5989.0, 2873.0),
        (4706.0, 2674.0), (4612.0, 2035.0), (6347.0, 2683.0), (6107.0, 669.0),
        (7611.0, 5184.0), (7462.0, 3590.0), (7732.0, 4723.0), (5900.0, 3561.0),
        (4483.0, 3369.0), (6101.0, 1110.0), (5199.0, 2182.0), (1633.0, 2809.0),
        (4307.0, 2322.0), (675.0, 1006.0),  (7555.0, 4819.0), (7541.0, 3981.0),
        (3177.0, 756.0),  (7352.0, 4506.0), (7545.0, 2801.0), (3245.0, 3305.0),
        (6426.0, 3173.0), (4608.0, 1198.0), (23.0, 2216.0),   (7248.0, 3779.0),
        (7762.0, 4595.0), (7392.0, 2244.0), (3484.0, 2829.0), (6271.0, 2135.0),
        (4985.0, 140.0),  (1916.0, 1569.0), (7280.0, 4899.0), (7509.0, 3239.0),
        (10.0, 2676.0),   (6807.0, 2993.0), (5185.0, 3258.0), (3023.0, 1942.0),
    ],
    "eil51": [
        (37.0, 52.0), (49.0, 49.0), (52.0, 64.0), (20.0, 26.0), (40.0, 30.0),
        (21.0, 47.0), (17.0, 63.0), (31.0, 62.0), (52.0, 33.0), (51.0, 21.0),
        (42.0, 41.0), (31.0, 32.0), (5.0, 25.0),  (12.0, 42.0), (36.0, 16.0),
        (52.0, 41.0), (27.0, 23.0), (17.0, 33.0), (13.0, 13.0), (57.0, 58.0),
        (62.0, 42.0), (42.0, 57.0), (16.0, 57.0), (8.0, 52.0),  (7.0, 38.0),
        (27.0, 68.0), (30.0, 48.0), (43.0, 67.0), (58.0, 48.0), (58.0, 27.0),
        (37.0, 69.0), (38.0, 46.0), (46.0, 10.0), (61.0, 33.0), (62.0, 63.0),
        (63.0, 69.0), (32.0, 22.0), (45.0, 35.0), (59.0, 15.0), (5.0, 6.0),
        (10.0, 17.0), (21.0, 10.0), (5.0, 64.0),  (30.0, 15.0), (39.0, 10.0),
        (32.0, 39.0), (25.0, 32.0), (25.0, 55.0), (48.0, 28.0), (56.0, 37.0),
        (30.0, 40.0),
    ],
    "berlin52": [
        (565.0, 575.0),  (25.0, 185.0),   (345.0, 750.0),  (945.0, 685.0),
        (845.0, 655.0),  (880.0, 660.0),  (25.0, 230.0),   (525.0, 1000.0),
        (580.0, 1175.0), (650.0, 1130.0), (1605.0, 620.0), (1220.0, 580.0),
        (1465.0, 200.0), (1530.0, 5.0),   (845.0, 680.0),  (725.0, 370.0),
        (145.0, 665.0),  (415.0, 635.0),  (510.0, 875.0),  (560.0, 365.0),
        (300.0, 465.0),  (520.0, 585.0),  (480.0, 415.0),  (835.0, 625.0),
        (975.0, 580.0),  (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0),
        (660.0, 180.0),  (410.0, 250.0),  (420.0, 555.0),  (575.0, 665.0),
        (1150.0, 1160.0),(700.0, 580.0),  (685.0, 595.0),  (685.0, 610.0),
        (770.0, 610.0),  (795.0, 645.0),  (720.0, 635.0),  (760.0, 650.0),
        (475.0, 960.0),  (95.0, 260.0),   (875.0, 920.0),  (700.0, 500.0),
        (555.0, 815.0),  (830.0, 485.0),  (1170.0, 65.0),  (830.0, 610.0),
        (605.0, 625.0),  (595.0, 360.0),  (1340.0, 725.0), (1740.0, 245.0),
    ],
    "st70": [
        (64.0, 96.0), (80.0, 39.0), (69.0, 23.0), (72.0, 42.0), (48.0, 67.0),
        (58.0, 43.0), (81.0, 34.0), (79.0, 17.0), (30.0, 23.0), (42.0, 67.0),
        (7.0, 76.0),  (29.0, 51.0), (78.0, 92.0), (64.0, 8.0),  (95.0, 57.0),
        (57.0, 91.0), (40.0, 35.0), (68.0, 40.0), (92.0, 34.0), (62.0, 1.0),
        (28.0, 43.0), (76.0, 73.0), (67.0, 88.0), (93.0, 54.0), (6.0, 8.0),
        (87.0, 18.0), (30.0, 9.0),  (77.0, 13.0), (78.0, 94.0), (55.0, 3.0),
        (82.0, 88.0), (73.0, 28.0), (20.0, 55.0), (27.0, 43.0), (95.0, 86.0),
        (67.0, 99.0), (48.0, 83.0), (75.0, 81.0), (8.0, 19.0),  (20.0, 18.0),
        (54.0, 38.0), (63.0, 36.0), (44.0, 33.0), (52.0, 18.0), (12.0, 13.0),
        (25.0, 5.0),  (58.0, 85.0), (5.0, 67.0),  (90.0, 9.0),  (41.0, 76.0),
        (25.0, 76.0), (37.0, 64.0), (56.0, 63.0), (10.0, 55.0), (98.0, 7.0),
        (16.0, 74.0), (89.0, 60.0), (48.0, 82.0), (81.0, 76.0), (29.0, 60.0),
        (17.0, 22.0), (5.0, 45.0),  (79.0, 70.0), (9.0, 100.0), (17.0, 82.0),
        (74.0, 67.0), (10.0, 68.0), (48.0, 19.0), (83.0, 86.0), (84.0, 94.0),
    ],
}

# Rotas ótimas oficiais (.opt.tour da TSPLIB95 convertidas para indexação 0-based).
# Usadas como referência de custo EUC_2D quando a busca exata excede o limite de tempo.
OFFICIAL_OPT_TOURS: Dict[str, List[int]] = {
    "ulysses16": [0, 13, 12, 11, 6, 5, 14, 4, 10, 8, 9, 15, 2, 1, 3, 7],
    "ulysses22": [0, 13, 12, 11, 6, 5, 14, 4, 10, 8, 9, 18, 19, 20, 15, 2, 1, 16, 21, 3, 17, 7],
    "bays29": [
        0, 27, 5, 11, 8, 4, 25, 28, 2, 1, 19, 9, 3, 14, 17, 16, 13, 21, 10, 18,
        24, 6, 22, 26, 7, 23, 15, 12, 20,
    ],
    "att48": [
        0, 7, 37, 30, 43, 17, 6, 27, 5, 36, 18, 26, 16, 42, 29, 35, 45, 32, 19,
        46, 20, 31, 38, 47, 4, 41, 23, 9, 44, 34, 3, 25, 1, 28, 33, 40, 15, 21,
        2, 22, 13, 24, 12, 10, 11, 14, 39, 8,
    ],
    "berlin52": [
        0, 48, 31, 44, 18, 40, 7, 8, 9, 42, 32, 50, 10, 51, 13, 12, 46, 25, 26,
        27, 11, 24, 3, 5, 14, 4, 23, 47, 37, 36, 39, 38, 35, 34, 33, 43, 45, 15,
        28, 49, 19, 22, 29, 1, 6, 41, 20, 16, 2, 17, 30, 21,
    ],
    "st70": [
        0, 35, 28, 12, 69, 34, 30, 68, 37, 58, 21, 65, 62, 56, 14, 23, 18, 6, 1,
        3, 17, 41, 31, 2, 7, 25, 54, 48, 27, 13, 19, 29, 43, 67, 26, 45, 24, 44,
        38, 60, 39, 8, 16, 42, 40, 5, 52, 4, 9, 51, 59, 11, 33, 20, 32, 61, 53,
        47, 66, 10, 63, 64, 55, 50, 49, 57, 36, 46, 15, 22,
    ],
}

# Especificação da bateria de 15 instâncias:
# - Faixa "main": 12 instâncias com 10 <= N <= 40 (80% da base).
# - Faixa "stress": 3 instâncias com 48 <= N <= 70 (20% da base).
BENCHMARK_SPEC = [
    {"name": "eil51_n10", "source": "eil51", "n": 10, "band": "main"},
    {"name": "berlin52_n12", "source": "berlin52", "n": 12, "band": "main"},
    {"name": "burma14", "source": "burma14", "n": 14, "band": "main"},
    {"name": "ulysses16", "source": "ulysses16", "n": 16, "band": "main"},
    {"name": "eil51_n18", "source": "eil51", "n": 18, "band": "main"},
    {"name": "berlin52_n20", "source": "berlin52", "n": 20, "band": "main"},
    {"name": "ulysses22", "source": "ulysses22", "n": 22, "band": "main"},
    {"name": "eil51_n25", "source": "eil51", "n": 25, "band": "main"},
    {"name": "bays29", "source": "bays29", "n": 29, "band": "main"},
    {"name": "berlin52_n33", "source": "berlin52", "n": 33, "band": "main"},
    {"name": "eil51_n38", "source": "eil51", "n": 38, "band": "main"},
    {"name": "st70_n40", "source": "st70", "n": 40, "band": "main"},
    {"name": "att48", "source": "att48", "n": 48, "band": "stress"},
    {"name": "berlin52", "source": "berlin52", "n": 52, "band": "stress"},
    {"name": "st70", "source": "st70", "n": 70, "band": "stress"},
]


def _euclidean_tour_cost(coords: List[Tuple[float, float]], tour: List[int]) -> float:
    """Calcula a soma exata das distâncias euclidianas ao longo de um ciclo fechado."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        total += math.hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1])
    return total


def write_tsp_file(
    filepath: Path, name: str, coords: List[Tuple[float, float]], comment: str
) -> None:
    """Serializa uma lista de coordenadas 2D no formato textual TSPLIB (NODE_COORD_SECTION)."""
    lines = [
        f"NAME : {name}",
        f"COMMENT : {comment}",
        "TYPE : TSP",
        f"DIMENSION : {len(coords)}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    # O padrão TSPLIB numera os vértices a partir de 1.
    for idx, (x, y) in enumerate(coords, start=1):
        lines.append(f"{idx} {x:.4f} {y:.4f}")
    lines.append("EOF")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_all_instances(output_dir: Path | None = None) -> Path:
    """Materializa os arquivos `.tsp`, `manifest.json` e `opt_references.json` em disco."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "tsplib"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "description": "Instâncias TSPLIB com coordenadas 2D nativas sob métrica EUC_2D",
        "total_instances": len(BENCHMARK_SPEC),
        "main_band_count": sum(1 for item in BENCHMARK_SPEC if item["band"] == "main"),
        "stress_band_count": sum(1 for item in BENCHMARK_SPEC if item["band"] == "stress"),
        "instances": [],
    }

    for spec in BENCHMARK_SPEC:
        name = spec["name"]
        source = spec["source"]
        n = spec["n"]
        band = spec["band"]
        # Recorte determinístico dos primeiros `n` vértices quando `name` é subconjunto de `source`.
        coords = RAW_TSPLIB_COORDS[source][:n]
        tsp_path = output_dir / f"{name}.tsp"
        comment = f"TSPLIB native 2D coordinates from {source} (N={n}, band={band}, EUC_2D)"
        write_tsp_file(tsp_path, name=name, coords=coords, comment=comment)
        manifest["instances"].append(
            {
                "name": name,
                "file": f"{name}.tsp",
                "source": source,
                "dimension": n,
                "band": band,
                "edge_weight_type": "EUC_2D",
                "has_tsplib_opt_tour": name in OFFICIAL_OPT_TOURS,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Gera o arquivo de referências ótimas avaliando cada .opt.tour sob distância euclidiana 2D.
    opt_references: Dict[str, dict] = {}
    for name, tour in OFFICIAL_OPT_TOURS.items():
        coords = RAW_TSPLIB_COORDS[name]
        opt_references[name] = {
            "euc2d_cost": round(_euclidean_tour_cost(coords, tour), 6),
            "source": "tsplib_opt_tour",
            "tour": tour,
        }
    opt_ref_path = output_dir / "opt_references.json"
    opt_ref_path.write_text(
        json.dumps(opt_references, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


if __name__ == "__main__":
    path = prepare_all_instances()
    print(f"Dataset TSPLIB preparado em: {path}")
