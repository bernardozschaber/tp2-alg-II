"""Junta os CSVs de varios jobs da Flautim (um por tag) em data/results/.

Uso:
  python tools/juntar_resultados.py <saida1> <saida2> ... [--destino data/results]

Cada <saida> e a pasta /output baixada de um job (ou um .tar dela); os CSVs ficam em
<saida>/resultados/. Linhas repetidas entre jobs (mesma chave do runner) ficam com a
versao do job listado por ultimo. Depois rode `python -m src.cli analyze`.
"""
from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

CHAVES = {
    "approx_profiling.csv": ["instance", "algorithm"],
    "bnb_ablation.csv": ["instance", "version", "warm_start", "use_gpu"],
    "bnb_warmstart.csv": ["instance", "warm_start"],
    "gpu_comparison.csv": ["instance", "version", "use_gpu", "batch_size"],
}


def pasta_resultados(origem: Path, tmp: Path) -> Path:
    if origem.is_file() and tarfile.is_tarfile(origem):
        destino = tmp / origem.stem
        with tarfile.open(origem) as t:
            t.extractall(destino, filter="data")
        origem = destino
    achados = sorted(origem.rglob("resultados"))
    if not achados:
        raise FileNotFoundError(f"{origem}: pasta resultados/ nao encontrada")
    return achados[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("saidas", nargs="+", type=Path)
    ap.add_argument("--destino", type=Path, default=Path("data/results"))
    args = ap.parse_args()
    args.destino.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        pastas = [pasta_resultados(s, Path(tmp)) for s in args.saidas]
        for nome, chave in CHAVES.items():
            partes = []
            for ordem, pasta in enumerate(pastas):
                arq = pasta / nome
                if arq.is_file() and arq.stat().st_size > 2:
                    df = pd.read_csv(arq, keep_default_na=False, dtype=str)
                    df["_ordem"] = ordem
                    partes.append(df)
            if not partes:
                print(f"{nome}: nenhum job tem este arquivo")
                continue
            df = pd.concat(partes, ignore_index=True).sort_values("_ordem", kind="stable")
            df = df.drop_duplicates(subset=chave, keep="last").drop(columns="_ordem")
            df.to_csv(args.destino / nome, index=False)
            resumo = df["status"].value_counts().to_dict() if "status" in df else {}
            print(f"{nome}: {len(df)} linhas, {df['instance'].nunique()} instancias {resumo}")


if __name__ == "__main__":
    main()
