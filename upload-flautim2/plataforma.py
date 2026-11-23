"""Infraestrutura da Flautim nova. Igual em todo experimento: nao precisa mexer.

Como a plataforma roda o job (visto no teste3, ambiente.json):
  - os arquivos enviados ficam lado a lado em /nfs/tmp/<usuario>/<job>/input
    (tambem visivel como /workspace), que e o cwd; "import dataset" funciona;
  - /output e um link para /nfs/tmp/<usuario>/<job>/output, ou seja, o
    ../output do cwd. O painel le /output/metrics.json; a plataforma ja grava
    o stdout em /output/output.txt;
  - venv novo em /tmp/venv, "pip install -r requirements.txt" do PyPI e
    depois "python run.py";
  - no com H200, mas o cgroup limita a 2 CPUs e 8 GB de RAM (os 224 nucleos
    e 2 TB que o /proc mostra sao do no inteiro);
  - nenhum dataset montado. Ao finalizar, a pasta do job e apagada, mas
    /nfs/tmp/<usuario>/ continua: o cache de datasets fica la.

So depende da biblioteca padrao; torch e numpy sao importados sob demanda.
"""
from pathlib import Path
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import socket
import sys
import time
import traceback
import urllib.request

OUTPUT_DIR = Path("../output")


def _cache_padrao():
    """/nfs/tmp/<usuario>/datasets: fora da pasta do job, sobrevive a ele."""
    cwd = Path.cwd().resolve()
    if len(cwd.parts) >= 5 and cwd.parts[1:3] == ("nfs", "tmp"):
        return Path(*cwd.parts[:4]) / "datasets"
    return Path("/tmp/datasets")


CACHE_DIR = Path(os.getenv("CACHE_DIR") or _cache_padrao())


# ##################################################################
# Log
# ##################################################################
def log(message, details=""):
    linha = f"[{time.strftime('%H:%M:%S')}] {message}"
    if details:
        linha += f" | {details}"
    print(linha, flush=True)


def log_erro(message, excecao=None):
    if excecao is not None:
        message += f" | {type(excecao).__name__}: {excecao}"
    log(f"ERRO: {message}")
    print(traceback.format_exc(), flush=True)


@contextlib.contextmanager
def etapa(nome):
    """INICIO/OK de cada etapa: a ultima linha INICIO sem OK diz onde parou."""
    log(f"INICIO: {nome}")
    t0 = time.time()
    try:
        yield
    except Exception as ex:
        log_erro(f"FALHOU na etapa '{nome}' apos {time.time() - t0:.1f}s", ex)
        raise
    log(f"OK: {nome}", details=f"{time.time() - t0:.1f}s")


# ##################################################################
# Configuracao
# ##################################################################
def aplicar_variaveis(config, prefixo="EXP_"):
    """Sobrescreve chaves do config com variaveis EXP_<CHAVE> do formulario.

    O tipo vem do valor padrao: EXP_EPOCHS=50 vira int, EXP_LEARNING_RATE=0.05
    vira float, EXP_SALVAR_MODELO=0 vira False, EXP_SEMENTES=42,43 vira lista
    de int. Variavel EXP_ sem chave correspondente e erro (pega erro de digitacao).
    """
    config = dict(config)
    desconhecidas = []
    for nome, texto in os.environ.items():
        if not nome.startswith(prefixo):
            continue
        chave = nome[len(prefixo):].lower()
        if chave not in config:
            desconhecidas.append(nome)
            continue
        padrao = config[chave]
        if isinstance(padrao, bool):
            valor = texto.strip().lower() in ("1", "true", "sim", "yes")
        elif isinstance(padrao, int):
            valor = int(texto)
        elif isinstance(padrao, float):
            valor = float(texto)
        elif isinstance(padrao, list):
            valor = [int(v) if v.strip().lstrip("-").isdigit() else v.strip()
                     for v in texto.split(",") if v.strip()]
        elif padrao is None:
            valor = None if texto.strip().lower() in ("", "none") else _numero_ou_texto(texto)
        else:
            valor = texto
        log(f"config > {chave} = {valor!r} (de {nome}; padrao {padrao!r})")
        config[chave] = valor
    if desconhecidas:
        raise ValueError(f"variaveis {desconhecidas} nao correspondem a nenhuma chave de "
                         f"CONFIG: {sorted(config)}")
    return config


def _numero_ou_texto(texto):
    for tipo in (int, float):
        try:
            return tipo(texto)
        except ValueError:
            pass
    return texto


# ##################################################################
# Ambiente
# ##################################################################
def cpus_disponiveis():
    """CPUs do cgroup (cpu.max), nao do no: 2 no teste3, contra 224 visiveis."""
    try:
        cota, periodo = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if cota != "max":
            return max(1, math.floor(int(cota) / int(periodo)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


def _memoria_limite_gb():
    try:
        valor = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        return None if valor == "max" else round(int(valor) / 2**30, 1)
    except (OSError, ValueError):
        return None


def resumo_ambiente():
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "gpu": gpu,
        "cpus": cpus_disponiveis(),
        "memoria_limite_gb": _memoria_limite_gb(),
        "cwd": os.getcwd(),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "cache_dir": str(CACHE_DIR),
        "cache_livre_gb": round(shutil.disk_usage(CACHE_DIR).free / 2**30, 1) if CACHE_DIR.exists() else None,
    }


def semear(semente):
    import numpy as np
    import torch
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


# ##################################################################
# Saidas
# ##################################################################
def _valor_json(valor):
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


def salvar_json(caminho, objeto):
    """Escrita atomica: um job morto no meio nao deixa JSON pela metade."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    temporario.write_text(json.dumps(objeto, indent=2, default=str))
    temporario.replace(caminho)


def salvar_metricas(metricas):
    """/output/metrics.json, no formato plano que o painel le ({"acc": 0.87, ...})."""
    planas = {k: _valor_json(float(v)) for k, v in metricas.items()}
    salvar_json(OUTPUT_DIR / "metrics.json", planas)


# ##################################################################
# Download e cache de datasets
# ##################################################################
def baixar(url, destino, md5=None, rotulo=None):
    """Baixa `url` para `destino`, com progresso a cada 10% e timeout."""
    socket.setdefaulttimeout(120)  # conexao parada vira erro em vez de travar o job
    rotulo = rotulo or Path(destino).name
    parcial = Path(str(destino) + ".part")
    t0, marcos = time.time(), set()

    def progresso(blocos, tamanho_bloco, total):
        if total > 0:
            pct = min(100, blocos * tamanho_bloco * 100 // total) // 10 * 10
            if pct not in marcos:
                marcos.add(pct)
                log(f"download > {rotulo}: {pct}% de {total / 2**20:.0f} MB ({time.time() - t0:.0f} s)")

    log(f"download > {rotulo}: {url}")
    urllib.request.urlretrieve(url, parcial, reporthook=progresso)
    if md5 is not None:
        obtido = hashlib.md5(parcial.read_bytes()).hexdigest()
        if obtido != md5:
            parcial.unlink()
            raise RuntimeError(f"md5 de {rotulo} nao confere: {obtido} (esperado {md5})")
    parcial.replace(destino)
    log(f"download > {rotulo}: concluido em {time.time() - t0:.0f} s")
    return Path(destino)


def pasta_em_cache(nome, preparar):
    """CACHE_DIR/<nome>, preparando so na primeira vez.

    `preparar(pasta_temporaria)` baixa/extrai os arquivos dentro dela. A
    pasta so ganha o nome final (rename atomico) quando termina, entao um
    job morto no meio do download nao deixa um dataset incompleto, e dois
    jobs simultaneos nao se atrapalham: o segundo descarta a propria copia.
    """
    final = CACHE_DIR / nome
    if (final / ".completo").is_file():
        log(f"cache > {nome}: reaproveitado de {final}")
        return final
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as erro:
        raise RuntimeError(f"cache {CACHE_DIR} nao gravavel ({erro}); defina CACHE_DIR") from erro
    temporaria = CACHE_DIR / f".{nome}.tmp-{socket.gethostname()}-{os.getpid()}"
    shutil.rmtree(temporaria, ignore_errors=True)
    temporaria.mkdir(parents=True)
    log(f"cache > {nome}: nao esta em {final}; preparando (so na primeira vez)")
    t0 = time.time()
    try:
        preparar(temporaria)
        (temporaria / ".completo").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        if final.exists() and not (final / ".completo").is_file():
            shutil.rmtree(final, ignore_errors=True)  # sobra de uma versao antiga sem marcador
        temporaria.rename(final)
    except OSError:
        if not (final / ".completo").is_file():
            raise
        log(f"cache > {nome}: outro job terminou antes; usando a copia dele")
    finally:
        shutil.rmtree(temporaria, ignore_errors=True)
    log(f"cache > {nome}: pronto em {final} ({(time.time() - t0) / 60:.1f} min)")
    return final
