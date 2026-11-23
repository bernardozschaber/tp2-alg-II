# Experimento: tp2-alg2-baterias
# TSP Geometrico (EUC_2D): roda de uma vez as 4 baterias experimentais do TP2
# e a analise (features, regressoes de complexidade, sensibilidade, figuras).
#
# Sobem para a plataforma os 4 arquivos desta pasta:
#   run.py            orquestracao das baterias (este arquivo)
#   plataforma.py     log, caminhos, metrics.json (copia identica da do iris)
#   tp2_codigo.zip    o projeto: src/ e data/tsplib/ (gerado por tools/empacotar_flautim.sh)
#   requirements.txt
#
# O zip existe porque a Flautim coloca os arquivos enviados lado a lado na
# pasta de entrada; o projeto depende da estrutura src/... e data/tsplib/...
# (caminhos resolvidos a partir da raiz). Se a pasta src/ ja estiver ao lado
# deste arquivo, ela e usada no lugar do zip.
#
# Baterias (python -m src.cli ..., em subprocessos):
#   A  run-approx     Twice-around, Christofides e heuristicas         (segundos)
#   B  run-ablation   B&B v0 -> v3 x {cold_start, christofides}, 120 s (~2-3 h)
#   D  run-gpu-bench  CPU vs --use-gpu em v1/v2/v3, 300 s              (~6-9 h)
#   C  run-warmstart  B&B v3 x 5 warm-starts, 1800 s                   (~12-20 h)
# A ordem padrao (A, B, D, C) deixa a mais longa por ultimo.
#
# Retomada: os CSVs ficam em CACHE_DIR/<tag>/results (/nfs/tmp/<usuario>/...,
# que sobrevive ao job) e o runner pula os casos ja gravados. Se o job cair
# ou estourar o tempo, basta submeter de novo com a mesma tag. Para comecar do
# zero: EXP_RETOMAR=0 (apaga a pasta da tag) ou outra EXP_TAG.
#
# Variaveis EXP_<CHAVE> no formulario sobrescrevem o CONFIG, ex.:
#   EXP_BATERIAS=B,C   EXP_TIMEOUT_C=900   EXP_WORKERS=1
#   EXP_INSTANCIAS=burma14,ulysses16   EXP_TAG=teste1   EXP_RETOMAR=0
#   EXP_VERSOES=v1   (so B e D: roda apenas as versoes listadas do B&B)
#
# Saidas em ../output (= /output, que o painel le):
#   metrics.json             contagens por bateria (atualizado durante a execucao)
#   resumo.json              config, ambiente, situacao de cada bateria
#   resultados/*.csv|json    os CSVs das baterias e da analise (espelhados a cada intervalo)
#   report/figures, tables   figuras e tabelas geradas pelo "analyze"
#   output.txt               o log (gravado pela propria plataforma)
# --------------------------------------------------------------
import csv
import glob
import os
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import plataforma
from plataforma import CACHE_DIR, OUTPUT_DIR, etapa, log, log_erro, salvar_json, salvar_metricas

NOME_EXPERIMENTO = "tp2-alg2-baterias"
PASTA_ENTRADA = Path(__file__).resolve().parent
ZIP_CODIGO = PASTA_ENTRADA / "tp2_codigo.zip"
N_INSTANCIAS_TOTAL = 15

CONFIG = {
    "baterias": ["A", "B", "D", "C"],
    "instancias": [],              # vazio = as 15
    "versoes": [],                 # B e D: vazio = todas (B: v0..v3; D: v1..v3), ex. ["v1"]
    "workers": 0,                  # 0 = CPUs do cgroup (2 no no padrao)
    "max_memory_mb": 0,            # 0 = 4096 por caso; workers automaticos cabem na RAM do cgroup
    "repeticoes_a": 5,
    "timeout_b": 120.0,
    "timeout_c": 1800.0,
    "timeout_d": 300.0,
    "gpu_bc": False,               # --use-gpu nas baterias B e C (o oficial e CPU)
    "analisar": True,
    "retomar": True,
    "tag": "tp2-alg2-v2",        # nova tag: o teste anterior gravou falsos OOM em tp2-alg2
    "intervalo_progresso_s": 300,
}

# bateria -> (subcomando, CSV de saida)
BATERIAS = {
    "A": ("run-approx", "approx_profiling.csv"),
    "B": ("run-ablation", "bnb_ablation.csv"),
    "C": ("run-warmstart", "bnb_warmstart.csv"),
    "D": ("run-gpu-bench", "gpu_comparison.csv"),
}


# ##################################################################
# Preparacao
# ##################################################################
def preparar_codigo():
    """Devolve a raiz do projeto (com src/ e data/tsplib/)."""
    if (PASTA_ENTRADA / "src" / "cli.py").is_file():
        log(f"codigo > usando src/ enviado em {PASTA_ENTRADA}")
        return PASTA_ENTRADA
    if not ZIP_CODIGO.is_file():
        conteudo = sorted(p.name for p in PASTA_ENTRADA.iterdir())
        raise FileNotFoundError(f"{ZIP_CODIGO.name} nao foi enviado; a pasta contem {conteudo}")
    raiz = PASTA_ENTRADA / "tp2"
    shutil.rmtree(raiz, ignore_errors=True)
    with zipfile.ZipFile(ZIP_CODIGO) as z:
        z.extractall(raiz)
    n_tsp = len(list((raiz / "data" / "tsplib").glob("*.tsp")))
    log(f"codigo > {ZIP_CODIGO.name} extraido em {raiz} ({n_tsp} instancias .tsp)")
    return raiz


def preparar_resultados(raiz, config):
    """Liga <raiz>/data/results a uma pasta persistente (retomada entre jobs)."""
    persistente = CACHE_DIR / config["tag"] / "results"
    try:
        if not config["retomar"] and persistente.exists():
            log(f"resultados > retomar=0: apagando {persistente}")
            shutil.rmtree(persistente)
        persistente.mkdir(parents=True, exist_ok=True)
        (persistente / ".teste").write_text("ok")
        (persistente / ".teste").unlink()
    except OSError as erro:
        log(f"resultados > {persistente} nao gravavel ({erro}); usando ../output/resultados (sem retomada)")
        persistente = (OUTPUT_DIR / "resultados").resolve()
        persistente.mkdir(parents=True, exist_ok=True)
    existentes = sorted(p.name for p in persistente.glob("*.csv"))
    log(f"resultados > pasta de trabalho {persistente}; ja existentes: {existentes or 'nenhum'}")

    destino = raiz / "data" / "results"
    if destino.is_symlink() or destino.is_file():
        destino.unlink()
    elif destino.is_dir():
        shutil.rmtree(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.symlink_to(persistente, target_is_directory=True)
    return persistente


def ambiente_subprocessos():
    """Variaveis para os subprocessos: 1 thread BLAS por worker e CUDA dos wheels nvidia-*."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    libs = sorted(glob.glob(str(nvidia / "*" / "lib")))
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    runtime = nvidia / "cuda_runtime"
    if runtime.is_dir():
        env.setdefault("CUDA_PATH", str(runtime))
    return env, libs


def resumo_ambiente(env, libs):
    amb = {
        "python": sys.version.split()[0],
        "cpus": plataforma.cpus_disponiveis(),
        "memoria_limite_gb": plataforma._memoria_limite_gb(),
        "cwd": os.getcwd(),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "cache_dir": str(CACHE_DIR),
        "nvidia_libs": [Path(p).parent.name for p in libs],
    }
    try:
        amb["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as erro:
        amb["nvidia_smi"] = f"indisponivel ({type(erro).__name__})"
    return amb


def testar_gpu(env, raiz):
    """Roda o backend do projeto em um subprocesso (o mesmo ambiente dos workers)."""
    codigo = (
        "import numpy as np, cupy as cp\n"
        "from src.core.gpu_utils import GPUBackend\n"
        "b = GPUBackend(use_gpu_requested=True)\n"
        "x = cp.arange(1024, dtype=cp.float64)\n"
        "s = float(cp.sqrt(x * x + 1.0).sum())\n"
        "print(b.backend_name, '|', b.device_name, '|', cp.__version__, '|', round(s, 3))\n"
    )
    proc = subprocess.run([sys.executable, "-c", codigo], cwd=str(raiz), env=env,
                          capture_output=True, text=True, timeout=600)
    if proc.returncode == 0 and "cupy_gpu" in proc.stdout:
        log(f"gpu > OK: {proc.stdout.strip()}")
        return True
    log(f"gpu > FALHOU (codigo {proc.returncode}): {proc.stdout.strip()} {proc.stderr.strip()[-2000:]}")
    return False


# ##################################################################
# Progresso
# ##################################################################
VERSOES_PADRAO = {"B": ["v0", "v1", "v2", "v3"], "D": ["v1", "v2", "v3"]}


def versoes(bateria, config):
    return [v for v in (config.get("versoes") or VERSOES_PADRAO[bateria]) if v in VERSOES_PADRAO[bateria]]


def total_esperado(bateria, n_inst, config=None):
    config = config or {}
    if bateria == "A":
        return 7 * n_inst
    if bateria == "B":
        return len(versoes("B", config)) * 2 * n_inst
    if bateria == "C":
        return 5 * n_inst
    # D: a v1 roda com 2 tamanhos de lote; cada caso em CPU e GPU
    return sum(4 if v == "v1" else 2 for v in versoes("D", config)) * n_inst


def ler_status(caminho):
    if not caminho.is_file():
        return 0, Counter()
    try:
        with open(caminho, newline="") as f:
            linhas = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return 0, Counter()
    return len(linhas), Counter(l.get("status", "?") for l in linhas)


class Monitor:
    """Thread que loga o progresso, atualiza metrics.json e espelha os CSVs no output."""

    def __init__(self, pasta_resultados, n_inst, intervalo, config=None):
        self.pasta = pasta_resultados
        self.n_inst = n_inst
        self.config = config or {}
        self.intervalo = max(30, int(intervalo))
        self.situacao = {}          # bateria -> "pendente" | "rodando" | "ok" | "falhou" | "pulada"
        self.t0 = time.time()
        self._parar = threading.Event()
        self._lock = threading.Lock()

    def metricas(self):
        m = {"horas_decorridas": (time.time() - self.t0) / 3600}
        for b, (_, nome) in BATERIAS.items():
            if b not in self.situacao:
                continue
            n, st = ler_status(self.pasta / nome)
            m[f"{b}_casos"] = n
            m[f"{b}_esperados"] = total_esperado(b, self.n_inst, self.config)
            if b != "A":
                m[f"{b}_optimal"] = st.get("OPTIMAL", 0)
                m[f"{b}_timeout"] = st.get("NA", 0)
                m[f"{b}_oom"] = st.get("OOM", 0)
            m[f"{b}_concluida"] = 1.0 if self.situacao[b] == "ok" else 0.0
        return m

    def atualizar(self, logar=True):
        with self._lock:
            m = self.metricas()
            salvar_metricas(m)
            espelhar(self.pasta)
            if logar:
                partes = []
                for b in self.situacao:
                    if f"{b}_casos" in m:
                        extra = (f" (OPT {m[f'{b}_optimal']}, NA {m[f'{b}_timeout']}, OOM {m[f'{b}_oom']})"
                                 if b != "A" else "")
                        partes.append(f"{b} {m[f'{b}_casos']}/{m[f'{b}_esperados']}{extra} [{self.situacao[b]}]")
                log(f"progresso > {m['horas_decorridas']:.2f} h | " + " | ".join(partes))
            return m

    def _laco(self):
        while not self._parar.wait(self.intervalo):
            try:
                self.atualizar()
            except Exception as erro:
                log(f"progresso > falha ao atualizar ({type(erro).__name__}: {erro})")

    def iniciar(self):
        threading.Thread(target=self._laco, daemon=True).start()

    def parar(self):
        self._parar.set()


def espelhar(pasta_resultados):
    destino = (OUTPUT_DIR / "resultados").resolve()
    if Path(pasta_resultados).resolve() == destino:
        return
    destino.mkdir(parents=True, exist_ok=True)
    for arquivo in Path(pasta_resultados).iterdir():
        if arquivo.is_file() and arquivo.suffix in (".csv", ".json"):
            shutil.copy2(arquivo, destino / arquivo.name)


# ##################################################################
# Execucao
# ##################################################################
def rodar_cli(args, raiz, env, rotulo):
    cmd = [sys.executable, "-m", "src.cli"] + args
    log(f"{rotulo} > comando: {' '.join(cmd[1:])}")
    proc = subprocess.Popen(cmd, cwd=str(raiz), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for linha in proc.stdout:
        linha = linha.rstrip()
        if linha:
            print(f"    [{rotulo}] {linha}", flush=True)
    return proc.wait()


def argumentos(bateria, config, workers, mem_mb, mem_mb_d):
    inst = ["--instances"] + config["instancias"] if config["instancias"] else []
    resume = [] if config["retomar"] else ["--no-resume"]
    gpu = ["--use-gpu"] if config["gpu_bc"] else []
    vers = ["--versions"] + versoes(bateria, config) if bateria in VERSOES_PADRAO and config.get("versoes") else []
    if bateria == "A":
        return ["run-approx", "--repeats", str(config["repeticoes_a"])] + inst
    if bateria == "B":
        return (["run-ablation", "--timeout", str(config["timeout_b"]), "--workers", str(workers),
                 "--max-memory-mb", str(mem_mb)] + vers + gpu + resume + inst)
    if bateria == "C":
        return (["run-warmstart", "--timeout", str(config["timeout_c"]), "--workers", str(workers),
                 "--max-memory-mb", str(mem_mb)] + gpu + resume + inst)
    if bateria == "D":
        return ["run-gpu-bench", "--timeout", str(config["timeout_d"]), "--max-memory-mb", str(mem_mb_d)] + vers + resume + inst
    raise ValueError(bateria)


if __name__ == "__main__":
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"run > experimento {NOME_EXPERIMENTO}")

    try:
        config = plataforma.aplicar_variaveis(CONFIG)
        config["baterias"] = [str(b).strip().upper() for b in config["baterias"]]
        invalidas = [b for b in config["baterias"] if b not in BATERIAS]
        if invalidas:
            raise ValueError(f"baterias {invalidas} invalidas; opcoes: {sorted(BATERIAS)}")
        config["instancias"] = [str(i) for i in config["instancias"]]
        config["versoes"] = [str(v).strip().lower() for v in config["versoes"]]
        if any(v not in ("v0", "v1", "v2", "v3") for v in config["versoes"]):
            raise ValueError(f"versoes {config['versoes']} invalidas; opcoes: v0 v1 v2 v3")
    except Exception as erro:
        log_erro("falha na configuracao", erro)
        raise SystemExit(1)

    env, libs = ambiente_subprocessos()
    ambiente = resumo_ambiente(env, libs)
    # O teto de 4 GB por caso e parte do experimento (ponto de OOM de v0/v1). Em modo automatico
    # os workers sao limitados pela RAM do cgroup (reserva de 2 GB para o orquestrador), em vez de
    # o teto encolher: 16 GB -> 3 workers, 28 GB -> 6, 8 GB -> 1.
    limite_mb = int((ambiente["memoria_limite_gb"] or 8.0) * 1024)
    mem_mb = config["max_memory_mb"] or 4096
    workers = config["workers"] or max(1, min(ambiente["cpus"], (limite_mb - 2048) // mem_mb))
    if workers * mem_mb + 2048 > limite_mb:
        mem_mb = max(1024, (limite_mb - 2048) // workers)
    mem_mb_d = min(config["max_memory_mb"] or 4096, limite_mb - 2048)   # bateria D roda 1 caso por vez
    if mem_mb < 4096 or mem_mb_d < 4096:
        log(f"run > AVISO: max_memory_mb={mem_mb} (D: {mem_mb_d}) < 4096; "
            f"o ponto de OOM de v0/v1 nao fica comparavel ao padrao de 4 GB")
    if workers < ambiente["cpus"]:
        log(f"run > workers={workers} (de {ambiente['cpus']} CPUs): a RAM de {limite_mb / 1024:.0f} GB "
            f"comporta {workers} casos de {mem_mb} MB em paralelo")
    log(f"run > ambiente: {ambiente}")
    log(f"run > configuracao: {config} | workers={workers} max_memory_mb={mem_mb} (D: {mem_mb_d})")
    resumo = {"experimento": NOME_EXPERIMENTO, "config": config, "ambiente": ambiente,
              "workers": workers, "max_memory_mb": mem_mb, "max_memory_mb_d": mem_mb_d}
    salvar_json(OUTPUT_DIR / "resumo.json", resumo)

    try:
        with etapa("codigo e pasta de resultados"):
            raiz = preparar_codigo()
            pasta_res = preparar_resultados(raiz, config)
    except Exception:
        raise SystemExit(1)

    gpu_ok = None
    if "D" in config["baterias"] or config["gpu_bc"]:
        try:
            with etapa("teste da GPU (CuPy)"):
                gpu_ok = testar_gpu(env, raiz)
        except Exception:
            gpu_ok = False
    resumo["gpu_ok"] = gpu_ok

    n_inst = len(config["instancias"]) or N_INSTANCIAS_TOTAL
    monitor = Monitor(pasta_res, n_inst, config["intervalo_progresso_s"], config)
    for b in config["baterias"]:
        monitor.situacao[b] = "pendente"
    monitor.iniciar()

    falhas = []
    for numero, b in enumerate(config["baterias"], 1):
        nome = f"bateria {b} ({numero}/{len(config['baterias'])})"
        if b == "D" and not gpu_ok:
            log(f"run > {nome} PULADA: CuPy/GPU indisponivel (sem GPU os numeros seriam CPU duas vezes)")
            monitor.situacao[b] = "pulada"
            falhas.append(b)
            continue
        if b == "A" and config["retomar"]:
            n, _ = ler_status(pasta_res / BATERIAS["A"][1])
            if n >= total_esperado("A", n_inst):
                log(f"run > {nome}: {BATERIAS['A'][1]} ja completo ({n} linhas), pulando")
                monitor.situacao[b] = "ok"
                continue
        monitor.situacao[b] = "rodando"
        t_b = time.time()
        try:
            with etapa(nome):
                codigo = rodar_cli(argumentos(b, config, workers, mem_mb, mem_mb_d), raiz, env, b)
                if codigo != 0:
                    raise RuntimeError(f"src.cli terminou com codigo {codigo}")
            monitor.situacao[b] = "ok"
        except Exception:
            monitor.situacao[b] = "falhou"
            falhas.append(b)
        resumo["baterias"] = dict(monitor.situacao)
        resumo["metricas"] = monitor.atualizar()
        salvar_json(OUTPUT_DIR / "resumo.json", resumo)
        log(f"run > {nome}: {monitor.situacao[b]} em {(time.time() - t_b) / 3600:.2f} h")

    if config["analisar"]:
        try:
            with etapa("analise (features, complexidade, sensibilidade, figuras)"):
                codigo = rodar_cli(["analyze"], raiz, env, "analyze")
                if codigo != 0:
                    raise RuntimeError(f"src.cli analyze terminou com codigo {codigo}")
                for sub in ("figures", "tables"):
                    origem = raiz / "report" / sub
                    if origem.is_dir():
                        shutil.copytree(origem, OUTPUT_DIR / "report" / sub, dirs_exist_ok=True)
        except Exception:
            falhas.append("analyze")

    monitor.parar()
    resumo["baterias"] = dict(monitor.situacao)
    resumo["metricas"] = monitor.atualizar()
    resumo["falhas"] = falhas
    salvar_json(OUTPUT_DIR / "resumo.json", resumo)
    log(f"run > execucao finalizada em {(time.time() - t0) / 3600:.2f} h; falhas: {falhas or 'nenhuma'}")
    if falhas:
        raise SystemExit(1)
