# Trabalho Prático 2 - Análise Empírica e Otimização para o Caixeiro Viajante

**Disciplina:** DCC207 – Algoritmos 2  
**Instituição:** Universidade Federal de Minas Gerais (UFMG) — ICEx — Departamento de Ciência da Computação (DCC)  
**Professor:** Prof. Renato Vimieiro  
**Integrantes:**
* Bernardo Zschaber Morato Nogueira -- 2024066121
* Mateus Matsura Teles Costa -- 2024066059
* Pedro Rangel Ferreira de Aguiar -- 2024065907

---

## Visão Geral

Este repositório reúne a implementação auditada, a suíte de testes unitários, o orquestrador de experimentos em subprocessos, os módulos de modelagem estatística/visualização e o artigo científico no formato SBC para o **Problema do Caixeiro Viajante (TSP) Euclidiano 2D (`EUC_2D`)**.

O trabalho segue uma metodologia de **otimização gradual guiada por hipóteses**, contrastando versões iniciais literais (`v0`) contra refatorações incrementais controladas nos algoritmos aproximados (*Twice-Around-The-Tree* e *Christofides*), nas heurísticas de inicialização (*warm-start*) e na busca exata (*Branch-and-Bound* `v0` a `v3`), com avaliação empírica em três eixos:
1. **Profiling por subrotina e complexidade empírica** (ajustes $R^2$ e RMSE, memória/GC e aceleração GPU via `--use-gpu`).
2. **Impacto do *warm-start* na poda do *Branch-and-Bound*** (nós expandidos, podados e *trade-off* temporal com limite de 30 minutos).
3. **Análise de sensibilidade topológica e instâncias adversariais** (descritores geométricos e da AGM modelados via regressão Ridge e árvores de decisão).

---

## Organização dos Diretórios

```text
tp2-alg-II/
├── README.md                              # Guia geral do repositório
├── relatorio.pdf                          # Relatório final
├── requirements.txt                       # Dependências Python do projeto (e opção CuPy/CUDA)
├── data/
│   ├── tsplib/                            # Base auditada de 15 instâncias EUC_2D nativas (.tsp)
│   │   ├── manifest.json                  # Metadados oficiais das 15 instâncias
│   │   └── opt_references.json            # Gabarito ótimo exato contínuo (DFJ) para cálculo de gap
│   └── results/                           # Saídas das baterias experimentais (CSVs e JSONs analíticos)
├── docs/
│   └── prompts_history.md                 # Registro consolidado do uso de ferramentas de IA
├── report/
│   ├── Makefile                           # Automação da compilação LaTeX (pdflatex + bibtex)
│   ├── main.tex                           # Documento principal no padrão SBC
│   ├── references.bib                     # Referências bibliográficas formatadas no estilo sbc.bst
│   ├── sbc-template.sty / sbc.bst         # Estilo oficial da Sociedade Brasileira de Computação (SBC)
│   ├── sections/                          # Seções modulares do artigo em ordem conceitual
│   ├── figures/                           # Gráficos vetoriais gerados por `analyze`
│   └── tables/                            # Tabelas booktabs (.tex) geradas por `analyze`
├── src/
│   ├── cli.py                             # Interface unificada de linha de comando
│   ├── core/                              # Núcleo de dados, instrumentação e abstração GPU
│   │   ├── instance.py                    # Parser TSPLIB e matriz de distâncias EUC_2D contígua
│   │   ├── prepare_dataset.py             # Construção determinística das 15 instâncias nativas
│   │   ├── profiling.py                   # Medição por subrotina
│   │   └── gpu_utils.py                   # Camada CuPy/NumPy sob `--use-gpu` e avaliação em lote
│   ├── algorithms/                        # Implementações graduais dos algoritmos
│   │   ├── common.py                      # Estrutura `TourResult`, `shortcut_walk` e `prim_dense`
│   │   ├── warm_start.py                  # 5 estratégias de warm-start
│   │   ├── twice_around/                  # Twice-Around-The-Tree (`v0` e `v1`)
│   │   ├── christofides/                  # Christofides (`v0` e `v1`)
│   │   └── bnb/                           # Branch-and-Bound (`v0` a `v3`)
│   ├── experiments/                       # Execução isolada e orquestração das baterias
│   │   ├── worker.py                      # Subprocesso isolado com timeout, RLIMIT_AS e captura de OOM
│   │   └── runner.py                      # Orquestração das Baterias A–D com checkpointing atômico
│   └── analysis/                          # Modelagem estatística e visualização offline
│       ├── build_opt_reference.py         # Solver DFJ MILP para certificação dos ótimos EUC_2D
│       ├── features.py                    # Extração de 8 descritores geométricos e estruturais da AGM
│       ├── complexity_fit.py              # Regressão não linear por mínimos quadrados (R² e RMSE)
│       ├── sensitivity.py                 # RidgeCV (LOO), Árvore de Decisão e Random Forest
│       └── plots.py                       # Geração automatizada das figuras e tabelas
├── tests/                                 # Suíte de verificação (29 testes unitários)
│   ├── oracles.py                         # Oráculos exatos para instâncias pequenas
│   ├── test_stage1_core.py                # Testes de base de dados, parser, profiling e utilitários GPU
│   ├── test_stage2_approximations.py      # Testes de Twice-Around, Christofides, Graham Scan e warm-starts
│   ├── test_stage3_bnb.py                 # Testes de exatidão e equivalência entre B&B v0–v3 e Held-Karp
│   └── test_stage4_analysis.py            # Testes de features, regressões, sensibilidade, worker e plots
├── tools/                                 # Scripts auxiliares de empacotamento e consolidação de resultados
│   ├── empacotar_flautim.sh               # Gera `upload-flautim2/tp2_codigo.zip` com `src/` e `data/tsplib/`
│   └── juntar_resultados.py               # Mescla e deduplica CSVs de múltiplos jobs em `data/results/`
└── upload-flautim2/                       # Pacote de execução das baterias na plataforma Flautim (GPU H200)
    ├── run.py                             # Orquestrador remoto das Baterias A–D + `analyze` (retomada via tag)
    ├── plataforma.py                      # Infraestrutura de cgroup, logs, `EXP_*` e escrita atômica de JSON
    └── requirements.txt                   # Dependências fixadas para Python 3.11 + CuPy/wheels NVIDIA CUDA 12
```

---

## Configuração do Ambiente Local

### 1. Criar e ativar o ambiente virtual (Python 3.10+)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Suporte opcional a GPU NVIDIA
Nas máquinas com GPU NVIDIA e driver CUDA 12.x, instale o `CuPy` dentro do ambiente virtual para habilitar aceleração real com a flag `--use-gpu` (sem CuPy, o sistema faz *fallback* automático para `NumPy` vetorizado na CPU):
```bash
pip install cupy-cuda12x
```

### 3. Validar a instalação com a suíte de testes unitários
```bash
python -m unittest discover -s tests -p "test_*.py"
```
Todos os **29 testes** devem finalizar com `OK`.

---

## Fluxos de Execução de Comandos

Toda a operação experimental e analítica é centralizada na CLI `python -m src.cli <subcomando>`.

### Referência rápida dos subcomandos

| Subcomando | O que executa | Saída principal |
| :--- | :--- | :--- |
| `prepare-opt-refs` | Recalcula e certifica os ótimos exatos contínuos `EUC_2D` via DFJ (já pré-gerado no repo) | `data/tsplib/opt_references.json` |
| `extract-features` | Extrai os 8 descritores geométricos e estruturais da AGM das 15 instâncias | `data/results/instance_features.csv` |
| `run-approx` | **Bateria A**: Twice-Around (`v0`/`v1`), Christofides (`v0`/`v1`) e *warm-starts* (15 instâncias) | `data/results/approx_profiling.csv` |
| `run-ablation` | **Bateria B**: Ablação gradual do B&B (`v0` $\to$ `v3`, padrão: 15 instâncias) | `data/results/bnb_ablation.csv` |
| `run-warmstart` | **Bateria C**: Experimento oficial de 30 min (15 instâncias $\times$ 5 *warm-starts* no `v3`) | `data/results/bnb_warmstart.csv` |
| `run-gpu-bench` | **Bateria D**: Comparativo direto CPU vs. `--use-gpu` em `v1`, `v2` e `v3` | `data/results/gpu_comparison.csv` |
| `analyze` | Roda features + ajustes de complexidade + sensibilidade + geração de gráficos e tabelas | `data/results/*`, `report/figures/*`, `report/tables/*` |

Consulte todas as opções de qualquer subcomando com `python -m src.cli <subcomando> --help`.

### Fluxo completo de execução na máquina experimental

Todas as baterias de *Branch-and-Bound* suportam **retomada automática** (caso interrompidas, basta reexecutar o mesmo comando; use `--no-resume` apenas para sobrescrever do zero). Recomenda-se rodar todas as baterias na mesma máquina para consistência de hardware:

```bash
# 1. Bateria A — Algoritmos aproximados e heurísticas de warm-start (segundos)
python -m src.cli run-approx --repeats 5

# 2. Bateria B — Ablação gradual v0 -> v3 (captura OOM via --max-memory-mb 4096 e timeout)
python -m src.cli run-ablation --timeout 120 --max-memory-mb 4096 --workers 4

# 3. Bateria C — Experimento oficial de 30 min (15 instâncias x 5 warm-starts no v3)
python -m src.cli run-warmstart --timeout 1800 --max-memory-mb 4096 --workers 4

# 4. Bateria D — Validação direta CPU vs. GPU (na máquina com CUDA + CuPy)
python -m src.cli run-gpu-bench --timeout 300 --batch-sizes 128 1024

# 5. Modelagem estatística + geração de todas as figuras (.pdf/.png) e tabelas (.tex)
python -m src.cli analyze
```

---

## Compilação do Relatório SBC (LaTeX)

A partir da raiz do repositório (requer `pdflatex` e `bibtex`):

```bash
# Compila o artigo completo (pdflatex -> bibtex -> pdflatex -> pdflatex) gerando report/main.pdf
make -C report

# Remove arquivos temporários de compilação (.aux, .bbl, .log, .pdf, etc.)
make -C report clean
```

**Ciclo integrado após atualizar análises ou texto:**
```bash
python -m src.cli analyze && make -C report
```

O projeto também é totalmente compatível com o **Overleaf**: basta importar o conteúdo da pasta `report/` (`main.tex`, `references.bib`, `sbc-template.sty`, `sbc.bst`, `sections/`, `figures/` e `tables/`) e compilar com **pdfLaTeX**.
