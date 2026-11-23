#!/usr/bin/env bash
# Regera upload-flautim2/tp2_codigo.zip com src/ e data/tsplib/ da arvore de trabalho
# (arquivos versionados + novos nao ignorados; sem __pycache__). Rode apos mudar o codigo.
set -euo pipefail
cd "$(dirname "$0")/.."
git ls-files -co --exclude-standard src data/tsplib | python3 -c '
import sys, zipfile
arquivos = [l.strip() for l in sys.stdin if l.strip()]
with zipfile.ZipFile("upload-flautim2/tp2_codigo.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for a in arquivos:
        z.write(a)
print(f"upload-flautim2/tp2_codigo.zip: {len(arquivos)} arquivos")
'
