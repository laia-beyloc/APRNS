#!/usr/bin/env bash
# Prepara una VM Linux (CPU) i llança part de l'estudi. Ús, des de lab7/:
#   bash cloud/setup_linux.sh --workers 8 --seeds 5 6 7
# Tots els arguments es passen a "python -m experiments.launch".
set -e
cd "$(dirname "$0")/.."

# Dependències del sistema per compilar Box2D (Debian/Ubuntu).
command -v swig >/dev/null || sudo apt-get install -y swig python3-venv build-essential

[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
# Mateixes versions que en local; torch de CPU (la roda CUDA és molt més gran i no s'aprofita).
pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest -q tests   # comprovació ràpida abans de llançar hores de càlcul

# nohup: l'execució continua si es tanca la sessió SSH. Segueix-la amb: tail -f launch.out
nohup python -m experiments.launch "$@" > launch.out 2>&1 &
echo "Llançat (PID $!). Log: $(pwd)/launch.out ; logs per execució a results/<nom>/seed_<n>.log"
