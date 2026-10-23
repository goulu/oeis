#!/usr/bin/env bash
# Script de lancement du serveur web OEIS (FastAPI / Uvicorn)

set -e

# Se positionner dans le répertoire racine du projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activer l'environnement virtuel si disponible
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Détecter la commande uvicorn
if command -v uvicorn >/dev/null 2>&1; then
    UVICORN_CMD="uvicorn"
elif [ -x "$SCRIPT_DIR/.venv/bin/uvicorn" ]; then
    UVICORN_CMD="$SCRIPT_DIR/.venv/bin/uvicorn"
elif python3 -m uvicorn --version >/dev/null 2>&1; then
    UVICORN_CMD="python3 -m uvicorn"
else
    echo "Erreur : uvicorn est introuvable. Veuillez installer les dépendances avec 'pip install -r requirements.txt'." >&2
    exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "=========================================================="
echo " Démarrage du serveur OEIS sur http://${HOST}:${PORT}/"
echo " Arrêt : Ctrl+C"
echo "=========================================================="

exec $UVICORN_CMD src.server:app --host "$HOST" --port "$PORT" --reload "$@"
