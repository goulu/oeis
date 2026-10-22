# oeis
Python class for infinite sequences, implementation of some, and testing against OEIS database

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Initialiser la base de données SQLite à partir de oeis.py
python src/init_db.py

# Lancer les tests
pytest

# Démarrer le serveur web local
uvicorn src.server:app --reload
```

- Le code source Python est organisé dans le dossier `src/` (`src/sequence.py`, `src/oeis.py`, `src/server.py`, etc.).
- Les tests unitaires sont dans `tests/`.
- Les fichiers Markdown générés pour chaque séquence sont dans `sequences/`.