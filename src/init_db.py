"""
Initialisation de la base de données SQLite pour le clone OEIS.
Ce script crée le schéma défini dans create.sql, crée l'utilisateur 'goulu'
et peuple la base avec les implémentations de séquences définies dans oeis.py.
"""

import ast
import itertools
import os
import sqlite3
import sys
import time

from goulib import decorators, itertools2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'create.sql')) else SRC_DIR
DB_PATH = os.path.join(ROOT_DIR, 'oeis.db')
SQL_PATH = os.path.join(ROOT_DIR, 'create.sql')
OEIS_PY_PATH = os.path.join(SRC_DIR, 'oeis.py')



def parse_oeis_ast(filepath):
    """Analyse l'AST de oeis.py pour extraire le code des déclarations et fonctions auxiliaires."""
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    tree = ast.parse(code)

    # Récupération des fonctions auxiliaires (ex: def fA051885(n): ...)
    aux_funcs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            aux_funcs[node.name] = ast.get_source_segment(code, node)

    # Récupération des définitions de séquences AXXXXXX
    seq_definitions = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Cas 1: Assignation simple: A000045 = ...
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith('A') and len(target.id) == 7:
                    val = node.value
                    full_stmt = ast.get_source_segment(code, node)
                    val_code = ast.get_source_segment(code, val)

                    info = {
                        'definition_code': full_stmt,
                        'expr_code': val_code,
                        'iterf_code': None,
                        'itemf_code': None,
                        'containf_code': None,
                        'desc_code': None,
                        'auxiliary_code': None,
                    }

                    if isinstance(val, ast.Call) and getattr(val.func, 'id', None) == 'Sequence':
                        for i, arg in enumerate(val.args):
                            arg_code = ast.get_source_segment(code, arg)
                            if i == 0:
                                info['iterf_code'] = arg_code
                            elif i == 1:
                                info['itemf_code'] = arg_code
                            elif i == 2:
                                info['containf_code'] = arg_code
                            elif i == 3:
                                info['desc_code'] = arg_code

                        for kw in val.keywords:
                            kw_code = ast.get_source_segment(code, kw.value)
                            if kw.arg == 'iterf':
                                info['iterf_code'] = kw_code
                            elif kw.arg == 'itemf':
                                info['itemf_code'] = kw_code
                            elif kw.arg == 'containf':
                                info['containf_code'] = kw_code
                            elif kw.arg == 'desc':
                                info['desc_code'] = kw_code

                    # Recherche d'éventuelles fonctions auxiliaires mentionnées
                    referenced_aux = []
                    for fname, fsource in aux_funcs.items():
                        if fname in val_code:
                            referenced_aux.append(fsource)
                    if referenced_aux:
                        info['auxiliary_code'] = '\n\n'.join(referenced_aux)

                    seq_definitions[target.id] = info

                # Cas 2: Assignation tuple: A000225, A003462, ... = (...)
                elif isinstance(target, ast.Tuple):
                    full_stmt = ast.get_source_segment(code, node)
                    val_code = ast.get_source_segment(code, node.value)
                    for elt in target.elts:
                        if isinstance(elt, ast.Name) and elt.id.startswith('A') and len(elt.id) == 7:
                            seq_definitions[elt.id] = {
                                'definition_code': full_stmt,
                                'expr_code': val_code,
                                'iterf_code': None,
                                'itemf_code': None,
                                'containf_code': None,
                                'desc_code': None,
                                'auxiliary_code': None,
                            }

    return seq_definitions


def compute_preview_and_terms(seq, max_terms=60, timeout_sec=0.2):
    """Tente de générer les premiers termes de manière protégée avec un timeout."""
    terms = []
    try:
        # Configuration d'un timeout temporaire pour la séquence
        orig_timeout = seq.timeout
        seq.timeout = timeout_sec
        it = iter(seq)
        for i, val in enumerate(it):
            if i >= max_terms:
                break
            if isinstance(val, int) and abs(val) < 10**18:
                terms.append(val)
            else:
                break
    except Exception:
        pass
    finally:
        seq.timeout = orig_timeout

    preview_str = ', '.join(str(t) for t in terms[:30]) if terms else ''
    return preview_str, terms



def init_database():
    print(f"Connexion et initialisation du schéma : {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Exécution de create.sql
    with open(SQL_PATH, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)

    # 2. Création de l'utilisateur 'goulu'
    cursor.execute("SELECT id FROM users WHERE username = 'goulu'")
    row = cursor.fetchone()
    if row:
        goulu_id = row[0]
        print(f"Utilisateur existant 'goulu' trouvé (ID={goulu_id})")
    else:
        cursor.execute(
            """
            INSERT INTO users (username, email, role)
            VALUES (?, ?, ?)
        """,
            ('goulu', 'goulib@goulu.net', 'admin'),
        )
        goulu_id = cursor.lastrowid
        print(f"Utilisateur 'goulu' créé (ID={goulu_id})")

    # 3. Chargement de oeis.py et analyse AST
    print("Analyse du code source oeis.py...")
    import oeis

    ast_defs = parse_oeis_ast(OEIS_PY_PATH)
    total_seqs = len(oeis.oeis)
    print(f"Séquences chargées depuis oeis.oeis : {total_seqs}")
    print(f"Définitions extraites via l'AST : {len(ast_defs)}")

    count_inserted = 0

    for sid, seq_obj in sorted(oeis.oeis.items()):
        name = getattr(seq_obj, 'desc', '') or ''
        offset_val = getattr(seq_obj, 'offset', 0) or 0

        # Récupération du code issu de l'AST
        ast_info = ast_defs.get(
            sid,
            {
                'definition_code': f"{sid} = Sequence(...)  # Code non extrait",
                'iterf_code': None,
                'itemf_code': None,
                'containf_code': None,
                'auxiliary_code': None,
            },
        )

        definition_code = ast_info['definition_code']
        iterf_code = ast_info.get('iterf_code')
        itemf_code = ast_info.get('itemf_code')
        containf_code = ast_info.get('containf_code')
        auxiliary_code = ast_info.get('auxiliary_code')

        # Calcul des premiers termes
        preview_str, terms = compute_preview_and_terms(seq_obj, max_terms=60)

        # Insertion ou remplacement de la séquence
        cursor.execute(
            """
            INSERT OR REPLACE INTO sequences (id, name, offset_val, terms_preview, author)
            VALUES (?, ?, ?, ?, ?)
        """,
            (sid, name, offset_val, preview_str, 'Philippe Guglielmetti'),
        )

        # Insertion de l'implémentation de goulu
        cursor.execute(
            """
            INSERT INTO implementations (
                sequence_id, user_id, status, definition_code,
                iterf_code, itemf_code, containf_code, auxiliary_code,
                description, test_status, terms_verified_count
            )
            VALUES (?, ?, 'approved', ?, ?, ?, ?, ?, ?, 'passed', ?)
        """,
            (
                sid,
                goulu_id,
                definition_code,
                iterf_code,
                itemf_code,
                containf_code,
                auxiliary_code,
                name,
                len(terms),
            ),
        )

        impl_id = cursor.lastrowid

        # Mise à jour du pointeur current_implementation_id
        cursor.execute(
            """
            UPDATE sequences SET current_implementation_id = ? WHERE id = ?
        """,
            (impl_id, sid),
        )

        # Insertion dans l'index plein-texte FTS5
        cursor.execute(
            """
            INSERT OR REPLACE INTO sequences_fts (id, name, keywords, author)
            VALUES (?, ?, ?, ?)
        """,
            (sid, name, '', 'Philippe Guglielmetti'),
        )

        # Indexation des 3-grams numériques
        if len(terms) >= 3:
            for pos in range(len(terms) - 2):
                triplet = (terms[pos], terms[pos + 1], terms[pos + 2])
                triplet_hash = hash(triplet)
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO ngram_index (ngram_hash, sequence_id, offset_pos)
                    VALUES (?, ?, ?)
                """,
                    (triplet_hash, sid, pos),
                )

        # Compression RLE dans la table seq historique (pour compatibilité)
        if terms:
            cursor.execute("DELETE FROM seq WHERE id = ?", (int(sid[1:]),))
            i = 1
            for n_val, r_val in itertools2.compress(terms):
                if n_val <= 9223372036854775807:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO seq (id, i, n, repeat)
                        VALUES (?, ?, ?, ?)
                    """,
                        (int(sid[1:]), i, n_val, r_val),
                    )
                i += r_val

        count_inserted += 1

    conn.commit()
    print(f"\nTerminé avec succès : {count_inserted} séquences insérées.")

    # Statistiques
    cursor.execute("SELECT count(*) FROM sequences")
    nb_seqs = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM implementations")
    nb_impls = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM ngram_index")
    nb_ngrams = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM seq")
    nb_seq_rows = cursor.fetchone()[0]

    print("\n--- Statistiques de la base oeis.db ---")
    print(f"Utilisateurs           : 1 ('goulu')")
    print(f"Séquences cataloguées  : {nb_seqs}")
    print(f"Implémentations actives: {nb_impls}")
    print(f"3-Grams indexés        : {nb_ngrams}")
    print(f"Lignes 'seq' (RLE)     : {nb_seq_rows}")

    conn.close()


if __name__ == '__main__':
    init_database()
