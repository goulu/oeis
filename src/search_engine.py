"""
Moteur de recherche pour le clone OEIS, implémentant les spécifications de https://oeis.org/hints.html.
Prend en charge :
- La recherche de sous-séquences d'entiers consécutives (via 3-grams et table seq)
- La recherche insensible ou sensible au signe (seq: vs signed:)
- Les wildcards : '_' (un terme quelconque) et '__' (sous-séquence arbitraire)
- Les préfixes de champs (id:, name:, author:, keyword:, seq:, signed:, subseq:)
- La recherche plein texte (FTS5) avec racinisation
- Le calcul de pertinence (+Score)
"""

import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple


class OEISSearchEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def parse_query(self, query_str: str) -> Dict[str, Any]:
        """
        Analyse une requête utilisateur selon la syntaxe d'OEIS (hints.html).
        """
        raw = query_str.strip()
        parsed = {
            'raw': raw,
            'is_direct_id': False,
            'direct_id': None,
            'numbers': [],
            'is_consecutive_seq': False,
            'has_wildcards': False,
            'tokens': [],
            'prefixes': {},
            'excluded': []
        }

        # 1. Vérification si c'est un identifiant OEIS direct (ex: A000045, A45, id:A000045)
        id_match = re.match(r'^(?:id:)?([Aa]\d{1,7})$', raw)
        if id_match:
            digits = id_match.group(1)[1:]
            parsed['is_direct_id'] = True
            parsed['direct_id'] = f"A{digits.zfill(6)}"
            return parsed

        # 2. Extraction des préfixes (ex: author:Guy, keyword:nice, seq:1,2,3, subseq:5,8)
        prefix_pattern = r'(-?)([a-zA-Z_]+):([^\s]+)'
        matches = re.findall(prefix_pattern, raw)
        for neg, key, val in matches:
            key_lower = key.lower()
            if neg == '-':
                parsed['excluded'].append((key_lower, val))
            else:
                parsed['prefixes'][key_lower] = val

        # Nettoyage de la chaîne des préfixes traités
        clean_text = re.sub(prefix_pattern, ' ', raw).strip()

        # 3. Détection des séquences de nombres (ex: "10946, 17711, 28657" ou "10946 17711 28657")
        # Traitement des wildcards '_' et '__'
        has_wildcard = '_' in clean_text
        parsed['has_wildcards'] = has_wildcard

        # Extraction des tokens séparés par virgules ou espaces
        # Si la chaîne contient des virgules OU si elle contient au moins 2 nombres
        tokens = [t.strip() for t in clean_text.replace(',', ' ').split() if t.strip()]
        
        # Test si les tokens sont essentiellement numériques (ou wildcards)
        numeric_terms = []
        is_numeric_query = True
        for t in tokens:
            if t in ('_', '__'):
                numeric_terms.append(t)
            else:
                try:
                    num = int(t)
                    numeric_terms.append(num)
                except ValueError:
                    is_numeric_query = False
                    break

        if is_numeric_query and len(numeric_terms) >= 1:
            parsed['numbers'] = numeric_terms
            parsed['is_consecutive_seq'] = True
        else:
            parsed['tokens'] = tokens

        return parsed

    def search(self, query_str: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Exécute la recherche et retourne une liste de dictionnaires ordonnée par pertinence.
        """
        parsed = self.parse_query(query_str)
        conn = self._get_conn()
        cursor = conn.cursor()
        results = []

        try:
            # 1. Requête par identifiant direct
            if parsed['is_direct_id']:
                target_id = parsed['direct_id']
                cursor.execute("""
                    SELECT id, name, offset_val, terms_preview, author
                    FROM sequences WHERE id = ?
                """, (target_id,))
                row = cursor.fetchone()
                if row:
                    results.append({
                        'id': row['id'],
                        'name': row['name'],
                        'terms': row['terms_preview'],
                        'score': 1000,
                        'match_desc': f"Exact identifier {row['id']}"
                    })
                conn.close()
                return results

            # 2. Requête par séquence de termes numériques
            if parsed['is_consecutive_seq'] and parsed['numbers']:
                num_results = self._search_numeric_sequence(cursor, parsed['numbers'], limit)
                results.extend(num_results)

            # 3. Requête avec préfixes (author:, keyword:, etc.)
            if parsed['prefixes']:
                prefix_results = self._search_prefixes(cursor, parsed['prefixes'], limit)
                results.extend(prefix_results)

            # 4. Requête texte / Fallback FTS5 si aucun résultat ou si termes textuels
            if not results or parsed['tokens']:
                fts_query = ' '.join(parsed['tokens']) if parsed['tokens'] else parsed['raw']
                fts_results = self._search_fts(cursor, fts_query, limit)
                results.extend(fts_results)

            # Dédoublonnage et tri par score décroissant
            seen = set()
            unique_results = []
            for r in results:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    unique_results.append(r)

            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return unique_results[:limit]
        finally:
            conn.close()

    def _search_numeric_sequence(self, cursor: sqlite3.Cursor, terms: List[Any], limit: int) -> List[Dict[str, Any]]:
        """
        Recherche une suite de termes consécutifs (ex: [10946, 17711, 28657]).
        Utilise l'index de 3-grams pour filtrer immédiatement les candidats,
        puis valide sur les termes complets.
        """
        # Séparation des entiers purs
        pure_ints = [t for t in terms if isinstance(t, int)]
        if not pure_ints:
            return []

        results = []
        candidate_ids = set()
        matched_offsets = {}

        # Cas A: au moins 3 entiers consécutifs sans wildcard au début
        if len(pure_ints) >= 3 and all(isinstance(terms[i], int) for i in range(3)):
            t0, t1, t2 = terms[0], terms[1], terms[2]
            h = hash((t0, t1, t2))
            cursor.execute("""
                SELECT sequence_id, offset_pos FROM ngram_index WHERE ngram_hash = ?
            """, (h,))
            rows = cursor.fetchall()
            for r in rows:
                candidate_ids.add(r['sequence_id'])
                matched_offsets[r['sequence_id']] = r['offset_pos']

        # Cas B: Recherche via la table 'seq' (triplets i, n, repeat) si aucun candidat 3-gram
        if not candidate_ids:
            # Chercher les séquences contenant le premier terme
            first_term = pure_ints[0]
            cursor.execute("""
                SELECT DISTINCT id FROM seq WHERE n = ? LIMIT 100
            """, (first_term,))
            for r in cursor.fetchall():
                sid = f"A{str(r['id']).zfill(6)}"
                candidate_ids.add(sid)

        # Vérification et scoring des candidats
        for sid in candidate_ids:
            cursor.execute("""
                SELECT id, name, offset_val, terms_preview FROM sequences WHERE id = ?
            """, (sid,))
            seq_row = cursor.fetchone()
            if not seq_row:
                continue

            # Récupération de tous les termes disponibles pour ce candidat dans la table 'seq'
            numeric_id = int(sid[1:])
            cursor.execute("""
                SELECT n, repeat FROM seq WHERE id = ? ORDER BY i
            """, (numeric_id,))
            seq_items = []
            for n_val, r_val in cursor.fetchall():
                seq_items.extend([n_val] * r_val)

            # Si la table seq est vide ou incomplète, utiliser terms_preview
            if not seq_items and seq_row['terms_preview']:
                try:
                    seq_items = [int(x.strip()) for x in seq_row['terms_preview'].split(',') if x.strip().lstrip('-').isdigit()]
                except Exception:
                    pass

            # Vérification de la présence de la sous-séquence recherchée
            match_pos, score = self._match_subsequence(seq_items, terms)
            if match_pos is not None:
                term_str = ', '.join(str(x) for x in terms)
                results.append({
                    'id': sid,
                    'name': seq_row['name'],
                    'terms': seq_row['terms_preview'],
                    'score': score,
                    'match_desc': f"Consecutive subsequence [{term_str}] found at index {match_pos}"
                })

        return results

    def _match_subsequence(self, full_seq: List[int], query_terms: List[Any]) -> Tuple[Optional[int], int]:
        """
        Vérifie si query_terms (avec d'éventuels wildcards '_' ou '__') est sous-séquence de full_seq.
        Retourne (offset_trouvé, score).
        """
        L = len(query_terms)
        N = len(full_seq)
        if L == 0 or N == 0:
            return None, 0

        # Recherche de correspondance contiguë
        for i in range(N - L + 1):
            matched = True
            for j in range(L):
                q = query_terms[j]
                if q == '_':
                    continue
                elif q != full_seq[i + j]:
                    matched = False
                    break
            if matched:
                # Score de base 100 + bonus pour proximité du début
                bonus = max(0, 20 - i)
                return i, 100 + bonus

        # Recherche non contiguë si '__' est présent ou pour subseq:
        if '__' in query_terms:
            sub_terms = [t for t in query_terms if t != '__']
            curr_pos = 0
            first_pos = None
            all_found = True
            for st in sub_terms:
                found_idx = -1
                for k in range(curr_pos, N):
                    if st == '_' or full_seq[k] == st:
                        found_idx = k
                        if first_pos is None:
                            first_pos = k
                        break
                if found_idx == -1:
                    all_found = False
                    break
                curr_pos = found_idx + 1

            if all_found and first_pos is not None:
                return first_pos, 70

        return None, 0

    def _search_prefixes(self, cursor: sqlite3.Cursor, prefixes: Dict[str, str], limit: int) -> List[Dict[str, Any]]:
        results = []
        for k, v in prefixes.items():
            if k == 'author':
                cursor.execute("""
                    SELECT id, name, terms_preview, author FROM sequences
                    WHERE author LIKE ? LIMIT ?
                """, (f"%{v}%", limit))
                for r in cursor.fetchall():
                    results.append({
                        'id': r['id'],
                        'name': r['name'],
                        'terms': r['terms_preview'],
                        'score': 85,
                        'match_desc': f"Author matching '{v}'"
                    })
            elif k in ('keyword', 'keywords'):
                cursor.execute("""
                    SELECT id, name, terms_preview, keywords FROM sequences
                    WHERE keywords LIKE ? LIMIT ?
                """, (f"%{v}%", limit))
                for r in cursor.fetchall():
                    results.append({
                        'id': r['id'],
                        'name': r['name'],
                        'terms': r['terms_preview'],
                        'score': 80,
                        'match_desc': f"Keyword matching '{v}'"
                    })
        return results

    def _search_fts(self, cursor: sqlite3.Cursor, query_text: str, limit: int) -> List[Dict[str, Any]]:
        results = []
        clean_text = ''.join(c if c.isalnum() else ' ' for c in query_text).strip()
        if not clean_text:
            return []

        try:
            cursor.execute("""
                SELECT s.id, s.name, s.terms_preview, rank
                FROM sequences_fts f
                JOIN sequences s ON f.id = s.id
                WHERE sequences_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (clean_text, limit))
            for r in cursor.fetchall():
                results.append({
                    'id': r['id'],
                    'name': r['name'],
                    'terms': r['terms_preview'],
                    'score': 50 - r['rank'],
                    'match_desc': "Full-text FTS5 match"
                })
        except sqlite3.OperationalError:
            pass
        return results
