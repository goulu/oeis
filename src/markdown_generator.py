"""
Générateur de fichiers Markdown pour les séquences OEIS.
Extrait les informations officielles d'une séquence depuis son fichier .seq (oeisdata sur GitHub)
selon les spécifications de https://oeis.org/eishelp1.html et les combine avec l'implémentation
locale SQLite si disponible.
"""

import os
import re
import sqlite3
import urllib.request
import urllib.error
import logging

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'oeis.db')) or os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'sequences')) else SRC_DIR
DEFAULT_DB_PATH = os.path.join(BASE_DIR, 'oeis.db')
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'sequences')
OEIS_RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/oeis/oeisdata/refs/heads/main/seq/{prefix}/{sid}.seq"


def normalize_seq_id(seq_id: str) -> str:
    """Normalise un identifiant OEIS en format standard Axxxxxx (ex: 3 -> A000003, a45 -> A000045)."""
    s = str(seq_id).strip().upper()
    if s.startswith('A'):
        num = s[1:]
    else:
        num = s
    if num.isdigit():
        return f"A{num.zfill(6)}"
    return s


def localize_sequence_links(text: str) -> str:
    """
    Remplace tous les liens vers d'autres séquences par des liens locaux :
    par exemple [A001622](https://oeis.org/A001622) ou (https://oeis.org/A001622) -> A001622.
    Les liens vers des sous-pages ou documents comme https://oeis.org/A000045/b000045.txt
    restent inchangés.
    """
    # 1. Liens Markdown vers une séquence seule : [texte](https://oeis.org/Axxxxxx) ou [texte](/Axxxxxx) -> [texte](Axxxxxx)
    text = re.sub(
        r'\[([^\]]*)\]\((?:https?://(?:www\.)?oeis\.org)?/+(A\d{6})/?\)',
        r'[\1](\2)',
        text,
        flags=re.IGNORECASE
    )

    # 2. Balises HTML résiduelles vers une séquence seule
    def _replace_html_a(m):
        href = m.group(1).strip()
        label = m.group(2).strip()
        sm = re.match(r'^(?:https?://(?:www\.)?oeis\.org)?/+(A\d{6})/?$', href, flags=re.IGNORECASE)
        if sm:
            return f"[{label}]({sm.group(1).upper()})"
        return m.group(0)

    text = re.sub(
        r'<a\s+href=[\"\']?([^\"\' >]+)[\"\']?\s*>(.*?)</a>',
        _replace_html_a,
        text,
        flags=re.IGNORECASE
    )

    return text


def format_oeis_link(line: str) -> str:
    """
    Convertit les balises HTML <a href="...">texte</a> issues de la ligne %H
    en liens Markdown [texte](url).
    Les liens vers d'autres séquences deviennent locaux (ex: A001622).
    Les liens vers des sous-pages ou documents (ex: https://oeis.org/A000045/b000045.txt) restent inchangés.
    """
    def _replace_tag(match):
        href = match.group(1).strip()
        text = match.group(2).strip()

        # Si le lien pointe directement vers une séquence seule (ex: /A001622 ou https://oeis.org/A001622)
        seq_match = re.match(r'^(?:https?://(?:www\.)?oeis\.org)?/+(A\d{6})/?$', href, flags=re.IGNORECASE)
        if seq_match:
            return f"[{text}]({seq_match.group(1).upper()})"

        # Liens vers des documents, sous-pages ou ancres relatives
        if href.startswith('/'):
            href = f"https://oeis.org{href}"
        elif not (href.startswith('http://') or href.startswith('https://') or href.startswith('mailto:')):
            href = f"https://oeis.org/{href}"
        return f"[{text}]({href})"

    # Support des balises avec ou sans guillemets
    formatted = re.sub(r'<a\s+href=[\"\']?([^\"\' >]+)[\"\']?\s*>(.*?)</a>', _replace_tag, line, flags=re.IGNORECASE)
    return formatted


def format_crossref(line: str) -> str:
    """Transforme les identifiants OEIS (Axxxxxx) en liens Markdown locaux [Axxxxxx](Axxxxxx)."""
    return re.sub(r'(?<!\[)\b(A\d{6})\b(?!\]|\()', r'[\1](\1)', line)


def parse_oeis_seq(content: str, sid: str = "") -> dict:
    """
    Parse les lignes d'un fichier .seq selon les abréviations définies dans https://oeis.org/eishelp1.html :
    %I : Ligne d'identification (A-number, M-number, N-number, version, date)
    %S, %T, %U : 1re, 2e et 3e lignes de termes de la suite
    %N : Nom / titre officiel de la séquence (requis)
    %C : Commentaires
    %D : Références détaillées (bibliographie)
    %H : Liens hypertextes
    %F : Formules
    %Y : Renvois et suites associées (cross-references)
    %A : Auteur / autorité
    %O : Décalage (offset a, b)
    %p : Programme Maple
    %t : Programme Mathematica
    %o : Programmes dans d'autres langages (PARI, Magma, Python, etc.)
    %E : Extensions et signalements d'erreurs
    %e : Exemples illustrant les premiers termes
    %K : Mots-clés (keywords)
    """
    fields = {}
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith('%'):
            continue
        parts = line[1:].split(' ', 2)
        if len(parts) < 3:
            continue
        tag, line_sid, val = parts[0], parts[1], parts[2].strip()
        if not sid:
            sid = line_sid
        fields.setdefault(tag, []).append(val)

    # 1. Identification (%I)
    ident_lines = fields.get('I', [])
    identification = ident_lines[0] if ident_lines else ""

    # 2. Termes (%S, %T, %U)
    terms_parts = []
    for tag in ('S', 'T', 'U'):
        if tag in fields:
            terms_parts.extend(fields[tag])
    raw_terms_str = ','.join(terms_parts)
    terms = [t.strip() for t in raw_terms_str.split(',') if t.strip()]

    # 3. Nom (%N)
    name_lines = fields.get('N', [])
    name = name_lines[0] if name_lines else ""

    # 4. Commentaires (%C)
    comments = fields.get('C', [])

    # 5. Références (%D)
    references = fields.get('D', [])

    # 6. Liens (%H)
    links = fields.get('H', [])

    # 7. Formules (%F)
    formulas = fields.get('F', [])

    # 8. Exemples (%e)
    examples = fields.get('e', [])

    # 9. Programmes (%p, %t, %o)
    programs = {
        'maple': fields.get('p', []),
        'mathematica': fields.get('t', []),
        'other': fields.get('o', []),
    }

    # 10. Renvois (%Y)
    crossrefs = fields.get('Y', [])

    # 11. Extensions / Erreurs (%E)
    extensions = fields.get('E', [])

    # 12. Auteur (%A)
    author_lines = fields.get('A', [])
    author = author_lines[0] if author_lines else ""

    # 13. Offset (%O)
    offset_lines = fields.get('O', [])
    offset_str = offset_lines[0] if offset_lines else "0, 1"
    offset_val = 0
    try:
        first_part = offset_str.split(',')[0].strip()
        offset_val = int(first_part)
    except Exception:
        offset_val = 0

    # 14. Mots-clés (%K)
    keyword_lines = fields.get('K', [])
    keywords = keyword_lines[0] if keyword_lines else ""

    return {
        'id': sid,
        'identification': identification,
        'name': name,
        'terms': terms,
        'terms_preview': ', '.join(terms[:20]) if terms else "",
        'offset_val': offset_val,
        'offset_str': offset_str,
        'keywords': keywords,
        'author': author,
        'comments': comments,
        'references': references,
        'links': links,
        'formulas': formulas,
        'examples': examples,
        'programs': programs,
        'crossrefs': crossrefs,
        'extensions': extensions,
    }


def fetch_oeis_seq_from_github(seq_id: str, timeout: int = 10) -> dict:
    """
    Télécharge et parse le fichier .seq d'une séquence depuis GitHub (oeisdata) :
    https://raw.githubusercontent.com/oeis/oeisdata/refs/heads/main/seq/{prefix}/{sid}.seq
    selon la spécification https://oeis.org/eishelp1.html.
    """
    sid = normalize_seq_id(seq_id)
    prefix = sid[:4]
    url = OEIS_RAW_URL_TEMPLATE.format(prefix=prefix, sid=sid)
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (compatible; OEIS-MarkdownGenerator/1.0)'}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode('utf-8', errors='replace')
            return parse_oeis_seq(content, sid)
    except Exception as e:
        logging.warning("Impossible de récupérer %s depuis %s : %s", sid, url, e)
        return None


def sync_sequence_to_db(seq_data: dict, conn) -> None:
    """
    Enregistre ou met à jour les informations de base de la séquence dans SQLite (tables sequences et sequences_fts).
    """
    if not conn or not seq_data.get('id'):
        return
    try:
        cursor = conn.cursor()
        sid = seq_data['id']
        name = seq_data.get('name', '')
        offset_val = seq_data.get('offset_val', 0)
        terms_preview = seq_data.get('terms_preview', '')
        keywords = seq_data.get('keywords', '')
        author = seq_data.get('author', '')

        cursor.execute("SELECT id FROM sequences WHERE id = ?", (sid,))
        exists = cursor.fetchone() is not None

        if exists:
            cursor.execute("""
                UPDATE sequences 
                SET name = COALESCE(NULLIF(?, ''), name),
                    offset_val = ?,
                    terms_preview = COALESCE(NULLIF(?, ''), terms_preview),
                    keywords = COALESCE(NULLIF(?, ''), keywords),
                    author = COALESCE(NULLIF(?, ''), author)
                WHERE id = ?
            """, (name, offset_val, terms_preview, keywords, author, sid))
        else:
            cursor.execute("""
                INSERT INTO sequences (id, name, offset_val, terms_preview, keywords, author)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sid, name, offset_val, terms_preview, keywords, author))
            try:
                cursor.execute("""
                    INSERT INTO sequences_fts (id, name, keywords, author)
                    VALUES (?, ?, ?, ?)
                """, (sid, name, keywords, author))
            except Exception:
                pass
        conn.commit()
    except Exception as e:
        logging.warning("Erreur lors de la synchronisation de %s en base : %s", seq_data.get('id'), e)


def get_sequence_data(seq_id, conn=None):
    """
    Récupère les données d'une séquence :
    1. Télécharge et parse le fichier .seq officiel depuis GitHub selon https://oeis.org/eishelp1.html
    2. Si disponible, synchronise avec la table SQLite et charge l'implémentation Python (Goulib Sequence)
    3. Si GitHub est inaccessible, bascule sur les informations locales en base.
    """
    sid = normalize_seq_id(seq_id)

    # 1. Extraction officielle depuis GitHub oeisdata
    oeis_data = fetch_oeis_seq_from_github(sid)

    # 2. Récupération des informations locales dans SQLite
    db_data = {}
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    s.id, s.name, s.offset_val, s.terms_preview, s.keywords, s.author,
                    i.definition_code, i.iterf_code, i.itemf_code, i.containf_code, i.auxiliary_code,
                    i.description, i.status, i.terms_verified_count, u.username
                FROM sequences s
                LEFT JOIN implementations i ON s.current_implementation_id = i.id
                LEFT JOIN users u ON i.user_id = u.id
                WHERE s.id = ?
            """, (sid,))
            row = cursor.fetchone()
            if row:
                cols = [
                    'id', 'name', 'offset_val', 'terms_preview', 'keywords', 'author',
                    'definition_code', 'iterf_code', 'itemf_code', 'containf_code', 'auxiliary_code',
                    'description', 'status', 'terms_verified_count', 'submitter'
                ]
                db_data = dict(zip(cols, row))
        except Exception as e:
            logging.debug("Erreur lecture SQLite pour %s : %s", sid, e)

    # Si ni GitHub ni la base n'ont de données
    if not oeis_data and not db_data:
        return None

    # Synchronisation de la base si données distantes disponibles
    if oeis_data and conn is not None:
        sync_sequence_to_db(oeis_data, conn)

    # Fusion des données
    if oeis_data:
        combined = dict(oeis_data)
        for key in ('definition_code', 'iterf_code', 'itemf_code', 'containf_code', 
                    'auxiliary_code', 'description', 'status', 'terms_verified_count', 'submitter'):
            if key in db_data:
                combined[key] = db_data[key]
        return combined
    else:
        return db_data


def format_sequence_markdown(data):
    """
    Formate les données d'une séquence en un document Markdown structuré inspiré d'OEIS
    utilisant les données extraites selon la spécification https://oeis.org/eishelp1.html.
    """
    sid = data['id']
    name = data.get('name') or 'Séquence sans titre'
    offset_str = data.get('offset_str') or f"{data.get('offset_val', 0)}, 1"
    keywords = data.get('keywords', '') or 'core,nice'
    author = data.get('author') or 'N. J. A. Sloane'
    submitter = data.get('submitter') or 'goulu'
    identification = data.get('identification', '')

    md = []
    # En-tête et Titre
    md.append(f"# {sid}")
    md.append(f"## {name}\n")

    # Barre de liens d'actions OEIS
    md.append("> `(list)` &nbsp; `(graph)` &nbsp; `(refs)` &nbsp; `(listen)` &nbsp; `(history)` &nbsp; `(text)` &nbsp; `(edit)`\n")

    # Section Termes (DATA - %S, %T, %U)
    terms_list = data.get('terms', [])
    md.append("### DATA")
    md.append("```text")
    if terms_list:
        chunks = []
        for i in range(0, len(terms_list), 20):
            chunks.append(', '.join(terms_list[i:i+20]))
        md.append(',\n'.join(chunks))
    elif data.get('terms_preview'):
        md.append(data['terms_preview'])
    else:
        md.append("Termes non calculés")
    md.append("```\n")

    # Section Métadonnées (METADATA - %O, %K, %A, %I)
    md.append("### METADATA")
    md.append(f"- **OFFSET** : `{offset_str}`")
    md.append(f"- **KEYWORDS** : `{keywords}`")
    md.append(f"- **AUTHOR** : {author}")
    if identification:
        md.append(f"- **IDENTIFICATION** : `{identification}`")
    if data.get('submitter'):
        md.append(f"- **CURRENT IMPLEMENTER** : @{submitter}")
    md.append("")

    # Section Commentaires (COMMENTS - %C)
    comments = data.get('comments', [])
    if comments:
        md.append("### COMMENTS")
        in_block = False
        for c in comments:
            c_clean = c.strip()
            if '(Start)' in c_clean:
                md.append(f"- {c_clean}")
                in_block = True
            elif in_block:
                md.append(f"  - {c_clean}")
                if '(End)' in c_clean:
                    in_block = False
            else:
                md.append(f"- {c_clean}")
        md.append("")

    # Section Exemples (EXAMPLES - %e)
    examples = data.get('examples', [])
    if examples:
        md.append("### EXAMPLES")
        for e in examples:
            md.append(f"- {e.strip()}")
        md.append("")

    # Section Formules (FORMULA - %F)
    formulas = data.get('formulas', [])
    if formulas:
        md.append("### FORMULA")
        for f in formulas:
            md.append(f"- {f.strip()}")
        md.append("")

    # Section Références (REFERENCES - %D)
    references = data.get('references', [])
    if references:
        md.append("### REFERENCES")
        for d in references:
            md.append(f"- {d.strip()}")
        md.append("")

    # Section Liens (LINKS - %H)
    links = data.get('links', [])
    if links:
        md.append("### LINKS")
        for h in links:
            md.append(f"- {format_oeis_link(h.strip())}")
        md.append("")

    # Section Programmes (%p, %t, %o)
    programs = data.get('programs', {})
    maple_progs = programs.get('maple', [])
    math_progs = programs.get('mathematica', [])
    other_progs = programs.get('other', [])

    if maple_progs or math_progs or other_progs:
        md.append("### PROGRAMS")
        if maple_progs:
            md.append("#### Maple")
            md.append("```maple")
            md.append('\n'.join(maple_progs))
            md.append("```\n")
        if math_progs:
            md.append("#### Mathematica")
            md.append("```mathematica")
            md.append('\n'.join(math_progs))
            md.append("```\n")
        if other_progs:
            md.append("#### Other Programs")
            for prog in other_progs:
                prog_clean = prog.strip()
                match_lang = re.match(r'^\(([^)]+)\)\s*(.*)', prog_clean, re.DOTALL)
                if match_lang:
                    lang, code = match_lang.group(1), match_lang.group(2)
                    md.append(f"- **({lang})** :")
                    md.append(f"```{lang.lower()}")
                    md.append(code)
                    md.append("```")
                else:
                    md.append("```text")
                    md.append(prog_clean)
                    md.append("```")
            md.append("")

    # Section Implémentation Python Goulib / Sequence
    md.append("### PYTHON IMPLEMENTATION")
    if data.get('definition_code'):
        md.append("Implémentation exécutable de la suite basée sur `goulib.sequence.Sequence` :\n")
        if data.get('auxiliary_code'):
            md.append("#### Fonctions auxiliaires :")
            md.append("```python")
            md.append(data['auxiliary_code'].strip())
            md.append("```\n")

        md.append("#### Définition principale :")
        md.append("```python")
        md.append(data['definition_code'].strip())
        md.append("```\n")

        md.append("#### Détail des méthodes fournies :")
        md.append("| Fonction | Expression | Description |")
        md.append("| :--- | :--- | :--- |")
        iterf_str = f"`{data.get('iterf_code')}`" if data.get('iterf_code') else "*(aucun / count)*"
        itemf_str = f"`{data.get('itemf_code')}`" if data.get('itemf_code') else "*(évaluation séquentielle)*"
        containf_str = f"`{data.get('containf_code')}`" if data.get('containf_code') else "*(par recherche monotone)*"
        md.append(f"| **`iterf`** | {iterf_str} | Générateur itérant sur la suite infinie |")
        md.append(f"| **`itemf(n)`** | {itemf_str} | Calcul direct du $n$-ième terme |")
        md.append(f"| **`containf(x)`** | {containf_str} | Test d'appartenance booléen ($x \\in S$) |")
    else:
        md.append("*Pas encore d'implémentation Python enregistrée dans la base locale.*")
    md.append("")

    # Section Renvois et Cross-references (CROSSREFERENCES - %Y)
    crossrefs = data.get('crossrefs', [])
    md.append("### CROSSREFERENCES")
    if crossrefs:
        for y in crossrefs:
            md.append(f"- {format_crossref(y.strip())}")
    md.append(f"- Voir également l'entrée officielle sur https://oeis.org/{sid}\n")

    # Section Extensions et Erreurs (%E)
    extensions = data.get('extensions', [])
    if extensions:
        md.append("### EXTENSIONS AND ERRORS")
        for ext in extensions:
            md.append(f"- {ext.strip()}")
        md.append("")

    content = "\n".join(md)
    return localize_sequence_links(content)


def generate_sequence_markdown(seq_id, conn=None, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Génère et écrit le fichier Markdown pour une séquence donnée.
    Extrait les données de GitHub (ou de SQLite) selon https://oeis.org/eishelp1.html.
    """
    os.makedirs(output_dir, exist_ok=True)
    sid = normalize_seq_id(seq_id)

    close_conn = False
    if conn is None and os.path.exists(DEFAULT_DB_PATH):
        try:
            conn = sqlite3.connect(DEFAULT_DB_PATH)
            close_conn = True
        except Exception:
            conn = None

    try:
        data = get_sequence_data(sid, conn)
        if not data:
            raise ValueError(f"Séquence {sid} introuvable sur GitHub oeisdata ou en base locale.")

        md_content = format_sequence_markdown(data)
        file_path = os.path.join(output_dir, f"{sid}.md")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return file_path, md_content
    finally:
        if close_conn and conn:
            conn.close()


if __name__ == "__main__":
    import sys

    target_id = sys.argv[1] if len(sys.argv) > 1 else "A000003"
    fpath, content = generate_sequence_markdown(target_id)
    print(f"Fichier Markdown généré avec succès : {fpath}")
    print(f"Taille du fichier généré : {len(content)} octets")
