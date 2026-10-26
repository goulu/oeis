"""
Générateur de fichiers Markdown pour les séquences OEIS.
Extrait les informations officielles d'une séquence depuis son fichier .seq (oeisdata sur GitHub)
selon les spécifications de https://oeis.org/eishelp1.html et les combine avec l'implémentation
locale SQLite si disponible.
"""

import itertools
import logging
import os
import re
import sqlite3
import urllib.error
import urllib.request

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'oeis.db')) or os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'seq')) else SRC_DIR
DEFAULT_DB_PATH = os.path.join(BASE_DIR, 'oeis.db')
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'seq')
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
    Remplace tous les noms de séquences Axxxxxx et liens vers des séquences par des liens locaux,
    et convertit tous les noms d'utilisateurs encadrés par des _ (ex: _Robert G. Wilson v_)
    en liens vers leur page wiki OEIS (https://oeis.org/wiki/User:Robert_G._Wilson_v).
    Les liens vers des sous-pages ou documents comme https://oeis.org/A000045/b000045.txt
    restent inchangés. Les blocs de code et le titre principal H1 sont préservés.
    """
    token_pattern = re.compile(
        r"(```[^\n]*\n.*?```)|"             # 1. Blocs de code (fenced code blocks)
        r"(`[^`\n]+`)|"                     # 2. Code inline
        r"(^#\s+A\d{6}\b)|"                 # 3. Titre principal H1
        r"(\[[^\]]*\]\([^)]*\))|"           # 4. Liens Markdown existants
        r"(https?://[^\s)\]>]+)|"           # 5. URLs autonomes
        r"(<a\s+[^>]*>.*?</a>)|"            # 6. Balises anchor HTML
        r"(<[^>]+>)|"                       # 7. Autres balises HTML / autolinks
        r"(\bA\d{6}\b)|"                    # 8. Noms de séquences OEIS Axxxxxx
        r"(?<![a-zA-Z0-9])_([A-Z][a-zA-Z0-9\.,\x27\u00C0-\u017F -]{1,50}[a-zA-Z0-9\.\u00C0-\u017F])_(?![a-zA-Z0-9])",  # 9. Utilisateurs _Nom Prénom_
        re.DOTALL | re.IGNORECASE | re.MULTILINE
    )

    def repl(m):
        code_block, inline_code, h1_title, md_link, url, a_tag, other_tag, seq_name, username = m.groups()
        if code_block:
            return code_block
        elif inline_code:
            return inline_code
        elif h1_title:
            return h1_title
        elif md_link:
            # Si le lien pointe directement vers une séquence seule (ex: /A001622 ou https://oeis.org/A001622)
            sm = re.match(r"^\[([^\]]*)\]\((?:https?://(?:www\.)?oeis\.org)?/+(A\d{6})/?\)$", md_link, re.I)
            if sm:
                label = sm.group(1)
                sid = sm.group(2).upper()
                return f"[{label}]({sid})"
            return md_link
        elif url:
            # Si l'URL pointe directement vers une séquence seule (ex: https://oeis.org/A001622)
            sm = re.match(r"^https?://(?:www\.)?oeis\.org/+(A\d{6})/?$", url, re.I)
            if sm:
                sid = sm.group(1).upper()
                return f"[{sid}]({sid})"
            return url
        elif a_tag:
            sm = re.match(r"^<a\s+href=[\"']?(?:https?://(?:www\.)?oeis\.org)?/+(A\d{6})/?[\"']?\s*>(.*?)</a>$", a_tag, re.I | re.DOTALL)
            if sm:
                sid = sm.group(1).upper()
                label = sm.group(2).strip()
                return f"[{label}]({sid})"
            sm_other = re.match(r"^<a\s+href=[\"']?([^\"' >]+)[\"']?\s*>(.*?)</a>$", a_tag, re.I | re.DOTALL)
            if sm_other:
                href = sm_other.group(1).strip()
                label = sm_other.group(2).strip()
                if href.startswith("/"):
                    href = f"https://oeis.org{href}"
                elif not (href.startswith("http://") or href.startswith("https://") or href.startswith("mailto:")):
                    href = f"https://oeis.org/{href}"
                return f"[{label}]({href})"
            return a_tag
        elif other_tag:
            return other_tag
        elif seq_name:
            sid = seq_name.upper()
            return f"[{sid}]({sid})"
        elif username:
            if username in ("Start", "End"):
                return m.group(0)
            user_slug = "_".join(username.strip().split())
            return f"[_{username}_](https://oeis.org/wiki/User:{user_slug})"
        return m.group(0)

    return token_pattern.sub(repl, text)


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
    return localize_sequence_links(line)


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
        tag, line_sid = parts[0], parts[1]
        if tag in ('p', 't', 'o'):
            val = parts[2].rstrip()
        else:
            val = parts[2].strip()
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


def fetch_oeis_seq_from_github(seq_id: str, timeout: int = 10, conn=None) -> dict:
    """
    Télécharge et parse le fichier .seq d'une séquence depuis GitHub (oeisdata) :
    https://raw.githubusercontent.com/oeis/oeisdata/refs/heads/main/seq/{prefix}/{sid}.seq
    selon la spécification https://oeis.org/eishelp1.html.
    Met à jour ou crée les DATA dans la base de données SQLite.
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
            data = parse_oeis_seq(content, sid)
            sync_sequence_to_db(data, conn=conn)
            return data
    except Exception as e:
        logging.warning("Impossible de récupérer %s depuis %s : %s", sid, url, e)
        return None


def read_oeis_seq_file(file_path_or_content: str, sid: str = "", conn=None) -> dict:
    """
    Lit un fichier .seq (chemin de fichier sur le disque ou contenu texte),
    parse les données selon la spécification https://oeis.org/eishelp1.html
    et met à jour ou crée les DATA dans la base de données SQLite.
    """
    if os.path.exists(file_path_or_content) and os.path.isfile(file_path_or_content):
        with open(file_path_or_content, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if not sid:
            base = os.path.basename(file_path_or_content)
            name_part = os.path.splitext(base)[0]
            if name_part.upper().startswith('A') and name_part[1:].isdigit():
                sid = normalize_seq_id(name_part)
    else:
        content = file_path_or_content

    data = parse_oeis_seq(content, sid=sid)
    sync_sequence_to_db(data, conn=conn)
    return data


def sync_sequence_to_db(seq_data: dict, conn=None) -> None:
    """
    Enregistre ou met à jour les informations et les DATA d'une séquence dans SQLite :
    - Table 'sequences' (métadonnées et terms_preview)
    - Table 'seq' (termes numériques bruts avec compression RLE id, i, n, repeat)
    - Table 'ngram_index' (index inversé de 3-grams pour la recherche instantanée)
    - Table 'sequences_fts' (index plein texte FTS5)
    """
    if not seq_data or not seq_data.get('id'):
        return

    close_conn = False
    if conn is None:
        if os.path.exists(DEFAULT_DB_PATH):
            try:
                conn = sqlite3.connect(DEFAULT_DB_PATH)
                close_conn = True
            except Exception as e:
                logging.warning("Impossible d'ouvrir la base pour synchroniser %s : %s", seq_data.get('id'), e)
                return
        else:
            return

    try:
        cursor = conn.cursor()
        sid = normalize_seq_id(seq_data['id'])
        name = seq_data.get('name', '')
        offset_val = seq_data.get('offset_val', 0)
        keywords = seq_data.get('keywords', '')
        author = seq_data.get('author', '')

        # 1. Extraction et conversion des termes numériques (DATA)
        raw_terms = seq_data.get('terms', [])
        int_terms = []
        for t in raw_terms:
            try:
                s_val = str(t).strip()
                if s_val:
                    int_terms.append(int(s_val))
            except (ValueError, TypeError):
                continue

        terms_preview = ', '.join(str(t) for t in int_terms[:30]) if int_terms else seq_data.get('terms_preview', '')

        # 2. Mise à jour / Création dans la table sequences
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

        # 3. Mise à jour / Création des DATA dans la table seq (RLE)
        numeric_id = int(sid[1:]) if sid[1:].isdigit() else None
        if numeric_id is not None:
            cursor.execute("DELETE FROM seq WHERE id = ?", (numeric_id,))
            if int_terms:
                i = 1
                for n_val, group in itertools.groupby(int_terms):
                    r_val = len(list(group))
                    if -9223372036854775808 <= n_val <= 9223372036854775807:
                        cursor.execute("""
                            INSERT OR REPLACE INTO seq (id, i, n, repeat)
                            VALUES (?, ?, ?, ?)
                        """, (numeric_id, i, n_val, r_val))
                    i += r_val

        # 4. Mise à jour / Création des DATA dans la table ngram_index (3-Grams)
        cursor.execute("DELETE FROM ngram_index WHERE sequence_id = ?", (sid,))
        if len(int_terms) >= 3:
            for pos in range(len(int_terms) - 2):
                triplet = (int_terms[pos], int_terms[pos + 1], int_terms[pos + 2])
                triplet_hash = hash(triplet)
                cursor.execute("""
                    INSERT OR IGNORE INTO ngram_index (ngram_hash, sequence_id, offset_pos)
                    VALUES (?, ?, ?)
                """, (triplet_hash, sid, pos))

        # 5. Indexation plein-texte FTS5
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO sequences_fts (id, name, keywords, author)
                VALUES (?, ?, ?, ?)
            """, (sid, name, keywords, author))
        except Exception:
            pass

        conn.commit()
    except Exception as e:
        logging.warning("Erreur lors de la synchronisation des DATA de %s en base : %s", seq_data.get('id'), e)
    finally:
        if close_conn and conn:
            conn.close()


def get_sequence_data(seq_id, conn=None):
    """
    Récupère les données d'une séquence :
    1. Télécharge et parse le fichier .seq officiel depuis GitHub selon https://oeis.org/eishelp1.html
    2. Si disponible, synchronise avec la table SQLite et charge l'implémentation Python (Goulib Sequence)
    3. Si GitHub est inaccessible, bascule sur les informations locales en base.
    """
    sid = normalize_seq_id(seq_id)

    # 1. Extraction officielle depuis GitHub oeisdata (met à jour/crée la base SQLite et ses DATA)
    oeis_data = fetch_oeis_seq_from_github(sid, conn=conn)

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

    # Synchronisation de secours si besoin
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


def normalize_code_lang(lang: str) -> str:
    """Normalise l'identifiant de langage pour le bloc de code Markdown."""
    lang_clean = lang.strip().lower()
    mapping = {
        'c++': 'cpp',
        'c#': 'csharp',
        'f#': 'fsharp',
        'pari/gp': 'pari',
        'gp': 'pari',
        'sagemath': 'sagemath',
        'sage': 'sagemath',
        'mathematica': 'mathematica',
        'maple': 'maple',
        'maxima': 'maxima',
        'magma': 'magma',
        'axiom': 'axiom',
        'python': 'python',
        'haskell': 'haskell',
        'julia': 'julia',
    }
    return mapping.get(lang_clean, lang_clean)


def split_by_alternative(lines: list) -> list:
    """
    Découpe une liste de lignes de code en plusieurs sous-blocs
    dès qu'une ligne commence par un indicateur d'alternative (# Alternative:, (* Alternative: *), etc.)
    afin de produire des blocs de code séparés avec le même langage.
    """
    blocks = []
    current_block = []
    alt_regex = re.compile(r'^\s*(?:#|//|\\\\|--|\(\*)\s*Alternative\b', re.IGNORECASE)

    for line in lines:
        if alt_regex.match(line) and current_block:
            blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    return blocks


def format_sequence_markdown(data):
    """
    Formate les données d'une séquence en un document Markdown structuré inspiré d'OEIS
    utilisant les données extraites selon la spécification https://oeis.org/eishelp1.html.
    """
    sid = data['id']
    name = data.get('name') or 'Untitled sequence'
    offset_str = data.get('offset_str') or f"{data.get('offset_val', 0)}, 1"
    keywords = data.get('keywords', '') or 'core,nice'
    author = data.get('author') or 'N. J. A. Sloane'
    submitter = data.get('submitter') or 'goulu'
    identification = data.get('identification', '')

    md = []
    # En-tête et Titre
    md.append(f"# {sid}")
    md.append(f"## {name}\n")

    # Section Termes (DATA - %S, %T, %U)
    terms_list = data.get('terms', [])
    md.append("### DATA")
    md.append("```text")
    if terms_list:
        clean_terms = [str(t).strip() for t in terms_list if str(t).strip()]
        md.append(', '.join(clean_terms))
    elif data.get('terms_preview'):
        md.append(' '.join(str(data['terms_preview']).split()))
    else:
        md.append("Terms not computed")
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
            maple_blocks = split_by_alternative(maple_progs)
            for block in maple_blocks:
                code_text = '\n'.join(block).strip('\r\n')
                if code_text.strip():
                    md.append("```maple")
                    md.append(code_text)
                    md.append("```")
            md.append("")
        if math_progs:
            md.append("#### Mathematica")
            math_blocks = split_by_alternative(math_progs)
            for block in math_blocks:
                code_text = '\n'.join(block).strip('\r\n')
                if code_text.strip():
                    md.append("```mathematica")
                    md.append(code_text)
                    md.append("```")
            md.append("")
        if other_progs:
            md.append("#### Other Programs")
            programs_list = []
            current_lang = None
            current_lines = []
            lang_header_regex = re.compile(r'^\(([A-Z][a-zA-Z0-9#+/\.]{0,20}(?: [A-Za-z0-9#+/\.]{1,15})?)\)(?![a-zA-Z0-9_\.\[])\s*(.*)', re.DOTALL)

            for raw_line in other_progs:
                clean_stripped = raw_line.strip()
                match_lang = lang_header_regex.match(clean_stripped)
                if match_lang:
                    if current_lang is not None or current_lines:
                        programs_list.append((current_lang or "text", current_lines))
                    current_lang = match_lang.group(1).strip()
                    code_rest = match_lang.group(2)
                    current_lines = [code_rest] if code_rest.strip() else []
                else:
                    if current_lang is None:
                        current_lang = "text"
                    current_lines.append(raw_line)

            if current_lang is not None or current_lines:
                programs_list.append((current_lang or "text", current_lines))

            for lang, lines in programs_list:
                md.append(f"- **({lang})** :")
                fence_lang = normalize_code_lang(lang)
                sub_blocks = split_by_alternative(lines)
                for block in sub_blocks:
                    code_text = '\n'.join(block).strip('\r\n')
                    if code_text.strip():
                        md.append(f"```{fence_lang}")
                        md.append(code_text)
                        md.append("```")
            md.append("")

    # Section Implémentation Python Goulib / Sequence
    md.append("### PYTHON IMPLEMENTATION")
    if data.get('definition_code'):
        md.append("Executable sequence implementation based on `goulib.sequence.Sequence`:\n")
        if data.get('auxiliary_code'):
            md.append("#### Auxiliary functions:")
            md.append("```python")
            md.append(data['auxiliary_code'].strip())
            md.append("```\n")

        md.append("#### Main definition:")
        md.append("```python")
        md.append(data['definition_code'].strip())
        md.append("```\n")

        md.append("#### Detail of provided methods:")
        md.append("| Function | Expression | Description |")
        md.append("| :--- | :--- | :--- |")
        iterf_str = f"`{data.get('iterf_code')}`" if data.get('iterf_code') else "*(none / count)*"
        itemf_str = f"`{data.get('itemf_code')}`" if data.get('itemf_code') else "*(sequential evaluation)*"
        containf_str = f"`{data.get('containf_code')}`" if data.get('containf_code') else "*(monotonic search)*"
        md.append(f"| **`iterf`** | {iterf_str} | Generator iterating over the infinite sequence |")
        md.append(f"| **`itemf(n)`** | {itemf_str} | Direct computation of $n$-th term |")
        md.append(f"| **`containf(x)`** | {containf_str} | Boolean membership test ($x \\in S$) |")
    else:
        md.append("*No Python implementation recorded in the local database yet.*")
    md.append("")

    # Section Renvois et Cross-references (CROSSREFERENCES - %Y)
    crossrefs = data.get('crossrefs', [])
    md.append("### CROSSREFERENCES")
    if crossrefs:
        for y in crossrefs:
            md.append(f"- {format_crossref(y.strip())}")
    md.append(f"- See also the official entry at <https://oeis.org/{sid}>\n")

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
            raise ValueError(f"Sequence {sid} not found on GitHub oeisdata or local database.")

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
    print(f"Markdown file successfully generated: {fpath}")
    print(f"Generated file size: {len(content)} bytes")
