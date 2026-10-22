"""
Serveur Web pour le clone OEIS.
- Convertit à la volée les fichiers Markdown des séquences en HTML.
- Génère automatiquement le fichier Markdown après modification de la base.
- Fournit une barre de recherche inspirée de hints.html et une console de test interactive.
"""

import os
import sqlite3
import markdown
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from markdown_generator import generate_sequence_markdown, DEFAULT_DB_PATH, DEFAULT_OUTPUT_DIR
from search_engine import OEISSearchEngine

app = FastAPI(title="OEIS Python Clone", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'oeis.db')
SEQUENCES_DIR = os.path.join(BASE_DIR, 'sequences')
search_engine = OEISSearchEngine(DB_PATH)



def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def render_page(title: str, content_html: str, query: str = "", seq_id: str = "") -> str:
    """Génère le gabarit HTML complet avec le design moderne inspiré d'OEIS."""
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - The On-Line Encyclopedia of Integer Sequences</title>
    <!-- MathJax pour rendu des formules LaTeX -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {{
            --bg-body: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --primary: #0284c7;
            --primary-hover: #0369a1;
            --oeis-blue: #034f84;
            --oeis-header-bg: #e6f0fa;
            --border-color: #cbd5e1;
            --code-bg: #f1f5f9;
            --highlight: #fef08a;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            line-height: 1.6;
        }}

        /* Header Style OEIS */
        header {{
            background: linear-gradient(180deg, #dbeafe 0%, #eff6ff 100%);
            border-bottom: 2px solid #93c5fd;
            padding: 1rem 2rem;
        }}

        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto 0.8rem auto;
        }}

        .logo-area {{
            display: flex;
            align-items: center;
            gap: 1rem;
            text-decoration: none;
            color: var(--oeis-blue);
        }}

        .logo-icon {{
            background: var(--oeis-blue);
            color: white;
            font-weight: 800;
            font-size: 1.4rem;
            padding: 0.3rem 0.7rem;
            border-radius: 6px;
            letter-spacing: 1px;
        }}

        .logo-text h1 {{
            font-size: 1.3rem;
            font-weight: 700;
            line-height: 1.2;
        }}

        .logo-text span {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .header-links a {{
            color: var(--primary);
            text-decoration: none;
            font-size: 0.9rem;
            margin-left: 1rem;
            font-weight: 500;
        }}
        .header-links a:hover {{
            text-decoration: underline;
        }}

        /* Search Bar */
        .search-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .search-form {{
            display: flex;
            gap: 0.5rem;
        }}

        .search-input {{
            flex: 1;
            padding: 0.65rem 1rem;
            font-size: 1rem;
            border: 2px solid #93c5fd;
            border-radius: 6px;
            outline: none;
            transition: border-color 0.2s;
        }}

        .search-input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.2);
        }}

        .search-btn {{
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0 1.5rem;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }}

        .search-btn:hover {{
            background-color: var(--primary-hover);
        }}

        .search-hints {{
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 0.3rem;
        }}

        /* Main Content Container */
        main {{
            max-width: 1200px;
            margin: 1.5rem auto;
            padding: 0 1rem;
        }}

        /* Action bar for sequence */
        .sequence-actions {{
            background: white;
            padding: 0.5rem 1rem;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .sequence-actions .btn-group a {{
            color: var(--primary);
            text-decoration: none;
            font-size: 0.85rem;
            margin-right: 0.8rem;
            font-weight: 500;
        }}
        .sequence-actions .btn-group a:hover {{
            text-decoration: underline;
        }}

        /* Sequence Card & Markdown Body */
        .sequence-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}

        .markdown-body h1 {{
            color: var(--oeis-blue);
            font-size: 1.8rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0.4rem;
            margin-bottom: 0.5rem;
        }}

        .markdown-body h2 {{
            color: #334155;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}

        .markdown-body h3 {{
            color: var(--oeis-blue);
            font-size: 1.05rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 0.2rem;
        }}

        .markdown-body h4 {{
            font-size: 0.95rem;
            color: #475569;
            margin-top: 1rem;
            margin-bottom: 0.3rem;
        }}

        .markdown-body pre {{
            background: var(--code-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
            overflow-x: auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0.8rem 0;
        }}

        .markdown-body code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: var(--code-bg);
            padding: 0.15rem 0.3rem;
            border-radius: 4px;
            font-size: 0.88rem;
        }}

        .markdown-body table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}

        .markdown-body table th, .markdown-body table td {{
            border: 1px solid var(--border-color);
            padding: 0.5rem 0.8rem;
            text-align: left;
            font-size: 0.9rem;
        }}

        .markdown-body table th {{
            background: #f8fafc;
            font-weight: 600;
        }}

        .markdown-body ul {{
            padding-left: 1.5rem;
            margin: 0.5rem 0;
        }}

        .markdown-body li {{
            margin-bottom: 0.3rem;
        }}

        .markdown-body blockquote {{
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
            padding: 0.5rem 1rem;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: #166534;
        }}

        /* Interactive Runner Box */
        .interactive-box {{
            margin-top: 2rem;
            padding: 1.2rem;
            background: #f8fafc;
            border: 1px dashed #94a3b8;
            border-radius: 8px;
        }}

        .interactive-box h4 {{
            color: var(--oeis-blue);
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .test-controls {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 0.8rem;
        }}

        .test-input-group {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .test-input {{
            padding: 0.4rem 0.6rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            width: 100px;
        }}

        .test-button {{
            background: var(--oeis-blue);
            color: white;
            border: none;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
        }}
        .test-button:hover {{
            background: #02365c;
        }}

        .test-output {{
            margin-top: 0.8rem;
            padding: 0.6rem;
            background: white;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9rem;
            min-height: 2.2rem;
            color: #0f172a;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            margin-top: 3rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-top">
            <a href="/" class="logo-area">
                <div class="logo-icon">OEIS</div>
                <div class="logo-text">
                    <h1>The On-Line Encyclopedia of Integer Sequences®</h1>
                    <span>Python Sequence Platform & Collaboration</span>
                </div>
            </a>
            <div class="header-links">
                <a href="/">Accueil</a>
                <a href="/A000045">A000045 (Fibonacci)</a>
                <a href="/hints.html">Aide Recherche</a>
            </div>
        </div>
        <div class="search-container">
            <form action="/search" method="GET" class="search-form">
                <input type="text" name="q" value="{query}" placeholder="Recherche par termes (ex: 1, 1, 2, 3, 5, 8) ou mot-clé ou identifiant (ex: A000045)..." class="search-input">
                <button type="submit" class="search-btn">Rechercher</button>
            </form>
            <div class="search-hints">
                Exemples : <code>1, 1, 2, 3, 5, 8</code> &bull; <code>A000045</code> &bull; <code>Fibonacci</code> &bull; <code>author:Guglielmetti</code>
            </div>
        </div>
    </header>

    <main>
        {content_html}
    </main>

    <footer>
        <p>Inspiré par Neil Sloane's OEIS Foundation &bull; Implémentation basée sur Goulib <code>Sequence</code> &bull; Rendu Markdown &rarr; HTML à la volée</p>
    </footer>

    <script>
    async function testItemf(seqId) {{
        const n = document.getElementById('itemf-n').value;
        const out = document.getElementById('test-result');
        out.innerText = 'Calcul en cours...';
        try {{
            const res = await fetch(`/api/sequence/${{seqId}}/eval?itemf_n=${{n}}`);
            const data = await res.json();
            if (data.error) {{
                out.innerText = 'Erreur : ' + data.error;
            }} else {{
                out.innerText = `itemf(${{n}}) = ${{data.result}}`;
            }}
        }} catch(e) {{
            out.innerText = 'Erreur réseau : ' + e.message;
        }}
    }}

    async function testContainf(seqId) {{
        const x = document.getElementById('containf-x').value;
        const out = document.getElementById('test-result');
        out.innerText = 'Test en cours...';
        try {{
            const res = await fetch(`/api/sequence/${{seqId}}/eval?containf_x=${{x}}`);
            const data = await res.json();
            if (data.error) {{
                out.innerText = 'Erreur : ' + data.error;
            }} else {{
                out.innerText = `containf(${{x}}) = ${{data.result}} (${{data.result ? 'Appartient à la suite' : 'N\'appartient PAS à la suite'}})`;
            }}
        }} catch(e) {{
            out.innerText = 'Erreur réseau : ' + e.message;
        }}
    }}
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Page d'accueil avec moteur de recherche et aperçu des suites remarquables."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM sequences")
    nb_seqs = cursor.fetchone()[0]

    # Récupérer quelques suites phares
    sample_ids = ('A000045', 'A000040', 'A000079', 'A000217', 'A000290', 'A000004')
    cursor.execute(f"SELECT id, name, terms_preview FROM sequences WHERE id IN ({','.join(['?']*len(sample_ids))})", sample_ids)
    samples = cursor.fetchall()
    conn.close()

    samples_html = '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin-top: 1.5rem;">'
    for s in samples:
        samples_html += f"""
        <div style="background: white; border: 1px solid var(--border-color); border-radius: 6px; padding: 1.2rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; font-size: 1.1rem;"><a href="/{s['id']}" style="color: var(--primary); text-decoration: none;">{s['id']}</a></div>
            <div style="color: #475569; font-size: 0.9rem; margin: 0.4rem 0;">{s['name']}</div>
            <div style="font-family: monospace; font-size: 0.8rem; background: var(--code-bg); padding: 0.4rem; border-radius: 4px; overflow-x: auto; white-space: nowrap;">{s['terms_preview']}</div>
        </div>
        """
    samples_html += '</div>'

    home_content = f"""
    <div class="sequence-card">
        <h2 style="color: var(--oeis-blue); font-size: 1.5rem; margin-bottom: 0.5rem;">Bienvenue sur le Clone OEIS en Python</h2>
        <p style="color: var(--text-muted); font-size: 1rem; margin-bottom: 1.5rem;">
            Plateforme collaborative pour suites numériques infinies intégrant la classe <code>Sequence</code> avec les fonctions <code>iterf</code>, <code>itemf</code> et <code>containf</code>.
        </p>

        <div style="display: flex; gap: 2rem; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <div><strong>{nb_seqs}</strong> suites indexées</div>
            <div><strong>100%</strong> définies avec <code>Sequence</code></div>
            <div><strong>Génération Markdown</strong> automatique & conversion HTML à la volée</div>
        </div>

        <h3 style="color: var(--oeis-blue); border-bottom: 1px solid var(--border-color); padding-bottom: 0.3rem;">Suites Remarquables</h3>
        {samples_html}
    </div>
    """
    return render_page("Accueil", home_content)


@app.get("/hints.html", response_class=HTMLResponse)
def hints_page():
    """Page d'aide à la recherche conforme à hints.html."""
    hints_content = """
    <div class="sequence-card">
        <h2 style="color: var(--oeis-blue); font-size: 1.5rem; margin-bottom: 0.5rem;">Aide et Syntaxe de Recherche</h2>
        <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
            Le moteur de recherche implémente les fonctionnalités décrites sur <a href="https://oeis.org/hints.html" target="_blank">oeis.org/hints.html</a>.
        </p>

        <h3 style="color: var(--oeis-blue);">1. Recherche de séquences de nombres</h3>
        <ul>
            <li><strong>Termes consécutifs</strong> : entrez les termes séparés par des virgules ou des espaces :<br>
                <code>1, 1, 2, 3, 5, 8, 13</code> ou <code>10946, 17711, 28657</code>
            </li>
            <li><strong>Wildcard d'un terme (<code>_</code>)</strong> : remplace un nombre quelconque :<br>
                <code>1, 2, _, 4, 5</code>
            </li>
            <li><strong>Wildcard de sous-séquence (<code>__</code>)</strong> : remplace une suite de termes :<br>
                <code>1, 2, __, 8, 13</code>
            </li>
        </ul>

        <h3 style="color: var(--oeis-blue); margin-top: 1.5rem;">2. Préfixes de champs</h3>
        <ul>
            <li><code>id:A000045</code> : recherche par identifiant direct</li>
            <li><code>author:Philippe</code> : recherche par nom d'auteur</li>
            <li><code>keyword:core</code> : recherche par mot-clé OEIS</li>
        </ul>

        <h3 style="color: var(--oeis-blue); margin-top: 1.5rem;">3. Recherche plein texte</h3>
        <p>Les mots simples sont recherchés dans les descriptions et métadonnées via l'index FTS5 :</p>
        <ul>
            <li><code>Fibonacci</code></li>
            <li><code>square numbers</code></li>
            <li><code>prime</code></li>
        </ul>
    </div>
    """
    return render_page("Aide Recherche", hints_content)


@app.get("/search", response_class=HTMLResponse)
def search_page(q: str = ""):
    """Recherche conforme à hints.html via OEISSearchEngine."""
    query = q.strip()
    if not query:
        return RedirectResponse(url="/")

    # Si c'est un A-number direct sans préfixe
    if query.upper().startswith('A') and query[1:].isdigit():
        return RedirectResponse(url=f"/{query.upper()}")

    # Exécution de la recherche via OEISSearchEngine
    results = search_engine.search(query, limit=50)

    res_html = f'<div class="sequence-card"><h2>Résultats de recherche pour : "{query}"</h2>'
    if not results:
        res_html += f"""
        <p style="color: var(--text-muted); margin-top: 1rem;">
            Aucune séquence trouvée pour cette requête.
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #64748b;">
            Conseils : essayez avec moins de termes, vérifiez les virgules ou consultez <a href="/hints.html">l'aide à la recherche</a>.
        </p>
        """
    else:
        res_html += f'<p style="color: var(--text-muted); margin-bottom: 1.5rem;">{len(results)} résultat(s) trouvé(s) (classés par score de pertinence) :</p>'
        for r in results:
            score = r.get('score', 0)
            score_badge = f'<span style="background: #e0f2fe; color: #0369a1; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700; font-size: 0.8rem; margin-left: 0.5rem;">+{score}</span>'
            res_html += f"""
            <div style="border-bottom: 1px solid var(--border-color); padding: 1rem 0;">
                <div style="font-size: 1.15rem; font-weight: 700;">
                    <a href="/{r['id']}" style="color: var(--primary); text-decoration: none;">{r['id']}</a> : {r['name']} {score_badge}
                </div>
                <div style="font-family: monospace; font-size: 0.85rem; background: var(--code-bg); padding: 0.4rem; border-radius: 4px; margin: 0.4rem 0;">
                    {r['terms']}
                </div>
                <div style="font-size: 0.82rem; color: #16a34a; font-weight: 500;">&check; {r.get('match_desc', '')}</div>
            </div>
            """
    res_html += '</div>'
    return render_page(f"Recherche: {query}", res_html, query=query)


@app.get("/raw/{seq_id}.md", response_class=PlainTextResponse)
def raw_markdown(seq_id: str):
    """Retourne le contenu Markdown brut du fichier sur le disque."""
    sid = seq_id.strip().upper()
    if sid.startswith('A') and sid[1:].isdigit():
        sid = f"A{sid[1:].zfill(6)}"
    md_file_path = os.path.join(SEQUENCES_DIR, f"{sid}.md")
    if not os.path.exists(md_file_path):
        conn = get_db()
        try:
            generate_sequence_markdown(sid, conn=conn, output_dir=SEQUENCES_DIR)
        except ValueError:
            raise HTTPException(status_code=404, detail="Séquence introuvable.")
        finally:
            conn.close()

    with open(md_file_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/sequence/{seq_id}/eval")
def api_eval_sequence(seq_id: str, itemf_n: int = None, containf_x: int = None):
    """
    Évalue en direct l'objet Sequence correspondant à seq_id dans le module oeis.
    """
    sid = seq_id.strip().upper()
    try:
        import oeis
        if sid not in oeis.oeis:
            return {"error": f"Séquence {sid} introuvable dans le module Python."}
        seq = oeis.oeis[sid]

        if itemf_n is not None:
            if seq.itemf:
                val = seq.itemf(itemf_n)
            else:
                val = seq[itemf_n]
            return {"sequence": sid, "type": "itemf", "n": itemf_n, "result": str(val)}

        if containf_x is not None:
            if seq.containf:
                is_in = seq.containf(containf_x)
            else:
                is_in = containf_x in seq
            return {"sequence": sid, "type": "containf", "x": containf_x, "result": bool(is_in)}

        return {"error": "Paramètre itemf_n ou containf_x requis."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/sequence/{seq_id}/update")
def api_update_sequence(
    seq_id: str,
    name: str = Form(None),
    definition_code: str = Form(None),
    iterf_code: str = Form(None),
    itemf_code: str = Form(None),
    containf_code: str = Form(None)
):
    """
    Met à jour une séquence dans la base de données SQLite et
    DÉCLENCHE LA RÉGÉNÉRATION IMMÉDIATE DU FICHIER MARKDOWN correspondant.
    """
    sid = seq_id.strip().upper()
    conn = get_db()
    cursor = conn.cursor()

    # 1. Mise à jour de la table sequences
    if name:
        cursor.execute("UPDATE sequences SET name = ? WHERE id = ?", (name, sid))

    # 2. Insertion d'une nouvelle implémentation
    if definition_code:
        cursor.execute("""
            INSERT INTO implementations (
                sequence_id, user_id, status, definition_code,
                iterf_code, itemf_code, containf_code, description, test_status
            ) VALUES (?, 1, 'approved', ?, ?, ?, ?, ?, 'passed')
        """, (sid, definition_code, iterf_code, itemf_code, containf_code, name or ''))
        new_impl_id = cursor.lastrowid
        cursor.execute("UPDATE sequences SET current_implementation_id = ? WHERE id = ?", (new_impl_id, sid))

    conn.commit()

    # 3. CRÉATION / MISE À JOUR DU FICHIER MARKDOWN APRÈS MODIFICATION DE LA BD
    file_path, _ = generate_sequence_markdown(sid, conn=conn, output_dir=SEQUENCES_DIR)
    conn.close()

    return {
        "status": "success",
        "message": f"Séquence {sid} mise à jour et fichier Markdown régénéré avec succès.",
        "markdown_file": file_path
    }


@app.get("/{seq_id}", response_class=HTMLResponse)
def sequence_page(seq_id: str):
    """
    Page d'affichage d'une séquence :
    1. Vérifie si le fichier Markdown existe. S'il n'existe pas, le génère à partir de la BD.
    2. Convertit le Markdown en HTML à la volée.
    3. Présente la page avec le style OEIS et le widget interactif de test.
    """
    # Nettoyage de l'ID (ex: A45 -> A000045)
    sid = seq_id.strip().upper()
    if sid.startswith('A') and sid[1:].isdigit():
        num = sid[1:]
        sid = f"A{num.zfill(6)}"
    else:
        raise HTTPException(status_code=404, detail="Format d'identifiant OEIS invalide (attendu: Axxxxxx)")

    md_file_path = os.path.join(SEQUENCES_DIR, f"{sid}.md")

    # Si le fichier Markdown n'existe pas encore, on le génère depuis SQLite
    if not os.path.exists(md_file_path):
        conn = get_db()
        try:
            generate_sequence_markdown(sid, conn=conn, output_dir=SEQUENCES_DIR)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Séquence {sid} introuvable dans la base.")
        finally:
            conn.close()

    # Lecture du fichier Markdown
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Conversion à la volée du Markdown en HTML
    html_body = markdown.markdown(
        md_text,
        extensions=[
            'markdown.extensions.extra',
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.toc'
        ]
    )

    # Barre d'actions & Widget interactif
    actions_bar = f"""
    <div class="sequence-actions">
        <div class="btn-group">
            <a href="/raw/{sid}.md" target="_blank">&#128196; Voir Markdown brut</a>
            <a href="https://oeis.org/{sid}" target="_blank" rel="noopener">&#127760; Page officielle OEIS.org</a>
            <a href="/edit/{sid}">&#9998; Proposer une implémentation</a>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
            Source : <code>sequences/{sid}.md</code>
        </div>
    </div>
    """

    interactive_box = f"""
    <div class="interactive-box">
        <h4>&#9881; Console d'évaluation en direct de la Sequence</h4>
        <p style="font-size: 0.85rem; color: var(--text-muted);">
            Testez directement dans cette page les fonctions <code>itemf(n)</code> et <code>containf(x)</code> :
        </p>
        <div class="test-controls">
            <div class="test-input-group">
                <label for="itemf-n">Calculer terme <em>n</em> :</label>
                <input type="number" id="itemf-n" value="10" min="0" max="1000" class="test-input">
                <button type="button" onclick="testItemf('{sid}')" class="test-button">Calculer itemf(n)</button>
            </div>
            <div class="test-input-group">
                <label for="containf-x">Tester nombre <em>x</em> :</label>
                <input type="number" id="containf-x" value="55" class="test-input">
                <button type="button" onclick="testContainf('{sid}')" class="test-button">Tester containf(x)</button>
            </div>
        </div>
        <div id="test-result" class="test-output">Résultat prêt. Cliquez sur un bouton ci-dessus.</div>
    </div>
    """

    full_content = f"""
    {actions_bar}
    <div class="sequence-card markdown-body">
        {html_body}
        {interactive_box}
    </div>
    """

    return render_page(f"{sid}", full_content, query=sid, seq_id=sid)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

