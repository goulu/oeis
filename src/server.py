"""
Serveur Web pour le clone OEIS.
- Convertit à la volée les fichiers Markdown des séquences en HTML.
- Génère automatiquement le fichier Markdown après modification de la base.
- Fournit une barre de recherche inspirée de hints.html et une console de test interactive.
"""

import os
import json
import sqlite3
import markdown
from fastapi import FastAPI, HTTPException, Request, Form, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from markdown_generator import generate_sequence_markdown, DEFAULT_DB_PATH, DEFAULT_OUTPUT_DIR
from search_engine import OEISSearchEngine

app = FastAPI(title="OEIS Python Clone", version="1.0.0")

ROOT_DIR = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'oeis.db')) or os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'seq')) else SRC_DIR
DB_PATH = os.path.join(ROOT_DIR, 'oeis.db')
SEQUENCES_DIR = os.path.join(ROOT_DIR, 'seq')
search_engine = OEISSearchEngine(DB_PATH)




def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def render_page(title: str, content_html: str, query: str = "", seq_id: str = "") -> str:
    """Génère le gabarit HTML complet avec le design moderne inspiré d'OEIS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - The On-Line Encyclopedia of Integer Sequences</title>
    <link rel="ai-catalog" href="/.well-known/ai-catalog.json">
    <link rel="api-catalog" href="/.well-known/api-catalog">
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
            overflow-x: hidden;
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
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: anywhere;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0.8rem 0;
        }}

        .markdown-body pre code {{
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: anywhere;
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

        /* Écran d'attente "Please wait" pour séquences en cours de génération */
        .please-wait-card {{
            max-width: 550px;
            margin: 4rem auto;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        }}

        .spinner {{
            width: 52px;
            height: 52px;
            border: 5px solid #e2e8f0;
            border-top: 5px solid var(--primary);
            border-radius: 50%;
            margin: 0 auto 1.5rem auto;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .please-wait-title {{
            font-size: 1.6rem;
            color: var(--oeis-blue);
            margin-bottom: 0.6rem;
        }}

        .please-wait-subtitle {{
            font-size: 0.95rem;
            color: var(--text-muted);
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }}

        .please-wait-badge {{
            display: inline-block;
            background: #e0f2fe;
            color: #0369a1;
            font-weight: 700;
            padding: 0.5rem 1.4rem;
            border-radius: 30px;
            font-size: 1.1rem;
            letter-spacing: 0.5px;
            animation: pulse 1.6s ease-in-out infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.8; transform: scale(1.03); }}
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
                <a href="/">Home</a>
                <a href="/A000045">A000045 (Fibonacci)</a>
                <a href="/hints.html">Search Hints</a>
            </div>
        </div>
        <div class="search-container">
            <form action="/search" method="GET" class="search-form" toolname="oeis_search" tooldescription="Search integer sequences in the OEIS database by numbers, ID, author, or keyword">
                <input type="text" name="q" value="{query}" placeholder="Search by terms (e.g. 1, 1, 2, 3, 5, 8), keyword, or sequence ID (e.g. A000045)..." class="search-input" toolparamdescription="Search query: numbers e.g. 1, 1, 2, 3, 5, 8 or ID e.g. A000045">
                <button type="submit" class="search-btn">Search</button>
            </form>
            <div class="search-hints">
                Examples: <code>1, 1, 2, 3, 5, 8</code> &bull; <code>A000045</code> &bull; <code>Fibonacci</code> &bull; <code>author:Guglielmetti</code>
            </div>
        </div>
    </header>

    <main>
        {content_html}
    </main>

    <footer>
        <p>Inspired by Neil Sloane's OEIS Foundation &bull; Implementation based on Goulib <code>Sequence</code> &bull; On-the-fly Markdown &rarr; HTML rendering</p>
    </footer>

    <script>
    async function testItemf(seqId) {{
        const n = document.getElementById('itemf-n').value;
        const out = document.getElementById('test-result');
        out.innerText = 'Computing...';
        try {{
            const res = await fetch(`/api/sequence/${{seqId}}/eval?itemf_n=${{n}}`);
            const data = await res.json();
            if (data.error) {{
                out.innerText = 'Error: ' + data.error;
            }} else {{
                out.innerText = `itemf(${{n}}) = ${{data.result}}`;
            }}
        }} catch(e) {{
            out.innerText = 'Network error: ' + e.message;
        }}
    }}

    async function testContainf(seqId) {{
        const x = document.getElementById('containf-x').value;
        const out = document.getElementById('test-result');
        out.innerText = 'Testing...';
        try {{
            const res = await fetch(`/api/sequence/${{seqId}}/eval?containf_x=${{x}}`);
            const data = await res.json();
            if (data.error) {{
                out.innerText = 'Error: ' + data.error;
            }} else {{
                out.innerText = `containf(${{x}}) = ${{data.result}} (${{data.result ? 'Belongs to sequence' : 'Does NOT belong to sequence'}})`;
            }}
        }} catch(e) {{
            out.innerText = 'Network error: ' + e.message;
        }}
    }}

    // WebMCP API tool registration for AI agents
    (function() {{
        try {{
            var mc = (typeof document !== 'undefined' && document.modelContext) || 
                     (typeof navigator !== 'undefined' && navigator.modelContext);
            if (mc && typeof mc.registerTool === 'function') {{
                mc.registerTool({{
                    name: "oeis_search",
                    description: "Search the On-Line Encyclopedia of Integer Sequences (OEIS) by numbers, keywords, or sequence ID",
                    inputSchema: {{
                        type: "object",
                        properties: {{
                            query: {{ type: "string", description: "Search query e.g. 1, 1, 2, 3, 5, 8 or A000045" }}
                        }},
                        required: ["query"]
                    }},
                    execute: async function(params) {{
                        var q = params && params.query ? encodeURIComponent(params.query) : '';
                        var res = await fetch('/api/search?q=' + q);
                        return await res.json();
                    }}
                }}).catch(function() {{}});
            }}
        }} catch (e) {{}}
    }})();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request = None):
    """Page d'accueil avec moteur de recherche, Link headers (RFC 8288) et support Markdown."""
    if request:
        accept = request.headers.get("accept", "").lower()
        if "text/markdown" in accept:
            md_text = (
                "# The On-Line Encyclopedia of Integer Sequences (OEIS)\n\n"
                "Collaborative platform for infinite integer sequences integrating the `Sequence` class with `iterf`, `itemf` and `containf` methods.\n\n"
                "## Key Resources\n"
                "- Direct sequence lookup: `/{Axxxxxx}` (e.g. `/A000045`)\n"
                "- Search sequences: `/search?q={query}` (e.g. `/search?q=1,1,2,3,5,8`)\n"
                "- Search hints: `/hints.html`\n"
                "- API Catalog: `/.well-known/api-catalog`\n"
                "- AI Capabilities Manifest: `/.well-known/ai-catalog.json`\n"
                "- OpenAPI Specification: `/openapi.json`\n"
                "- API Documentation: `/docs`\n"
            )
            tokens = str(len(md_text.split()) * 4 // 3)
            return Response(
                content=md_text,
                media_type="text/markdown; charset=utf-8",
                headers={"x-markdown-tokens": tokens}
            )

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
            <div style="font-family: monospace; font-size: 0.8rem; background: var(--code-bg); padding: 0.4rem; border-radius: 4px; overflow-wrap: anywhere; word-break: break-word;">{s['terms_preview']}</div>
        </div>
        """
    samples_html += '</div>'

    home_content = f"""
    <div class="sequence-card">
        <h2 style="color: var(--oeis-blue); font-size: 1.5rem; margin-bottom: 0.5rem;">Welcome to the OEIS Python Clone</h2>
        <p style="color: var(--text-muted); font-size: 1rem; margin-bottom: 1.5rem;">
            Collaborative platform for infinite integer sequences integrating the <code>Sequence</code> class with <code>iterf</code>, <code>itemf</code> and <code>containf</code> methods.
        </p>

        <div style="display: flex; gap: 2rem; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <div><strong>{nb_seqs}</strong> indexed sequences</div>
            <div><strong>100%</strong> defined with <code>Sequence</code></div>
            <div><strong>Automatic Markdown generation</strong> & on-the-fly HTML conversion</div>
        </div>

        <h3 style="color: var(--oeis-blue); border-bottom: 1px solid var(--border-color); padding-bottom: 0.3rem;">Featured Sequences</h3>
        {samples_html}
    </div>
    """
    headers = {
        "Link": '</.well-known/api-catalog>; rel="api-catalog", </docs>; rel="service-doc", </openapi.json>; rel="service-desc"'
    }
    return HTMLResponse(content=render_page("Home", home_content), headers=headers)


@app.get("/hints.html", response_class=HTMLResponse)
def hints_page(request: Request = None):
    """Page d'aide à la recherche conforme à hints.html."""
    if request:
        accept = request.headers.get("accept", "").lower()
        if "text/markdown" in accept:
            md_text = (
                "# Search Hints and Syntax\n\n"
                "## 1. Number sequence search\n"
                "- Consecutive terms: `1, 1, 2, 3, 5, 8, 13`\n"
                "- Single term wildcard (`_`): `1, 2, _, 4, 5`\n"
                "- Subsequence wildcard (`__`): `1, 2, __, 8, 13`\n\n"
                "## 2. Field prefixes\n"
                "- `id:A000045`: direct sequence lookup\n"
                "- `author:Philippe`: search by author name\n"
                "- `keyword:core`: search by OEIS keyword\n\n"
                "## 3. Full-text search\n"
                "- Simple words are searched in descriptions and metadata (e.g. `Fibonacci`, `prime`).\n"
            )
            tokens = str(len(md_text.split()) * 4 // 3)
            return Response(
                content=md_text,
                media_type="text/markdown; charset=utf-8",
                headers={"x-markdown-tokens": tokens}
            )

    hints_content = """
    <div class="sequence-card">
        <h2 style="color: var(--oeis-blue); font-size: 1.5rem; margin-bottom: 0.5rem;">Search Hints and Syntax</h2>
        <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
            The search engine implements features described at <a href="https://oeis.org/hints.html" target="_blank">oeis.org/hints.html</a>.
        </p>

        <h3 style="color: var(--oeis-blue);">1. Number sequence search</h3>
        <ul>
            <li><strong>Consecutive terms</strong>: enter terms separated by commas or spaces:<br>
                <code>1, 1, 2, 3, 5, 8, 13</code> or <code>10946, 17711, 28657</code>
            </li>
            <li><strong>Single term wildcard (<code>_</code>)</strong>: matches any single number:<br>
                <code>1, 2, _, 4, 5</code>
            </li>
            <li><strong>Subsequence wildcard (<code>__</code>)</strong>: matches a sequence of terms:<br>
                <code>1, 2, __, 8, 13</code>
            </li>
        </ul>

        <h3 style="color: var(--oeis-blue); margin-top: 1.5rem;">2. Field prefixes</h3>
        <ul>
            <li><code>id:A000045</code>: direct sequence lookup</li>
            <li><code>author:Philippe</code>: search by author name</li>
            <li><code>keyword:core</code>: search by OEIS keyword</li>
        </ul>

        <h3 style="color: var(--oeis-blue); margin-top: 1.5rem;">3. Full-text search</h3>
        <p>Simple words are searched in descriptions and metadata using the FTS5 index:</p>
        <ul>
            <li><code>Fibonacci</code></li>
            <li><code>square numbers</code></li>
            <li><code>prime</code></li>
        </ul>
    </div>
    """
    return render_page("Search Hints", hints_content)


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

    res_html = f'<div class="sequence-card"><h2>Search results for: "{query}"</h2>'
    if not results:
        res_html += f"""
        <p style="color: var(--text-muted); margin-top: 1rem;">
            No sequences found for this query.
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #64748b;">
            Tips: try fewer terms, check commas, or see <a href="/hints.html">search hints</a>.
        </p>
        """
    else:
        res_html += f'<p style="color: var(--text-muted); margin-bottom: 1.5rem;">{len(results)} result(s) found (ranked by relevance score):</p>'
        for r in results:
            score = r.get('score', 0)
            score_badge = f'<span style="background: #e0f2fe; color: #0369a1; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700; font-size: 0.8rem; margin-left: 0.5rem;">+{score}</span>'
            res_html += f"""
            <div style="border-bottom: 1px solid var(--border-color); padding: 1rem 0;">
                <div style="font-size: 1.15rem; font-weight: 700;">
                    <a href="/{r['id']}" style="color: var(--primary); text-decoration: none;">{r['id']}</a> : {r['name']} {score_badge}
                </div>
                <div style="font-family: monospace; font-size: 0.85rem; background: var(--code-bg); padding: 0.4rem; border-radius: 4px; margin: 0.4rem 0; overflow-wrap: anywhere; word-break: break-word;">
                    {r['terms']}
                </div>
                <div style="font-size: 0.82rem; color: #16a34a; font-weight: 500;">&check; {r.get('match_desc', '')}</div>
            </div>
            """
    res_html += '</div>'
    return render_page(f"Search: {query}", res_html, query=query)


@app.get("/sitemap.xml")
def sitemap_xml():
    """Génère le sitemap XML canonique pour le site selon le protocole sitemaps.org."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM sequences ORDER BY id")
    seq_rows = cursor.fetchall()
    conn.close()

    base_url = "https://oeis.drgoulu.com"
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
        f'  <url><loc>{base_url}/</loc><changefreq>daily</changefreq><priority>1.0</priority></url>',
        f'  <url><loc>{base_url}/hints.html</loc><changefreq>monthly</changefreq><priority>0.8</priority></url>',
    ]
    for row in seq_rows:
        sid = row[0]
        xml_lines.append(f'  <url><loc>{base_url}/{sid}</loc><changefreq>weekly</changefreq><priority>0.6</priority></url>')
    xml_lines.append('</urlset>')
    return Response(content="\n".join(xml_lines), media_type="application/xml; charset=utf-8")


@app.get("/.well-known/api-catalog")
def api_catalog():
    """Publie le catalogue d'API selon la RFC 9727."""
    catalog = {
        "linkset": [
            {
                "anchor": "https://oeis.drgoulu.com/",
                "service-desc": [
                    { "href": "https://oeis.drgoulu.com/openapi.json", "type": "application/json" }
                ],
                "service-doc": [
                    { "href": "https://oeis.drgoulu.com/docs", "type": "text/html" }
                ],
                "status": [
                    { "href": "https://oeis.drgoulu.com/api/search?q=1,2,3", "type": "application/json" }
                ]
            }
        ]
    }
    return Response(
        content=json.dumps(catalog, indent=2),
        media_type="application/linkset+json; charset=utf-8"
    )


@app.get("/.well-known/mcp/server-card.json")
def mcp_server_card():
    """Publie l'MCP Server Card (SEP-1649) pour la découverte par agents."""
    card = {
        "serverInfo": {
            "name": "oeis-mcp",
            "version": "1.0.0",
            "description": "On-Line Encyclopedia of Integer Sequences (OEIS) Agent MCP Server"
        },
        "endpoint": "https://oeis.drgoulu.com/mcp",
        "capabilities": {
            "tools": { "listChanged": False },
            "resources": { "subscribe": False, "listChanged": False },
            "prompts": { "listChanged": False }
        }
    }
    return Response(
        content=json.dumps(card, indent=2),
        media_type="application/json; charset=utf-8"
    )


@app.get("/.well-known/agent-skills/index.json")
def agent_skills_index():
    """Publie l'index de découverte des compétences d'agents (Agent Skills Discovery RFC)."""
    index = {
        "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
        "skills": [
            {
                "name": "oeis-lookup",
                "type": "skill-md",
                "description": "Look up integer sequences by A-number or sequence terms in the OEIS database.",
                "url": "https://oeis.drgoulu.com/hints.html",
                "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            }
        ]
    }
    return Response(
        content=json.dumps(index, indent=2),
        media_type="application/json; charset=utf-8"
    )


@app.get("/.well-known/ai-catalog.json")
def ai_catalog():
    """Publie le manifeste de découverte de ressources d'agents (ARD / ai-catalog)."""
    manifest = {
        "specVersion": "1.0",
        "host": {
            "displayName": "OEIS Python Clone",
            "identifier": "did:web:oeis.drgoulu.com"
        },
        "entries": [
            {
                "identifier": "urn:air:oeis.drgoulu.com:service:search",
                "displayName": "OEIS Sequence Search & Lookup",
                "type": "application/json",
                "url": "https://oeis.drgoulu.com/api/search",
                "representativeQueries": [
                    "lookup Fibonacci numbers A000045",
                    "search integer sequence 1, 1, 2, 3, 5, 8",
                    "find prime number sequence A000040"
                ]
            },
            {
                "identifier": "urn:air:oeis.drgoulu.com:server:mcp",
                "displayName": "OEIS MCP Server Card",
                "type": "application/mcp-server-card+json",
                "url": "https://oeis.drgoulu.com/.well-known/mcp/server-card.json",
                "representativeQueries": [
                    "query integer sequences via MCP",
                    "access OEIS tools"
                ]
            }
        ]
    }
    return Response(
        content=json.dumps(manifest, indent=2),
        media_type="application/json; charset=utf-8",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/.well-known/agent-card.json")
def agent_card():
    """Publie la carte d'agent A2A (Agent-to-Agent protocol)."""
    card = {
        "name": "OEIS Python Agent",
        "version": "1.0.0",
        "description": "Agent for discovering, searching, computing, and analyzing integer sequences from OEIS.",
        "supportedInterfaces": [
            {
                "serviceUrl": "https://oeis.drgoulu.com/",
                "protocol": "http"
            }
        ],
        "capabilities": {
            "tools": True,
            "streaming": False
        },
        "skills": [
            {
                "id": "oeis-search",
                "name": "OEIS Search",
                "description": "Search sequences by number, terms, or keywords"
            },
            {
                "id": "oeis-evaluate",
                "name": "OEIS Sequence Evaluation",
                "description": "Compute term n or test containment for supported sequences"
            }
        ]
    }
    return Response(
        content=json.dumps(card, indent=2),
        media_type="application/json; charset=utf-8",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/auth.md", response_class=PlainTextResponse)
def auth_md():
    """Publie les métadonnées et instructions d'authentification pour agents (auth.md)."""
    content = """# auth.md

## Authentication & Access for AI Agents

The OEIS Python service at `https://oeis.drgoulu.com` provides open, unauthenticated access to integer sequence data, metadata, and calculation tools.

### Public Access Policy
- **Authentication**: None required. All search, calculation, and documentation endpoints are publicly accessible.
- **Audience**: AI Agents, web crawlers, and developers.
- **Endpoints**:
  - `GET /api/search?q={query}` : Search sequences by ID, terms, or keywords.
  - `GET /api/compute/{seq_id}` : Evaluate itemf(n) or containf(x).
  - `GET /{seq_id}` : Get HTML or Markdown (with `Accept: text/markdown`).
- **Fair Use**: Rate limiting is minimal but agents are expected to respect `robots.txt` guidelines.

### Discovery Endpoints
- A2A Agent Card: `/.well-known/agent-card.json`
- Model Context Protocol (MCP): `/.well-known/mcp/server-card.json`
- API Catalog (RFC 9727): `/.well-known/api-catalog`
- Agentic Resource Discovery (ARD): `/.well-known/ai-catalog.json`
- Agent Skills Index: `/.well-known/agent-skills/index.json`
"""
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Type": "text/markdown; charset=utf-8"}
    )


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    """Publie /robots.txt conforme à la RFC 9309 (Robots Exclusion Protocol)."""
    robots_file = os.path.join(ROOT_DIR, "robots.txt")
    if os.path.exists(robots_file):
        with open(robots_file, "r", encoding="utf-8") as f:
            return f.read()
    return (
        "# robots.txt for https://oeis.drgoulu.com\n"
        "# Conforms to RFC 9309\n\n"
        "User-agent: *\n"
        "Allow: /\n"
        "Allow: /hints.html\n"
        "Disallow: /api/\n"
        "Disallow: /search\n"
        "Disallow: /raw/\n"
    )


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
            raise HTTPException(status_code=404, detail="Sequence not found.")
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
            return {"error": f"Sequence {sid} not found in Python module."}
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

        return {"error": "Parameter itemf_n or containf_x required."}
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
        "message": f"Sequence {sid} updated and Markdown file regenerated successfully.",
        "markdown_file": file_path
    }


@app.post("/api/create_and_serve/{seq_id}")
def api_create_and_serve(seq_id: str):
    """
    Génère le fichier Markdown pour une séquence inexistante en extrayant les données depuis OEIS,
    puis synchronise la base de données SQLite locale.
    """
    sid = seq_id.strip().upper()
    if sid.startswith('A') and sid[1:].isdigit():
        num = sid[1:]
        sid = f"A{num.zfill(6)}"
    else:
        raise HTTPException(status_code=400, detail="Invalid OEIS identifier format (expected: Axxxxxx)")

    conn = get_db()
    try:
        file_path, _ = generate_sequence_markdown(sid, conn=conn, output_dir=SEQUENCES_DIR)
        return {"status": "success", "sequence_id": sid, "markdown_file": file_path}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Sequence {sid} not found on OEIS.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating {sid}: {str(e)}")
    finally:
        conn.close()


@app.get("/{seq_id}")
def sequence_page(seq_id: str, request: Request = None, direct: bool = False):
    """
    Page d'affichage d'une séquence :
    1. Si le fichier Markdown existe, le lit et le convertit en HTML à la volée.
    2. S'il n'existe pas, affiche immédiatement 'Please wait...' pendant l'extraction
       et la génération du fichier .md, puis recharge et sert la page HTML.
    """
    # Nettoyage de l'ID (ex: A45 -> A000045)
    sid = seq_id.strip().upper()
    if sid.startswith('A') and sid[1:].isdigit():
        num = sid[1:]
        sid = f"A{num.zfill(6)}"
    else:
        raise HTTPException(status_code=404, detail="Invalid OEIS identifier format (expected: Axxxxxx)")

    wants_markdown = request and "text/markdown" in request.headers.get("accept", "").lower()

    md_file_path = os.path.join(SEQUENCES_DIR, f"{sid}.md")

    # Si le fichier Markdown n'existe pas encore
    if not os.path.exists(md_file_path):
        if direct or wants_markdown:
            # Mode direct sans écran d'attente (pour tests, agents ou fallback sans JS)
            conn = get_db()
            try:
                generate_sequence_markdown(sid, conn=conn, output_dir=SEQUENCES_DIR)
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Sequence {sid} not found on OEIS.")
            finally:
                conn.close()
        else:
            # Affichage de l'écran "Please wait" pendant le processus de création
            loading_html = f"""
            <noscript>
                <meta http-equiv="refresh" content="0;url=/{sid}?direct=true">
            </noscript>
            <div class="please-wait-card">
                <div class="spinner"></div>
                <h2 class="please-wait-title">Sequence {sid}</h2>
                <p class="please-wait-subtitle">
                    This sequence entry is not yet stored locally.<br>
                    Fetching official data from OEIS in progress...
                </p>
                <div class="please-wait-badge" id="wait-badge">⏳ Please wait...</div>
                <div id="error-container" style="display: none; margin-top: 1.5rem;"></div>
            </div>
            <script>
            (async function() {{
                try {{
                    const response = await fetch('/api/create_and_serve/{sid}', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }}
                    }});
                    if (response.ok) {{
                        window.location.replace('/{sid}');
                    }} else {{
                        const err = await response.json().catch(() => ({{}}));
                        const badge = document.getElementById('wait-badge');
                        if (badge) badge.style.display = 'none';
                        const spinner = document.querySelector('.spinner');
                        if (spinner) spinner.style.display = 'none';
                        const errDiv = document.getElementById('error-container');
                        if (errDiv) {{
                            errDiv.style.display = 'block';
                            errDiv.innerHTML = '<div style="color: #dc2626; font-weight: 600; margin-bottom: 1rem;">❌ ' + 
                                (err.detail || "Unable to retrieve sequence from OEIS.") + 
                                '</div><a href="/" style="display: inline-block; padding: 0.4rem 1.2rem; background: var(--oeis-blue); color: white; border-radius: 6px; text-decoration: none;">Back to home</a>';
                        }}
                    }}
                }} catch (e) {{
                    const badge = document.getElementById('wait-badge');
                    if (badge) badge.style.display = 'none';
                    const spinner = document.querySelector('.spinner');
                    if (spinner) spinner.style.display = 'none';
                    const errDiv = document.getElementById('error-container');
                    if (errDiv) {{
                        errDiv.style.display = 'block';
                        errDiv.innerHTML = '<div style="color: #dc2626; font-weight: 600;">❌ Network error during generation.</div>';
                    }}
                }}
            }})();
            </script>
            """
            return render_page(f"Sequence {sid} - Please wait", loading_html, seq_id=sid)

    # Lecture du fichier Markdown
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    if wants_markdown:
        tokens = len(md_text.split()) * 4 // 3
        return Response(
            content=md_text,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Type": "text/markdown; charset=utf-8",
                "x-markdown-tokens": str(tokens),
            }
        )

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
            <a href="/raw/{sid}.md" target="_blank">&#128196; View raw Markdown</a>
            <a href="https://oeis.org/{sid}" target="_blank" rel="noopener">&#127760; Official OEIS.org page</a>
            <a href="/edit/{sid}">&#9998; Submit implementation</a>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
            Source: <code>seq/{sid}.md</code>
        </div>
    </div>
    """

    interactive_box = f"""
    <div class="interactive-box">
        <h4>&#9881; Live Sequence Evaluation Console</h4>
        <p style="font-size: 0.85rem; color: var(--text-muted);">
            Test the <code>itemf(n)</code> and <code>containf(x)</code> functions directly in this page:
        </p>
        <div class="test-controls">
            <div class="test-input-group">
                <label for="itemf-n">Compute term <em>n</em>:</label>
                <input type="number" id="itemf-n" value="10" min="0" max="1000" class="test-input">
                <button type="button" onclick="testItemf('{sid}')" class="test-button">Compute itemf(n)</button>
            </div>
            <div class="test-input-group">
                <label for="containf-x">Test value <em>x</em>:</label>
                <input type="number" id="containf-x" value="55" class="test-input">
                <button type="button" onclick="testContainf('{sid}')" class="test-button">Test containf(x)</button>
            </div>
        </div>
        <div id="test-result" class="test-output">Result ready. Click a button above.</div>
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

