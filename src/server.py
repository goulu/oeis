"""
Serveur Web pour le clone OEIS.
- Convertit à la volée les fichiers Markdown des séquences en HTML.
- Génère automatiquement le fichier Markdown après modification de la base.
- Fournit une barre de recherche inspirée de hints.html et une console de test interactive.
"""

import os
import re
import json
import sqlite3
import datetime
import html
import markdown
from fastapi import FastAPI, HTTPException, Request, Form, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
import subprocess
import pygments
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from markdown_generator import generate_sequence_markdown, DEFAULT_DB_PATH, DEFAULT_OUTPUT_DIR, ensure_db_schema
from search_engine import OEISSearchEngine
from i18n import (
    resolve_language,
    t,
    LANGUAGE_NAMES,
    LANGUAGE_URL_PARAM,
    build_language_switch_url,
)

app = FastAPI(title="OEIS Python Clone", version="1.0.0")

ROOT_DIR = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'oeis.db')) or os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'seq')) else SRC_DIR
DB_PATH = os.path.join(ROOT_DIR, 'oeis.db')
SEQUENCES_DIR = os.path.join(ROOT_DIR, 'seq')
search_engine = OEISSearchEngine(DB_PATH)




def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_db_schema(conn)
    return conn


def render_page(title: str, content_html: str, query: str = "", seq_id: str = "", lang: str = "en", request: Request = None) -> str:
    """Génère le gabarit HTML complet avec le design moderne inspiré d'OEIS et support multilingue."""
    if request is not None and lang == "en":
        lang = resolve_language(request)

    try:
        current_url = str(request.url) if request else "/"
    except Exception:
        current_url = "/"
    lang_links = []
    for code in ("en", "fr", "de", "es"):
        is_active = " active" if code == lang else ""
        href = build_language_switch_url(current_url, code)
        lang_links.append(f'<a href="{href}" class="lang-link{is_active}">{LANGUAGE_NAMES[code]}</a>')
    lang_selector_html = f'<div class="lang-selector">{"".join(lang_links)}</div>'

    lang_hidden = f'<input type="hidden" name="language" value="{LANGUAGE_URL_PARAM.get(lang, lang)}">' if lang != "en" else ""
    hints_url = f"/hints.html?language={LANGUAGE_URL_PARAM.get(lang, lang)}" if lang != "en" else "/hints.html"
    home_url = f"/?language={LANGUAGE_URL_PARAM.get(lang, lang)}" if lang != "en" else "/"

    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {t("app_title", lang)}</title>
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

        /* Language Selector in Header */
        .lang-selector {{
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.8rem;
            flex-wrap: wrap;
        }}

        .lang-link {{
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid #bfdbfe;
            transition: all 0.15s ease;
        }}

        .lang-link:hover {{
            background: #dbeafe;
            color: var(--oeis-blue);
            text-decoration: none;
        }}

        .lang-link.active {{
            background: var(--oeis-blue);
            color: white;
            border-color: var(--oeis-blue);
            font-weight: 700;
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

        .search-hints-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.35rem;
            font-size: 0.78rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .search-hints {{
            font-size: 0.78rem;
            color: var(--text-muted);
        }}

        .search-hints-action a {{
            color: var(--primary);
            text-decoration: none;
            font-size: 0.82rem;
            font-weight: 600;
            white-space: nowrap;
        }}

        .search-hints-action a:hover {{
            text-decoration: underline;
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
            padding: 0.85rem 1rem;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: anywhere;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.88rem;
            line-height: 1.45;
            margin: 0.8rem 0;
        }}

        .markdown-body pre code {{
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: anywhere;
            font-family: inherit;
        }}

        .markdown-body code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: var(--code-bg);
            padding: 0.15rem 0.3rem;
            border-radius: 4px;
            font-size: 0.88rem;
        }}

        /* Pygments Syntax Highlighting pour Mathematica, Maple, Python, PARI, etc. */
        .codehilite {{
            background: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            margin: 0.8rem 0;
            overflow-x: auto;
        }}
        .codehilite pre {{
            background: transparent !important;
            padding: 0.85rem 1rem !important;
            margin: 0 !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
            font-size: 0.88rem !important;
            line-height: 1.45 !important;
            color: #1e293b;
        }}
        .codehilite .k, .codehilite .kd, .codehilite .kn, .codehilite .kp, .codehilite .kr {{ color: #0284c7; font-weight: 700; }}
        .codehilite .kt {{ color: #0284c7; }}
        .codehilite .na, .codehilite .nf, .codehilite .fm {{ color: #2563eb; font-weight: 700; }}
        .codehilite .nc, .codehilite .nn {{ color: #0f766e; font-weight: 700; }}
        .codehilite .s, .codehilite .sa, .codehilite .sb, .codehilite .sc, .codehilite .dl,
        .codehilite .sd, .codehilite .s2, .codehilite .se, .codehilite .sh, .codehilite .si,
        .codehilite .sx, .codehilite .sr, .codehilite .s1, .codehilite .ss {{ color: #15803d; }}
        .codehilite .m, .codehilite .mb, .codehilite .mf, .codehilite .mh, .codehilite .mi,
        .codehilite .il, .codehilite .mo {{ color: #b45309; }}
        .codehilite .o, .codehilite .ow {{ color: #334155; font-weight: 600; }}
        .codehilite .c, .codehilite .ch, .codehilite .cm, .codehilite .cp, .codehilite .cpf,
        .codehilite .c1, .codehilite .cs {{ color: #64748b; font-style: italic; }}
        .codehilite .nb, .codehilite .bp {{ color: #0369a1; }}
        .codehilite .p {{ color: #475569; }}

        /* Code Runner Buttons & Output */
        .code-runner {{
            margin: -0.4rem 0 1.2rem 0;
        }}
        .runner-controls {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-top: 0.35rem;
            margin-bottom: 0.4rem;
        }}
        .btn-run {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: #0284c7;
            color: #ffffff;
            border: none;
            padding: 0.35rem 0.9rem;
            font-size: 0.82rem;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.15s, transform 0.1s;
        }}
        .btn-run:hover {{
            background: #0369a1;
        }}
        .btn-run:active {{
            transform: scale(0.98);
        }}
        .btn-run:disabled {{
            background: #94a3b8;
            cursor: not-allowed;
        }}
        .run-status {{
            font-size: 0.82rem;
            color: var(--text-muted);
        }}
        .run-output {{
            background: #0f172a;
            color: #f8fafc;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.85rem;
            line-height: 1.4;
            white-space: pre-wrap;
            word-break: break-word;
            margin-top: 0.5rem;
            border-left: 3px solid #0284c7;
            max-height: 320px;
            overflow-y: auto;
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

        /* Sequences Table (Homepage) */
        .sequences-table-container {{
            overflow-x: auto;
            margin-top: 1rem;
        }}

        .sequences-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        .sequences-table th, .sequences-table td {{
            border: 1px solid var(--border-color);
            padding: 0.55rem 0.75rem;
            text-align: left;
            vertical-align: top;
        }}

        .sequences-table th {{
            background: #f1f5f9;
            color: var(--oeis-blue);
            font-weight: 600;
            white-space: nowrap;
        }}

        .sequences-table tr:hover {{
            background-color: #f8fafc;
        }}

        .sequences-table .col-id {{
            white-space: nowrap;
            font-weight: 700;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}

        .sequences-table .col-id a {{
            color: var(--primary);
            text-decoration: none;
        }}

        .sequences-table .col-id a:hover {{
            text-decoration: underline;
        }}

        .sequences-table .col-title {{
            color: var(--text-main);
            line-height: 1.4;
        }}

        .sequences-table .col-terms {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.82rem;
            color: #334155;
            background: #f8fafc;
            overflow-wrap: anywhere;
            word-break: break-word;
            line-height: 1.4;
        }}

        .sequences-table .col-code {{
            text-align: center;
            font-weight: 700;
            white-space: nowrap;
            vertical-align: middle;
        }}

        .badge-code-s {{
            display: inline-block;
            background: #dbeafe;
            color: var(--oeis-blue);
            font-weight: 700;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            font-size: 0.85rem;
            border: 1px solid #bfdbfe;
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
            <a href="{home_url}" class="logo-area">
                <div class="logo-icon">OEIS</div>
                <div class="logo-text">
                    <h1>{t("app_title", lang)}</h1>
                    <span>{t("app_subtitle", lang)}</span>
                </div>
            </a>
            {lang_selector_html}
        </div>
        <div class="search-container">
            <form action="/search" method="GET" class="search-form" toolname="oeis_search" tooldescription="{t('search_tool_desc', lang)}">
                {lang_hidden}
                <input type="text" name="q" value="{query}" placeholder="{t('search_placeholder', lang)}" class="search-input" toolparamdescription="{t('search_param_desc', lang)}">
                <button type="submit" class="search-btn">{t("search_button", lang)}</button>
            </form>
            <div class="search-hints-row">
                <div class="search-hints">
                    {t("search_examples_label", lang)} <code>1, 1, 2, 3, 5, 8</code> &bull; <code>A000045</code> &bull; <code>Fibonacci</code> &bull; <code>author:Guglielmetti</code>
                </div>
                <div class="search-hints-action">
                    <a href="{hints_url}">&#128161; {t("search_hints", lang)}</a>
                </div>
            </div>
        </div>
    </header>

    <main>
        {content_html}
    </main>

    <footer>
        <p>{t("footer_text", lang)}</p>
    </footer>

    <script>
    async function testItemf(seqId) {{
        const n = document.getElementById('itemf-n').value;
        const out = document.getElementById('test-result');
        out.innerText = '{t("eval_computing", lang)}';
        try {{
            const res = await fetch(`/api/sequence/${{seqId}}/eval?itemf_n=${{n}}`);
            const data = await res.json();
            if (data.error) {{
                out.innerText = '{t("error_prefix", lang)}' + data.error;
            }} else {{
                out.innerText = `itemf(${{n}}) = ${{data.result}}`;
            }}
        }} catch(e) {{
            out.innerText = '{t("network_error", lang)}' + e.message;
        }}
    }}

    async function testContainf(seqId) {{
        const x = document.getElementById('containf-x').value;
        const out = document.getElementById('test-result');
        out.innerText = '{t("eval_testing", lang)}';
        try {{
            const res = await fetch(`/api/sequence/${{seqId}}/eval?containf_x=${{x}}`);
            const data = await res.json();
            if (data.error) {{
                out.innerText = '{t("error_prefix", lang)}' + data.error;
            }} else {{
                out.innerText = `containf(${{x}}) = ${{data.result}} (${{data.result ? '{t("belongs_to_sequence", lang)}' : '{t("not_belongs_to_sequence", lang)}'}})`;
            }}
        }} catch(e) {{
            out.innerText = '{t("network_error", lang)}' + e.message;
        }}
    }}

    async function runSnippet(btn) {{
        const runner = btn.closest('.code-runner');
        if (!runner) return;
        const code = runner.dataset.code || '';
        const output = runner.querySelector('.run-output');
        const status = runner.querySelector('.run-status');

        btn.disabled = true;
        if (status) status.innerText = '{t("run_running", lang)}';
        if (output) output.style.display = 'none';

        try {{
            const res = await fetch('/api/run_code', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{code: code}})
            }});
            const data = await res.json();
            if (output) {{
                output.style.display = 'block';
                if (data.status === 'success') {{
                    output.style.color = '#f8fafc';
                    output.innerText = data.output || '{t("run_no_output", lang)}';
                }} else {{
                    output.style.color = '#f87171';
                    output.innerText = (data.error || data.output || 'Execution error');
                }}
            }}
            if (status) status.innerText = '';
        }} catch(e) {{
            if (output) {{
                output.style.display = 'block';
                output.style.color = '#f87171';
                output.innerText = '{t("network_error", lang)}' + e.message;
            }}
            if (status) status.innerText = '';
        }} finally {{
            btn.disabled = false;
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
def index_page(request: Request = None, language: str = None):
    """Page d'accueil avec tableau des séquences, Link headers (RFC 8288), support i18n et Markdown."""
    lang = resolve_language(request, language)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            s.id, s.name, s.terms_preview, s.created_at as s_created_at,
            i.created_at as i_created_at, i.definition_code
        FROM sequences s
        LEFT JOIN implementations i ON s.current_implementation_id = i.id
    """)
    rows = cursor.fetchall()
    conn.close()

    seq_list = []
    for r in rows:
        sid = r["id"]
        dates = []
        if r["s_created_at"]:
            dates.append(str(r["s_created_at"]))
        if r["i_created_at"]:
            dates.append(str(r["i_created_at"]))
        md_file = os.path.join(SEQUENCES_DIR, f"{sid}.md")
        if os.path.exists(md_file):
            dates.append(datetime.datetime.fromtimestamp(os.path.getmtime(md_file)).strftime("%Y-%m-%d %H:%M:%S"))
        last_mod = max(dates) if dates else "1970-01-01 00:00:00"

        raw_terms = [t.strip() for t in (r["terms_preview"] or "").split(",") if t.strip()]
        first_20 = ", ".join(raw_terms[:20])
        has_code = bool(r["definition_code"] and r["definition_code"].strip())

        seq_list.append({
            "id": sid,
            "name": r["name"] or "",
            "terms": first_20,
            "code": "S" if has_code else "",
            "last_mod": last_mod,
        })

    # Ordre chronologique inverse de dernière modification (ou création)
    seq_list.sort(key=lambda x: (x["last_mod"], x["id"]), reverse=True)

    th_number = t("col_number", lang)
    th_title = t("col_title", lang)
    th_terms = t("col_terms", lang)
    th_code = t("col_code", lang)

    # Négociation de contenu Markdown pour agents
    if request:
        accept = request.headers.get("accept", "").lower()
        if "text/markdown" in accept:
            md_lines = [
                f"# {t('app_title', lang)}\n",
                f"{len(seq_list)} {t('sequences_count_info', lang, count=len(seq_list))}\n",
                f"| {th_number} | {th_title} | {th_terms} | {th_code} |",
                "| :--- | :--- | :--- | :---: |",
            ]
            for s in seq_list:
                md_lines.append(f"| [{s['id']}](https://oeis.drgoulu.com/{s['id']}) | {s['name']} | {s['terms']} | {s['code']} |")
            md_text = "\n".join(md_lines)
            tokens = str(len(md_text.split()) * 4 // 3)
            return Response(
                content=md_text,
                media_type="text/markdown; charset=utf-8",
                headers={"x-markdown-tokens": tokens}
            )

    badge_title = t("code_badge_title", lang)
    rows_html = []
    for s in seq_list:
        code_html = f'<span class="badge-code-s" title="{badge_title}">S</span>' if s["code"] == "S" else ""
        escaped_name = html.escape(s["name"])
        seq_href = f"/{s['id']}?language={LANGUAGE_URL_PARAM[lang]}" if lang != "en" else f"/{s['id']}"
        rows_html.append(f"""
        <tr>
            <td class="col-id"><a href="{seq_href}">{s['id']}</a></td>
            <td class="col-title">{escaped_name}</td>
            <td class="col-terms">{s['terms']}</td>
            <td class="col-code">{code_html}</td>
        </tr>""")

    home_content = f"""
    <div class="sequence-card">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
            <h2 style="color: var(--oeis-blue); font-size: 1.3rem; font-weight: 700; margin: 0;">{t("sequences_heading", lang)}</h2>
            <span style="color: var(--text-muted); font-size: 0.85rem;"><strong>{len(seq_list)}</strong> {t("sequences_count_info", lang, count=len(seq_list))}</span>
        </div>
        <div class="sequences-table-container">
            <table class="sequences-table">
                <thead>
                    <tr>
                        <th style="width: 105px;">{th_number}</th>
                        <th>{th_title}</th>
                        <th style="width: 42%;">{th_terms}</th>
                        <th style="width: 60px; text-align: center;">{th_code}</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
    </div>
    """
    headers = {
        "Link": '</.well-known/api-catalog>; rel="api-catalog", </docs>; rel="service-doc", </openapi.json>; rel="service-desc"'
    }
    response = HTMLResponse(content=render_page(t("home", lang), home_content, lang=lang, request=request), headers=headers)
    has_lang_param = False
    if request:
        try:
            has_lang_param = bool(request.query_params.get("language") or request.query_params.get("lang"))
        except Exception:
            pass
    if has_lang_param:
        response.set_cookie("language", lang, max_age=31536000, path="/")
    return response


@app.get("/hints.html", response_class=HTMLResponse)
def hints_page(request: Request = None, language: str = None):
    """Page d'aide à la recherche conforme à hints.html avec support i18n."""
    lang = resolve_language(request, language)

    hints_i18n = {
        "en": {
            "title": "Search Hints and Syntax",
            "desc": 'The search engine implements features described at <a href="https://oeis.org/hints.html" target="_blank">oeis.org/hints.html</a>.',
            "h1": "1. Number sequence search",
            "consec": "<strong>Consecutive terms</strong>: enter terms separated by commas or spaces:",
            "single_wild": "<strong>Single term wildcard (<code>_</code>)</strong>: matches any single number:",
            "subseq_wild": "<strong>Subsequence wildcard (<code>__</code>)</strong>: matches a sequence of terms:",
            "h2": "2. Field prefixes",
            "p_id": "<code>id:A000045</code>: direct sequence lookup",
            "p_auth": "<code>author:Philippe</code>: search by author name",
            "p_kw": "<code>keyword:core</code>: search by OEIS keyword",
            "h3": "3. Full-text search",
            "ft_desc": "Simple words are searched in descriptions and metadata using the FTS5 index:",
        },
        "fr": {
            "title": "Aide et syntaxe de recherche",
            "desc": 'Le moteur de recherche implémente les fonctionnalités décrites sur <a href="https://oeis.org/hints.html" target="_blank">oeis.org/hints.html</a>.',
            "h1": "1. Recherche de suites de nombres",
            "consec": "<strong>Termes consécutifs</strong> : entrez des nombres séparés par des virgules ou des espaces :",
            "single_wild": "<strong>Joker de terme unique (<code>_</code>)</strong> : correspond à n'importe quel nombre :",
            "subseq_wild": "<strong>Joker de sous-suite (<code>__</code>)</strong> : correspond à une suite de nombres :",
            "h2": "2. Préfixes de champs",
            "p_id": "<code>id:A000045</code> : accès direct à la suite",
            "p_auth": "<code>author:Philippe</code> : recherche par nom d'auteur",
            "p_kw": "<code>keyword:core</code> : recherche par mot-clé OEIS",
            "h3": "3. Recherche plein texte",
            "ft_desc": "Les mots simples sont recherchés dans les descriptions et métadonnées via l'index FTS5 :",
        },
        "de": {
            "title": "Suchhinweise und Syntax",
            "desc": 'Die Suchmaschine implementiert die auf <a href="https://oeis.org/hints.html" target="_blank">oeis.org/hints.html</a> beschriebenen Funktionen.',
            "h1": "1. Suche nach Zahlenfolgen",
            "consec": "<strong>Aufeinanderfolgende Terme</strong>: Geben Sie Zahlen getrennt durch Kommas oder Leerzeichen ein:",
            "single_wild": "<strong>Einzelner Term-Platzhalter (<code>_</code>)</strong>: entspricht einer beliebigen Zahl:",
            "subseq_wild": "<strong>Teilfolge-Platzhalter (<code>__</code>)</strong>: entspricht einer Folge von Zahlen:",
            "h2": "2. Feldpräfixe",
            "p_id": "<code>id:A000045</code>: direkter Aufruf der Folge",
            "p_auth": "<code>author:Philippe</code>: Suche nach Autorenname",
            "p_kw": "<code>keyword:core</code>: Suche nach OEIS-Schlüsselwort",
            "h3": "3. Volltextsuche",
            "ft_desc": "Einfache Wörter werden in Beschreibungen und Metadaten über den FTS5-Index durchsucht:",
        },
        "es": {
            "title": "Consejos y sintaxis de búsqueda",
            "desc": 'El motor de búsqueda implementa las funciones descritas en <a href="https://oeis.org/hints.html" target="_blank">oeis.org/hints.html</a>.',
            "h1": "1. Búsqueda de secuencias numéricas",
            "consec": "<strong>Términos consecutivos</strong>: ingrese términos separados por comas o espacios:",
            "single_wild": "<strong>Comodín de término único (<code>_</code>)</strong>: coincide con cualquier número:",
            "subseq_wild": "<strong>Comodín de subsecuencia (<code>__</code>)</strong>: coincide con una secuencia de términos:",
            "h2": "2. Prefijos de campo",
            "p_id": "<code>id:A000045</code>: búsqueda directa de secuencia",
            "p_auth": "<code>author:Philippe</code>: búsqueda por nombre de autor",
            "p_kw": "<code>keyword:core</code>: búsqueda por palabra clave de OEIS",
            "h3": "3. Búsqueda de texto completo",
            "ft_desc": "Las palabras simples se buscan en descripciones y metadatos utilizando el índice FTS5:",
        },
    }

    h = hints_i18n.get(lang, hints_i18n["en"])

    if request:
        accept = request.headers.get("accept", "").lower()
        if "text/markdown" in accept:
            md_text = (
                f"# {h['title']}\n\n"
                f"## {h['h1']}\n"
                "- Consecutive terms: `1, 1, 2, 3, 5, 8, 13`\n"
                "- Single term wildcard (`_`): `1, 2, _, 4, 5`\n"
                "- Subsequence wildcard (`__`): `1, 2, __, 8, 13`\n\n"
                f"## {h['h2']}\n"
                "- `id:A000045`: direct sequence lookup\n"
                "- `author:Philippe`: search by author name\n"
                "- `keyword:core`: search by OEIS keyword\n\n"
                f"## {h['h3']}\n"
                "- Simple words (e.g. `Fibonacci`, `prime`).\n"
            )
            tokens = str(len(md_text.split()) * 4 // 3)
            return Response(
                content=md_text,
                media_type="text/markdown; charset=utf-8",
                headers={"x-markdown-tokens": tokens}
            )

    hints_content = f"""
    <div class="sequence-card">
        <h2 style="color: var(--oeis-blue); font-size: 1.5rem; margin-bottom: 0.5rem;">{h['title']}</h2>
        <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
            {h['desc']}
        </p>

        <h3 style="color: var(--oeis-blue);">{h['h1']}</h3>
        <ul>
            <li>{h['consec']}<br>
                <code>1, 1, 2, 3, 5, 8, 13</code> or <code>10946, 17711, 28657</code>
            </li>
            <li>{h['single_wild']}<br>
                <code>1, 2, _, 4, 5</code>
            </li>
            <li>{h['subseq_wild']}<br>
                <code>1, 2, __, 8, 13</code>
            </li>
        </ul>

        <h3 style="color: var(--oeis-blue); margin-top: 1.5rem;">{h['h2']}</h3>
        <ul>
            <li>{h['p_id']}</li>
            <li>{h['p_auth']}</li>
            <li>{h['p_kw']}</li>
        </ul>

        <h3 style="color: var(--oeis-blue); margin-top: 1.5rem;">{h['h3']}</h3>
        <p>{h['ft_desc']}</p>
        <ul>
            <li><code>Fibonacci</code></li>
            <li><code>square numbers</code></li>
            <li><code>prime</code></li>
        </ul>
    </div>
    """
    response = HTMLResponse(content=render_page(t("search_hints", lang), hints_content, lang=lang, request=request))
    if request and ("language" in request.query_params or "lang" in request.query_params):
        response.set_cookie("language", lang, max_age=31536000, path="/")
    return response


@app.get("/search", response_class=HTMLResponse)
def search_page(q: str = "", language: str = None, request: Request = None):
    """Recherche conforme à hints.html via OEISSearchEngine avec support i18n."""
    lang = resolve_language(request, language)
    query = q.strip()
    if not query:
        redirect_url = f"/?language={LANGUAGE_URL_PARAM[lang]}" if lang != "en" else "/"
        return RedirectResponse(url=redirect_url)

    # Si c'est un A-number direct sans préfixe
    if query.upper().startswith('A') and query[1:].isdigit():
        target_sid = query.upper()
        redirect_url = f"/{target_sid}?language={LANGUAGE_URL_PARAM[lang]}" if lang != "en" else f"/{target_sid}"
        return RedirectResponse(url=redirect_url)

    # Exécution de la recherche via OEISSearchEngine
    results = search_engine.search(query, limit=50)

    res_html = f'<div class="sequence-card"><h2>{t("search_results_for", lang, query=query)}</h2>'
    if not results:
        res_html += f"""
        <p style="color: var(--text-muted); margin-top: 1rem;">
            {t("no_results_found", lang)}
        </p>
        """
    else:
        res_html += f'<p style="color: var(--text-muted); margin-bottom: 1.5rem;">{t("sequences_found", lang, count=len(results))}:</p>'
        for r in results:
            score = r.get('score', 0)
            score_badge = f'<span style="background: #e0f2fe; color: #0369a1; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700; font-size: 0.8rem; margin-left: 0.5rem;">+{score}</span>'
            seq_href = f"/{r['id']}?language={LANGUAGE_URL_PARAM[lang]}" if lang != "en" else f"/{r['id']}"
            res_html += f"""
            <div style="border-bottom: 1px solid var(--border-color); padding: 1rem 0;">
                <div style="font-size: 1.15rem; font-weight: 700;">
                    <a href="{seq_href}" style="color: var(--primary); text-decoration: none;">{r['id']}</a> : {html.escape(r['name'])} {score_badge}
                </div>
                <div style="font-family: monospace; font-size: 0.85rem; background: var(--code-bg); padding: 0.4rem; border-radius: 4px; margin: 0.4rem 0; overflow-wrap: anywhere; word-break: break-word;">
                    {r['terms']}
                </div>
                <div style="font-size: 0.82rem; color: #16a34a; font-weight: 500;">&check; {r.get('match_desc', '')}</div>
            </div>
            """
    res_html += '</div>'
    response = HTMLResponse(content=render_page(t("search_results_title", lang, query=query), res_html, query=query, lang=lang, request=request))
    if request and ("language" in request.query_params or "lang" in request.query_params):
        response.set_cookie("language", lang, max_age=31536000, path="/")
    return response


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
                "url": "https://oeis.drgoulu.com/",
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

## Agent Registration

This service provides public, open-access resources and automated agent discovery.

### Registration Endpoint
No prior manual registration or token provisioning is required to interact with the OEIS APIs. Automated agents may register or self-identify by including a descriptive `User-Agent` header in HTTP requests.

### Supported Methods
- Anonymous access: `identity_types_supported: ["anonymous"]`
- Method: Direct HTTP requests without bearer token.
- Discovery: Available via A2A (`/.well-known/agent-card.json`) and MCP (`/.well-known/mcp/server-card.json`).

### Credential Use
Credentials and bearer tokens are not required. Automated agents should specify appropriate content negotiation headers (`Accept: text/markdown` or `Accept: application/json`).
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


def highlight_code_block(code: str, lang: str) -> str:
    """Colorise un bloc de code avec Pygments (supporte Maple, Mathematica, Python, etc.)."""
    lang_clean = lang.strip().lower() if lang else 'text'
    alias_map = {
        'mma': 'mathematica',
        'wolfram': 'mathematica',
        'pari/gp': 'pari',
        'gp': 'pari',
        'sage': 'sagemath',
        'py': 'python',
    }
    lang_clean = alias_map.get(lang_clean, lang_clean)
    try:
        lexer = get_lexer_by_name(lang_clean, stripall=True)
    except Exception:
        lexer = TextLexer()
    formatter = HtmlFormatter(cssclass='codehilite', nowrap=False)
    return pygments.highlight(code, lexer, formatter)


def render_sequence_markdown_to_html(md_text: str, sid: str, lang: str = "en") -> str:
    """
    Convertit le Markdown d'une séquence en HTML avec :
    1. Coloration syntaxique complète via Pygments (supportant Mathematica, Maple, Python, etc.)
    2. Bouton 'Run' sous chaque bloc de code exécutable (Python)
    3. Console d'évaluation en direct 'Live Sequence Evaluation Console' placée sous le bloc Sequence(...)
    """
    has_sequence_widget = False

    def repl_fenced(m):
        nonlocal has_sequence_widget
        code_lang = m.group(1).strip().lower()
        raw_code = m.group(2)

        if code_lang == 'text':
            esc = html.escape(raw_code.strip())
            return f'\n\n<div class="codehilite"><pre><code>{esc}</code></pre></div>\n\n'

        highlighted = highlight_code_block(raw_code.strip(), code_lang)
        is_python = code_lang in ('python', 'py')
        has_sequence = is_python and 'Sequence(' in raw_code

        if has_sequence:
            has_sequence_widget = True
            console_html = f"""
<div class="interactive-box">
    <h4>&#9881; {t("eval_console_title", lang)}</h4>
    <p style="font-size: 0.85rem; color: var(--text-muted);">
        {t("eval_console_desc", lang)}
    </p>
    <div class="test-controls">
        <div class="test-input-group">
            <label for="itemf-n">{t("compute_term_n", lang)}</label>
            <input type="number" id="itemf-n" value="10" min="0" max="1000" class="test-input">
            <button type="button" onclick="testItemf('{sid}')" class="test-button">{t("compute_itemf", lang)}</button>
        </div>
        <div class="test-input-group">
            <label for="containf-x">{t("test_value_x", lang)}</label>
            <input type="number" id="containf-x" value="55" class="test-input">
            <button type="button" onclick="testContainf('{sid}')" class="test-button">{t("test_containf", lang)}</button>
        </div>
    </div>
    <div id="test-result" class="test-output">{t("eval_result_ready", lang)}</div>
</div>
"""
            return f'\n\n{highlighted}\n{console_html}\n\n'
        elif is_python:
            esc_code = html.escape(raw_code.strip())
            runner_html = f"""
<div class="code-runner" data-code="{esc_code}">
    <div class="runner-controls">
        <button type="button" class="btn-run" onclick="runSnippet(this)">
            &#9654; {t("run_button", lang)}
        </button>
        <span class="run-status"></span>
    </div>
    <div class="run-output" style="display: none;"></div>
</div>
"""
            return f'\n\n{highlighted}\n{runner_html}\n\n'
        else:
            return f'\n\n{highlighted}\n\n'

    processed_md = re.sub(r'```([a-zA-Z0-9_\-+]+)\n(.*?)```', repl_fenced, md_text, flags=re.DOTALL)
    html_body = markdown.markdown(
        processed_md,
        extensions=[
            'markdown.extensions.extra',
            'markdown.extensions.tables',
            'markdown.extensions.toc'
        ]
    )

    if not has_sequence_widget:
        try:
            import oeis
            if sid in oeis.oeis:
                fallback_console = f"""
<div class="interactive-box">
    <h4>&#9881; {t("eval_console_title", lang)}</h4>
    <p style="font-size: 0.85rem; color: var(--text-muted);">
        {t("eval_console_desc", lang)}
    </p>
    <div class="test-controls">
        <div class="test-input-group">
            <label for="itemf-n">{t("compute_term_n", lang)}</label>
            <input type="number" id="itemf-n" value="10" min="0" max="1000" class="test-input">
            <button type="button" onclick="testItemf('{sid}')" class="test-button">{t("compute_itemf", lang)}</button>
        </div>
        <div class="test-input-group">
            <label for="containf-x">{t("test_value_x", lang)}</label>
            <input type="number" id="containf-x" value="55" class="test-input">
            <button type="button" onclick="testContainf('{sid}')" class="test-button">{t("test_containf", lang)}</button>
        </div>
    </div>
    <div id="test-result" class="test-output">{t("eval_result_ready", lang)}</div>
</div>
"""
                html_body += "\n" + fallback_console
        except Exception:
            pass

    return html_body


@app.post("/api/run_code")
async def api_run_code(request: Request):
    """
    Exécute un bloc de code Python dans un sous-processus isolé
    avec timeout et capture de la sortie standard.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"status": "error", "error": "Invalid JSON body"})

    code = data.get("code", "")
    if not code or not code.strip():
        return JSONResponse(status_code=400, content={"status": "error", "error": "No code provided"})

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR + ":" + env.get("PYTHONPATH", "")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
            cwd=ROOT_DIR
        )
        output = proc.stdout or ""
        if proc.stderr:
            if output:
                output += "\n--- stderr ---\n" + proc.stderr
            else:
                output = proc.stderr
        return {
            "status": "success" if proc.returncode == 0 else "error",
            "output": output,
            "returncode": proc.returncode
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Execution timed out (5s limit)."}
    except Exception as e:
        return {"status": "error", "error": str(e)}


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
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        cursor.execute("""
            INSERT INTO implementations (
                sequence_id, user_id, status, language, author, code_date, definition_code,
                iterf_code, itemf_code, containf_code, description, test_status
            ) VALUES (?, 1, 'approved', 'python', 'User', ?, ?, ?, ?, ?, ?, 'passed')
        """, (sid, today_str, definition_code, iterf_code, itemf_code, containf_code, name or ''))
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


@app.get("/{seq_id}", response_class=HTMLResponse)
def sequence_page(seq_id: str, request: Request = None, direct: bool = False, language: str = None):
    """
    Page d'affichage d'une séquence avec support i18n :
    1. Si le fichier Markdown existe, le lit et le convertit en HTML à la volée.
    2. S'il n'existe pas, affiche immédiatement 'Please wait...' pendant l'extraction
       et la génération du fichier .md, puis recharge et sert la page HTML.
    """
    lang = resolve_language(request, language)

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
                <h2 class="please-wait-title">{t("please_wait_title", lang, sid=sid)}</h2>
                <p class="please-wait-subtitle">
                    {t("please_wait_not_stored", lang)}<br>
                    {t("please_wait_fetching", lang)}
                </p>
                <div class="please-wait-badge" id="wait-badge">{t("please_wait_badge", lang)}</div>
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
                                '</div><a href="/" style="display: inline-block; padding: 0.4rem 1.2rem; background: var(--oeis-blue); color: white; border-radius: 6px; text-decoration: none;">{t("back_to_home", lang)}</a>';
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
            return render_page(f"Sequence {sid} - Please wait", loading_html, seq_id=sid, lang=lang, request=request)

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

    # Conversion à la volée du Markdown en HTML avec coloration syntaxique et widgets d'exécution
    html_body = render_sequence_markdown_to_html(md_text, sid, lang)

    # Barre d'actions avec i18n
    actions_bar = f"""
    <div class="sequence-actions">
        <div class="btn-group">
            <a href="/raw/{sid}.md" target="_blank">&#128196; {t("view_raw_markdown", lang)}</a>
            <a href="https://oeis.org/{sid}" target="_blank" rel="noopener">&#127760; {t("official_oeis_page", lang)}</a>
            <a href="/edit/{sid}">&#9998; {t("submit_implementation", lang)}</a>
        </div>
    </div>
    """

    full_content = f"""
    {actions_bar}
    <div class="sequence-card markdown-body">
        {html_body}
    </div>
    """

    response = HTMLResponse(content=render_page(f"{sid}", full_content, query=sid, seq_id=sid, lang=lang, request=request))
    if request and ("language" in request.query_params or "lang" in request.query_params):
        response.set_cookie("language", lang, max_age=31536000, path="/")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

