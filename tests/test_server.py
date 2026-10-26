"""
Tests pour les routes de src/server.py :
- Affichage de l'écran 'Please wait...' pour une séquence inexistante
- Génération à la volée via /api/create_and_serve/{seq_id}
- Affichage de la séquence une fois le fichier Markdown créé
"""

import os
import json
import unittest
from starlette.requests import Request

from src.server import (
    sequence_page,
    api_create_and_serve,
    robots_txt,
    sitemap_xml,
    api_catalog,
    mcp_server_card,
    agent_skills_index,
    ai_catalog,
    agent_card,
    auth_md,
    index_page,
    hints_page,
    search_page,
    SEQUENCES_DIR,
)
from src.i18n import resolve_language, t, build_language_switch_url


class TestServerSequenceFlow(unittest.TestCase):

    def test_robots_txt(self):
        res = robots_txt()
        content = res if isinstance(res, str) else res.body.decode('utf-8')
        self.assertIn("User-agent:", content)
        self.assertIn("Allow: /", content)
        self.assertIn("Disallow: /api/", content)
        self.assertIn("Disallow: /search", content)
        self.assertIn("Content-Signal: ai-train=no, search=yes, ai-input=yes", content)
        self.assertIn("Sitemap: https://oeis.drgoulu.com/sitemap.xml", content)
        self.assertIn("User-agent: OAI-SearchBot", content)
        self.assertIn("User-agent: GPTBot", content)

    def test_sitemap_xml(self):
        res = sitemap_xml()
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/xml", res.media_type)
        content = res.body.decode('utf-8')
        self.assertIn('<?xml version="1.0" encoding="UTF-8"?>', content)
        self.assertIn('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">', content)
        self.assertIn('<loc>https://oeis.drgoulu.com/</loc>', content)

    def test_api_catalog(self):
        res = api_catalog()
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/linkset+json", res.media_type)
        data = json.loads(res.body.decode('utf-8'))
        self.assertIn("linkset", data)
        self.assertEqual(data["linkset"][0]["anchor"], "https://oeis.drgoulu.com/")

    def test_mcp_server_card(self):
        res = mcp_server_card()
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.body.decode('utf-8'))
        self.assertIn("serverInfo", data)
        self.assertEqual(data["serverInfo"]["name"], "oeis-mcp")

    def test_agent_skills_index(self):
        res = agent_skills_index()
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.body.decode('utf-8'))
        self.assertIn("skills", data)
        self.assertTrue(len(data["skills"]) > 0)

    def test_ai_catalog(self):
        res = ai_catalog()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), "*")
        data = json.loads(res.body.decode('utf-8'))
        self.assertEqual(data.get("specVersion"), "1.0")
        self.assertIn("entries", data)

    def test_agent_card(self):
        res = agent_card()
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.body.decode('utf-8'))
        self.assertEqual(data["name"], "OEIS Python Agent")
        self.assertIn("supportedInterfaces", data)

    def test_auth_md(self):
        res = auth_md()
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/markdown", res.media_type)
        content = res.body.decode('utf-8')
        self.assertIn("# auth.md", content)

    def test_link_headers_and_markdown_negotiation(self):
        # 1. HTML request on homepage has Link headers
        scope_html = {'type': 'http', 'headers': [(b'accept', b'text/html')]}
        res_html = index_page(Request(scope_html))
        self.assertEqual(res_html.status_code, 200)
        self.assertIn('rel="api-catalog"', res_html.headers.get("link", ""))
        self.assertIn('rel="service-doc"', res_html.headers.get("link", ""))

        # 2. Markdown negotiation on homepage
        scope_md = {'type': 'http', 'headers': [(b'accept', b'text/markdown')]}
        res_md = index_page(Request(scope_md))
        self.assertEqual(res_md.status_code, 200)
        self.assertIn("text/markdown", res_md.headers.get("content-type", ""))
        self.assertIn("x-markdown-tokens", res_md.headers)

        # 3. Markdown negotiation on sequence page
        res_seq_md = sequence_page("A000045", Request(scope_md))
        self.assertEqual(res_seq_md.status_code, 200)
        self.assertIn("text/markdown", res_seq_md.headers.get("content-type", ""))
        self.assertIn("x-markdown-tokens", res_seq_md.headers)

    def test_homepage_sequences_table(self):
        scope_html = {'type': 'http', 'headers': [(b'accept', b'text/html')]}
        res_html = index_page(Request(scope_html))
        self.assertEqual(res_html.status_code, 200)
        content = res_html.body.decode('utf-8')
        # Check welcome text is removed
        self.assertNotIn("Welcome to the OEIS Python Clone", content)
        self.assertNotIn("Collaborative platform for infinite integer sequences integrating the Sequence class", content)
        # Check table columns are present
        self.assertIn("Number</th>", content)
        self.assertIn("Title</th>", content)
        self.assertIn("First terms</th>", content)
        self.assertIn("Code</th>", content)
        # Check sequences table presence and badges
        self.assertIn('<table class="sequences-table">', content)
        self.assertIn('<a href="/A000045">A000045</a>', content)
        self.assertIn('class="badge-code-s"', content)

    def test_on_demand_sequence_generation(self):
        sid = "A022307"
        md_file = os.path.join(SEQUENCES_DIR, f"{sid}.md")

        # 1. Nettoyage préliminaire si le fichier existe
        if os.path.exists(md_file):
            os.remove(md_file)

        self.assertFalse(os.path.exists(md_file))

        # 2. Requête sur une séquence inexistante : doit afficher l'écran 'Please wait...'
        html_waiting = sequence_page(sid)
        content_waiting = html_waiting if isinstance(html_waiting, str) else html_waiting.body.decode('utf-8')

        self.assertIn("Please wait", content_waiting)
        self.assertIn("please-wait-card", content_waiting)
        self.assertIn("spinner", content_waiting)
        self.assertIn(f"/api/create_and_serve/{sid}", content_waiting)

        # 3. Appel de l'API de génération à la volée
        res = api_create_and_serve(sid)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["sequence_id"], sid)
        self.assertTrue(os.path.exists(md_file))

        # 4. Nouvelle requête : la page Markdown est maintenant servie en HTML
        html_served = sequence_page(sid)
        content_served = html_served if isinstance(html_served, str) else html_served.body.decode('utf-8')

        self.assertNotIn('<div class="please-wait-card">', content_served)
        self.assertIn("DATA", content_served)
        self.assertIn(sid, content_served)

    def test_i18n_language_resolution_and_rendering(self):
        # 1. URL parameter override for Spanish (full name and 2-letter code)
        res_es = index_page(language="spanish")
        content_es = res_es.body.decode('utf-8')
        self.assertIn('lang="es"', content_es)
        self.assertIn("Número</th>", content_es)
        self.assertIn("Título</th>", content_es)
        self.assertIn("Primeros términos</th>", content_es)
        self.assertIn("Código</th>", content_es)
        self.assertIn("Buscar", content_es)

        # 2. URL parameter override for French
        res_fr = index_page(language="french")
        content_fr = res_fr.body.decode('utf-8')
        self.assertIn('lang="fr"', content_fr)
        self.assertIn("Numéro</th>", content_fr)
        self.assertIn("Titre</th>", content_fr)
        self.assertIn("Premiers termes</th>", content_fr)
        self.assertIn("Code</th>", content_fr)
        self.assertIn("Rechercher", content_fr)

        # 3. URL parameter override for German
        res_de = index_page(language="german")
        content_de = res_de.body.decode('utf-8')
        self.assertIn('lang="de"', content_de)
        self.assertIn("Nummer</th>", content_de)
        self.assertIn("Titel</th>", content_de)
        self.assertIn("Erste Terme</th>", content_de)
        self.assertIn("Code</th>", content_de)
        self.assertIn("Suchen", content_de)

        # 4. Accept-Language header fallback when no URL parameter is provided
        scope_browser_fr = {
            'type': 'http',
            'path': '/',
            'headers': [(b'accept-language', b'fr-CH,fr;q=0.9,en;q=0.8')],
            'query_string': b'',
        }
        res_browser_fr = index_page(Request(scope_browser_fr))
        content_browser_fr = res_browser_fr.body.decode('utf-8')
        self.assertIn('lang="fr"', content_browser_fr)
        self.assertIn("Numéro</th>", content_browser_fr)

        scope_browser_de = {
            'type': 'http',
            'path': '/',
            'headers': [(b'accept-language', b'de-DE,de;q=0.9,en;q=0.8')],
            'query_string': b'',
        }
        res_browser_de = index_page(Request(scope_browser_de))
        content_browser_de = res_browser_de.body.decode('utf-8')
        self.assertIn('lang="de"', content_browser_de)
        self.assertIn("Nummer</th>", content_browser_de)

        # 5. Fallback to English when unsupported language or no header
        scope_browser_ja = {
            'type': 'http',
            'path': '/',
            'headers': [(b'accept-language', b'ja-JP,ja;q=0.9')],
            'query_string': b'',
        }
        res_browser_ja = index_page(Request(scope_browser_ja))
        content_browser_ja = res_browser_ja.body.decode('utf-8')
        self.assertIn('lang="en"', content_browser_ja)
        self.assertIn("Number</th>", content_browser_ja)

        # 6. Hints page i18n
        res_hints_fr = hints_page(language="french")
        content_hints_fr = res_hints_fr.body.decode('utf-8')
        self.assertIn("Aide et syntaxe de recherche", content_hints_fr)

        res_hints_es = hints_page(language="spanish")
        content_hints_es = res_hints_es.body.decode('utf-8')
        self.assertIn("Consejos y sintaxis de búsqueda", content_hints_es)

        res_hints_de = hints_page(language="german")
        content_hints_de = res_hints_de.body.decode('utf-8')
        self.assertIn("Suchhinweise und Syntax", content_hints_de)

        # 7. Language switch URL builder
        self.assertEqual(
            build_language_switch_url("https://oeis.org/", "spanish"),
            "https://oeis.org/?language=spanish"
        )
        self.assertEqual(
            build_language_switch_url("https://oeis.org/?language=spanish", "english"),
            "https://oeis.org/?language=english"
        )


if __name__ == "__main__":
    unittest.main()

