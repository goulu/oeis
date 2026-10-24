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
    index_page,
    SEQUENCES_DIR,
)


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


if __name__ == "__main__":
    unittest.main()

