"""
Tests pour les routes de src/server.py :
- Affichage de l'écran 'Please wait...' pour une séquence inexistante
- Génération à la volée via /api/create_and_serve/{seq_id}
- Affichage de la séquence une fois le fichier Markdown créé
"""

import os
import unittest

from src.server import sequence_page, api_create_and_serve, SEQUENCES_DIR


class TestServerSequenceFlow(unittest.TestCase):

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
