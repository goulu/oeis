"""
Tests pour markdown_generator.py : validation de l'extraction des informations
d'un fichier .seq selon https://oeis.org/eishelp1.html et conversion des liens locaux.
"""

import os
import unittest
import tempfile

from src.markdown_generator import (
    normalize_seq_id,
    parse_oeis_seq,
    format_oeis_link,
    format_crossref,
    localize_sequence_links,
    format_sequence_markdown,
    generate_sequence_markdown,
)

SAMPLE_SEQ_CONTENT = """%I A000003 M0196 N0073 #95 May 07 2026 17:55:20
%S A000003 1,1,1,1,2,2,1,2,2,2,3,2,2,4,2,2,4,2,3,4,4,2,3,4,2,6,3,2,6,4,3,4,4,4,
%T A000003 6,4,2,6,4,4,8,4,3,6,4,4,5,4,4,6,6,4,6,6,4,8,4,2,9,4,6,8,4,4,8,8,3,8,
%U A000003 8,4,7,4,4,10,6,6,8,4,5,8,6,4,9,8,4,10,6,4,12,8,6,6,4,8,8,8,4,8,6,4
%N A000003 Number of classes of primitive positive definite binary quadratic forms of discriminant D = -4n; or equivalently the class number of the quadratic order of discriminant D = -4n.
%C A000003 Equivalently, number of reduced primitive positive definite binary quadratic forms of determinant n.
%C A000003 From _Joerg Arndt_, Sep 02 2008: (Start)
%C A000003 It seems that 2*a(n) gives the degree of the minimal polynomial of (k_n)^2.
%C A000003 Also, when setting K3(x)... (End)
%D A000003 D. A. Buell, Binary Quadratic Forms. Springer-Verlag, NY, 1989.
%H A000003 N. J. A. Sloane, <a href="/A000003/b000003.txt">Table of n, a(n) for n = 1..20000</a>
%H A000003 Conway, <a href="http://example.com/splag.html">Sphere Packings</a>
%t A000003 a[1] = 1; a[n_] := 2;
%o A000003 (Magma) O1 := MaximalOrder(QuadraticField(D));
%o A000003 (PARI) {a(n) = qfbclassno(-4*n)};
%Y A000003 See A014599 for discriminant -(4n-1).
%Y A000003 A bisection of A014600.
%K A000003 nonn,nice,easy
%O A000003 1,5
%A A000003 _N. J. A. Sloane_
"""


class TestMarkdownGenerator(unittest.TestCase):

    def test_normalize_seq_id(self):
        self.assertEqual(normalize_seq_id("3"), "A000003")
        self.assertEqual(normalize_seq_id("a3"), "A000003")
        self.assertEqual(normalize_seq_id("A000003"), "A000003")
        self.assertEqual(normalize_seq_id("A45"), "A000045")

    def test_parse_oeis_seq(self):
        data = parse_oeis_seq(SAMPLE_SEQ_CONTENT, "A000003")
        self.assertEqual(data["id"], "A000003")
        self.assertIn("Number of classes", data["name"])
        self.assertEqual(data["offset_str"], "1,5")
        self.assertEqual(data["offset_val"], 1)
        self.assertEqual(data["keywords"], "nonn,nice,easy")
        self.assertEqual(data["author"], "_N. J. A. Sloane_")
        self.assertEqual(len(data["terms"]), 100)
        self.assertEqual(data["terms"][:4], ["1", "1", "1", "1"])
        self.assertEqual(len(data["comments"]), 4)
        self.assertEqual(len(data["references"]), 1)
        self.assertEqual(len(data["links"]), 2)
        self.assertEqual(len(data["crossrefs"]), 2)
        self.assertEqual(len(data["programs"]["other"]), 2)
        self.assertEqual(len(data["programs"]["mathematica"]), 1)

    def test_format_oeis_link(self):
        # Lien vers document sous-page : reste en URL absolue OEIS
        rel_link = 'Author, <a href="/A000003/b000003.txt">B-file</a>'
        formatted = format_oeis_link(rel_link)
        self.assertEqual(formatted, "Author, [B-file](https://oeis.org/A000003/b000003.txt)")

        # Lien externe
        ext_link = 'Author, <a href="http://example.com/test.html">External</a>'
        formatted_ext = format_oeis_link(ext_link)
        self.assertEqual(formatted_ext, "Author, [External](http://example.com/test.html)")

        # Lien vers une autre séquence seule : devient un lien local
        seq_link = 'Author, <a href="/A001622">Golden Ratio</a>'
        formatted_seq = format_oeis_link(seq_link)
        self.assertEqual(formatted_seq, "Author, [Golden Ratio](A001622)")

    def test_format_crossref(self):
        text = "See A014599 for details and cf. A014600."
        formatted = format_crossref(text)
        self.assertIn("[A014599](A014599)", formatted)
        self.assertIn("[A014600](A014600)", formatted)

    def test_localize_sequence_links(self):
        # Lien vers autre séquence devient local
        text1 = "Cf. [A001622](https://oeis.org/A001622) and [phi](http://oeis.org/A001622/)"
        self.assertEqual(localize_sequence_links(text1), "Cf. [A001622](A001622) and [phi](A001622)")

        # Liens vers documents et sous-pages restent inchangés
        text2 = "- [Table](https://oeis.org/A000045/b000045.txt) and [PDF](https://oeis.org/A000003/a000003.pdf)"
        self.assertEqual(localize_sequence_links(text2), text2)

    def test_format_sequence_markdown(self):
        data = parse_oeis_seq(SAMPLE_SEQ_CONTENT, "A000003")
        md = format_sequence_markdown(data)
        self.assertIn("# A000003", md)
        self.assertIn("### DATA", md)
        self.assertIn("### METADATA", md)
        self.assertIn("### COMMENTS", md)
        self.assertIn("### REFERENCES", md)
        self.assertIn("### LINKS", md)
        self.assertIn("### PROGRAMS", md)
        self.assertIn("### CROSSREFERENCES", md)
        self.assertIn("[A014599](A014599)", md)
        self.assertIn("[Table of n, a(n) for n = 1..20000](https://oeis.org/A000003/b000003.txt)", md)

    def test_generate_sequence_markdown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path, content = generate_sequence_markdown("A000003", output_dir=tmp_dir)
            self.assertTrue(os.path.exists(file_path))
            self.assertIn("# A000003", content)
            self.assertIn("Number of classes", content)
            self.assertIn("[A014599](A014599)", content)


if __name__ == "__main__":
    unittest.main()
