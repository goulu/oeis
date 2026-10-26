"""
Tests pour markdown_generator.py : validation de l'extraction des informations
d'un fichier .seq selon https://oeis.org/eishelp1.html et conversion des liens locaux.
"""

import os
import unittest
import tempfile
import sqlite3

from src.markdown_generator import (
    normalize_seq_id,
    parse_oeis_seq,
    format_oeis_link,
    format_crossref,
    localize_sequence_links,
    format_sequence_markdown,
    generate_sequence_markdown,
    sync_sequence_to_db,
    read_oeis_seq_file,
    extract_author_and_date,
    extract_programs_from_seq_data,
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

        # Noms de séquences bare (Axxxxxx) convertis en liens locaux
        text3 = "Row sums of A030528, A006356-A006359, A000032(n)^2 and cf. A014599."
        self.assertEqual(
            localize_sequence_links(text3),
            "Row sums of [A030528](A030528), [A006356](A006356)-[A006359](A006359), [A000032](A000032)(n)^2 and cf. [A014599](A014599)."
        )

        # Les blocs de code et code inline sont préservés
        text4 = "```python\nA000045 = 1\n```\n`A000045` and A000045"
        self.assertEqual(
            localize_sequence_links(text4),
            "```python\nA000045 = 1\n```\n`A000045` and [A000045](A000045)"
        )

        # Noms d'utilisateurs _Nom_ convertis en liens wiki User:
        text5 = "From _Robert G. Wilson v_, Jan 15 2002 and _N. J. A. Sloane_"
        self.assertEqual(
            localize_sequence_links(text5),
            "From [_Robert G. Wilson v_](https://oeis.org/wiki/User:Robert_G._Wilson_v), Jan 15 2002 and [_N. J. A. Sloane_](https://oeis.org/wiki/User:N._J._A._Sloane)"
        )

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
        self.assertIn("[_Joerg Arndt_](https://oeis.org/wiki/User:Joerg_Arndt)", md)
        self.assertIn("[_N. J. A. Sloane_](https://oeis.org/wiki/User:N._J._A._Sloane)", md)
        # Vérification qu'il n'y a pas de retour à la ligne dans la section DATA
        data_block = md.split("### DATA\n```text\n")[1].split("\n```")[0]
        self.assertNotIn("\n", data_block)
        self.assertTrue(data_block.startswith("1, 1, 1, 1, 2, 2"))

    def test_programs_formatting(self):
        seq_sample = """%I A000045
%S A000045 0, 1, 1, 2, 3
%N A000045 Fibonacci
%p A000045 A000045 := proc(n) combinat[fibonacci](n); end;
%p A000045 # Alternative:
%p A000045 ZL := proc(n) ... end;
%t A000045 Table[Fibonacci[k], {k, 0, 50}]
%t A000045 (* Alternative: *)
%t A000045 LinearRecurrence[{1, 1}, {0, 1}, 40]
%o A000045 (Python)
%o A000045 A000045 = lambda n: ...
%o A000045 (SageMath) # Demonstration program from _Jaap Spies_:
%o A000045 a = sloane.A000045; # choose sequence
%o A000045 print(a)            # This returns the name of the sequence.
%o A000045 print(a(38))        # This returns the 38th term of the sequence.
%o A000045 print(a.list(39))   # This returns a list of the first 39 terms.
"""
        data = parse_oeis_seq(seq_sample, "A000045")
        md = format_sequence_markdown(data)

        # 1. Vérification que les alternatives Maple et Mathematica créent des blocs séparés
        self.assertIn("```maple\nA000045 := proc(n) combinat[fibonacci](n); end;\n```", md)
        self.assertIn("```maple\n# Alternative:\nZL := proc(n) ... end;\n```", md)
        self.assertIn("```mathematica\nTable[Fibonacci[k], {k, 0, 50}]\n```", md)
        self.assertIn("```mathematica\n(* Alternative: *)\nLinearRecurrence[{1, 1}, {0, 1}, 40]\n```", md)

        # 2. Vérification que (Python) seul n'émet pas un bloc vide mais intègre la ligne suivante
        self.assertIn("- **(Python)** :\n```python\nA000045 = lambda n: ...\n```", md)
        self.assertNotIn("```python\n\n```", md)

        # 3. Vérification du groupement des lignes du programme SageMath dans un seul bloc
        expected_sage = (
            "- **(SageMath)** :\n"
            "```sagemath\n"
            "# Demonstration program from _Jaap Spies_:\n"
            "a = sloane.A000045; # choose sequence\n"
            "print(a)            # This returns the name of the sequence.\n"
            "print(a(38))        # This returns the 38th term of the sequence.\n"
            "print(a.list(39))   # This returns a list of the first 39 terms.\n"
            "```"
        )
        self.assertIn(expected_sage, md)

    def test_generate_sequence_markdown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path, content = generate_sequence_markdown("A000003", output_dir=tmp_dir)
            self.assertTrue(os.path.exists(file_path))
            self.assertIn("# A000003", content)
            self.assertIn("Number of classes", content)
            self.assertIn("[A014599](A014599)", content)

    def test_sync_data_to_db(self):
        schema = """
        CREATE TABLE sequences (id VARCHAR(7) PRIMARY KEY, name TEXT, offset_val INTEGER, terms_preview TEXT, keywords TEXT, author TEXT);
        CREATE TABLE seq (id INTEGER NOT NULL, i INTEGER NOT NULL, n INTEGER NOT NULL, repeat INTEGER NOT NULL, PRIMARY KEY(id, i, n));
        CREATE TABLE ngram_index (ngram_hash INTEGER NOT NULL, sequence_id VARCHAR(7) NOT NULL, offset_pos INTEGER NOT NULL, PRIMARY KEY(ngram_hash, sequence_id, offset_pos));
        """
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)

        # 1. Création initiale des DATA
        data = parse_oeis_seq(SAMPLE_SEQ_CONTENT, "A000003")
        sync_sequence_to_db(data, conn=conn)

        cursor = conn.cursor()
        cursor.execute("SELECT id, name, terms_preview FROM sequences WHERE id = 'A000003'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertTrue(row[2].startswith("1, 1, 1, 1, 2, 2"))

        # Vérifier table seq (RLE)
        cursor.execute("SELECT i, n, repeat FROM seq WHERE id = 3 ORDER BY i")
        seq_rows = cursor.fetchall()
        self.assertTrue(len(seq_rows) > 0)
        # Reconstruire les termes à partir de la table seq
        rebuilt = []
        for i_pos, n_val, r_val in seq_rows:
            rebuilt.extend([n_val] * r_val)
        self.assertEqual(len(rebuilt), 100)
        self.assertEqual(rebuilt[:6], [1, 1, 1, 1, 2, 2])

        # Vérifier table ngram_index
        cursor.execute("SELECT count(*) FROM ngram_index WHERE sequence_id = 'A000003'")
        ng_count = cursor.fetchone()[0]
        self.assertEqual(ng_count, 100 - 2)

        # 2. Mise à jour des DATA avec de nouvelles valeurs
        updated_content = """%I A000003
%S A000003 5, 5, 5, 10, 20
%N A000003 Updated Title
"""
        updated_data = parse_oeis_seq(updated_content, "A000003")
        sync_sequence_to_db(updated_data, conn=conn)

        cursor.execute("SELECT name, terms_preview FROM sequences WHERE id = 'A000003'")
        row2 = cursor.fetchone()
        self.assertEqual(row2[0], "Updated Title")
        self.assertEqual(row2[1], "5, 5, 5, 10, 20")

        cursor.execute("SELECT i, n, repeat FROM seq WHERE id = 3 ORDER BY i")
        new_seq_rows = cursor.fetchall()
        self.assertEqual(new_seq_rows, [(1, 5, 3), (4, 10, 1), (5, 20, 1)])

        cursor.execute("SELECT count(*) FROM ngram_index WHERE sequence_id = 'A000003'")
        self.assertEqual(cursor.fetchone()[0], 3)  # (5,5,5), (5,5,10), (5,10,20)

    def test_read_oeis_seq_file(self):
        schema = """
        CREATE TABLE sequences (id VARCHAR(7) PRIMARY KEY, name TEXT, offset_val INTEGER, terms_preview TEXT, keywords TEXT, author TEXT);
        CREATE TABLE seq (id INTEGER NOT NULL, i INTEGER NOT NULL, n INTEGER NOT NULL, repeat INTEGER NOT NULL, PRIMARY KEY(id, i, n));
        CREATE TABLE ngram_index (ngram_hash INTEGER NOT NULL, sequence_id VARCHAR(7) NOT NULL, offset_pos INTEGER NOT NULL, PRIMARY KEY(ngram_hash, sequence_id, offset_pos));
        """
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)

        with tempfile.TemporaryDirectory() as tmp_dir:
            seq_path = os.path.join(tmp_dir, "A000045.seq")
            with open(seq_path, "w", encoding="utf-8") as f:
                f.write("%I A000045\n%S A000045 0, 1, 1, 2, 3, 5, 8, 13\n%N A000045 Fibonacci numbers\n")

            data = read_oeis_seq_file(seq_path, conn=conn)
            self.assertEqual(data["id"], "A000045")
            self.assertEqual(data["terms"], ["0", "1", "1", "2", "3", "5", "8", "13"])

            cursor = conn.cursor()
            cursor.execute("SELECT terms_preview FROM sequences WHERE id = 'A000045'")
            self.assertEqual(cursor.fetchone()[0], "0, 1, 1, 2, 3, 5, 8, 13")

            cursor.execute("SELECT n, repeat FROM seq WHERE id = 45 ORDER BY i")
            self.assertEqual(cursor.fetchall(), [(0, 1), (1, 2), (2, 1), (3, 1), (5, 1), (8, 1), (13, 1)])

    def test_extract_author_and_date(self):
        auth1, d1 = extract_author_and_date("/* _Martin Ettl_, Oct 21 2012 */")
        self.assertEqual(auth1, "Martin Ettl")
        self.assertEqual(d1, "Oct 21 2012")

        auth2, d2 = extract_author_and_date("(* _Harvey P. Dale_, Aug 03 2014 *)")
        self.assertEqual(auth2, "Harvey P. Dale")
        self.assertEqual(d2, "Aug 03 2014")

        auth3, d3 = extract_author_and_date("# _N. J. A. Sloane_, 1991")
        self.assertEqual(auth3, "N. J. A. Sloane")
        self.assertEqual(d3, "1991")

    def test_extract_programs_and_db_sync(self):
        sample = """%I A000045
%S A000045 0, 1, 1, 2, 3, 5, 8
%N A000045 Fibonacci
%p A000045 with(combinat): fibonacci(n); # _Paul Barry_, Jan 15 2005
%t A000045 Table[Fibonacci[n], {n, 0, 20}] (* _Harvey P. Dale_, Aug 03 2014 *)
%o A000045 (Python)
%o A000045 def a(n):
%o A000045     return fib(n)
%o A000045 (PARI) {a(n) = fibonacci(n)}; \\\\ _Michel Marcus_, Nov 18 2013
"""
        parsed = parse_oeis_seq(sample, "A000045")
        programs = extract_programs_from_seq_data(parsed)
        self.assertEqual(len(programs), 4)

        prog_map = {p["language"]: p for p in programs}
        self.assertIn("maple", prog_map)
        self.assertEqual(prog_map["maple"]["author"], "Paul Barry")
        self.assertEqual(prog_map["maple"]["code_date"], "Jan 15 2005")

        self.assertIn("mathematica", prog_map)
        self.assertEqual(prog_map["mathematica"]["author"], "Harvey P. Dale")
        self.assertEqual(prog_map["mathematica"]["code_date"], "Aug 03 2014")

        self.assertIn("python", prog_map)
        self.assertIn("def a(n):", prog_map["python"]["code"])

        self.assertIn("pari", prog_map)
        self.assertEqual(prog_map["pari"]["author"], "Michel Marcus")
        self.assertEqual(prog_map["pari"]["code_date"], "Nov 18 2013")

        # Sync to database and verify implementations table
        conn = sqlite3.connect(":memory:")
        sync_sequence_to_db(parsed, conn=conn)

        cursor = conn.cursor()
        cursor.execute("SELECT language, author, code_date, definition_code FROM implementations WHERE sequence_id = 'A000045' ORDER BY id")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 4)

        db_langs = [r[0] for r in rows]
        self.assertEqual(db_langs, ["maple", "mathematica", "python", "pari"])
        self.assertEqual(rows[0][1], "Paul Barry")
        self.assertEqual(rows[0][2], "Jan 15 2005")


if __name__ == "__main__":
    unittest.main()
