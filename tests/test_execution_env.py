"""
Tests pour ExecutionEnvironment, python_env et sequence_env.
"""

import asyncio
import json
import os
import sys
import unittest

from fastapi import Request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.execution_env import (
    ExecutionEnvironment,
    ExternalWebEnvironment,
    SageMathCellEnvironment,
    TioRunEnvironment,
    WolframAlphaEnvironment,
    gap_env,
    haskell_env,
    html_env,
    julia_env,
    macaulay2_env,
    mathematica_env,
    matlab_env,
    maxima_env,
    octave_env,
    pari_env,
    python_env,
    r_env,
    sagemath_env,
    scala_env,
    sequence_env,
    singular_env,
)
from src.server import api_run_code, render_sequence_markdown_to_html


class TestExecutionEnvironment(unittest.TestCase):
    def test_python_env_imports_math_and_itertools(self):
        self.assertEqual(python_env.name, "python")
        self.assertIn("math", python_env.imports)
        self.assertIn("itertools", python_env.imports)

        res = python_env.execute(
            "print(math.factorial(5), list(itertools.islice(itertools.count(1), 4)))"
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["returncode"], 0)
        self.assertIn("120 [1, 2, 3, 4]", res["output"])

    def test_python_env_does_not_preimport_sequence(self):
        res = python_env.execute("print(Sequence)")
        self.assertEqual(res["status"], "error")
        self.assertIn("NameError", res["output"])

    def test_sequence_env_derived_from_python_env(self):
        self.assertEqual(sequence_env.name, "sequence")
        # Hérite de python_env
        self.assertIn("math", sequence_env.imports)
        self.assertIn("itertools", sequence_env.imports)
        # Ajoute goulib.math2, goulib.itertools2, goulib.containers
        self.assertIn("goulib.math2", sequence_env.imports)
        self.assertIn("goulib.itertools2", sequence_env.imports)
        self.assertIn("goulib.containers", sequence_env.imports)

        code = (
            "s = Sequence(math2.fibonacci_gen, math2.fibonacci, math2.is_fibonacci)\n"
            "print(s[10], goulib.math2.fibonacci(10), math.gcd(12, 18), "
            "list(itertools2.take(3, itertools.count(1))))"
        )
        res = sequence_env.execute(code)
        self.assertEqual(res["status"], "success", res)
        self.assertIn("55 55 6 [1, 2, 3]", res["output"])

    def test_derive_deduplicates_and_overrides(self):
        custom = python_env.derive("custom", imports=["math", "functools"], timeout=2.0)
        self.assertEqual(custom.name, "custom")
        self.assertEqual(custom.timeout, 2.0)
        self.assertEqual(custom.imports.count("math"), 1)
        self.assertIn("functools", custom.imports)

    def test_empty_code_and_timeout(self):
        res_empty = python_env.execute("   ")
        self.assertEqual(res_empty["status"], "error")
        self.assertIn("No code provided", res_empty["error"])

        short_env = ExecutionEnvironment("short", timeout=0.2)
        res_timeout = short_env.execute("import time; time.sleep(2)")
        self.assertEqual(res_timeout["status"], "error")
        self.assertIn("timed out", res_timeout["error"])

    def test_server_api_run_code_uses_environments(self):
        async def run_payload(payload):
            async def dummy_receive():
                return {
                    "type": "http.request",
                    "body": json.dumps(payload).encode("utf-8"),
                }

            req = Request({"type": "http", "method": "POST"}, receive=dummy_receive)
            resp = await api_run_code(req)
            if isinstance(resp, dict):
                return resp
            return json.loads(resp.body.decode("utf-8"))

        res_py = asyncio.run(run_payload({"code": "print(math.isqrt(144))"}))
        self.assertEqual(res_py["status"], "success")
        self.assertIn("12", res_py["output"])

        res_seq = asyncio.run(
            run_payload({"code": "s = Sequence(None, lambda n: math2.polygonal(24, n))\nprint(s[2])"})
        )
        self.assertEqual(res_seq["status"], "success")
        self.assertIn("24", res_seq["output"])

        res_matlab = asyncio.run(
            run_payload({"code": "fibonacci(10)", "language": "matlab"})
        )
        self.assertEqual(res_matlab["status"], "success")
        self.assertEqual(res_matlab["mode"], "external")
        self.assertTrue(res_matlab["url"].startswith("https://sagecell.sagemath.org/?z="))
        self.assertIn("lang=octave", res_matlab["url"])

        res_mma = asyncio.run(
            run_payload({"code": "Table[Fibonacci[n], {n, 0, 10}]", "language": "mathematica"})
        )
        self.assertEqual(res_mma["status"], "success")
        self.assertEqual(res_mma["mode"], "remote")
        self.assertIn("55", res_mma["output"])

    def test_matlab_and_octave_sagecell(self):
        for env_obj, expected_name in ((matlab_env, "matlab"), (octave_env, "octave")):
            self.assertIsInstance(env_obj, SageMathCellEnvironment)
            self.assertEqual(env_obj.name, expected_name)
            self.assertEqual(env_obj.base_url, "https://sagecell.sagemath.org")
            res = env_obj.execute("disp(fibonacci(10))")
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["mode"], "external")
            self.assertTrue(res["url"].startswith("https://sagecell.sagemath.org/?z="))
            self.assertIn("lang=octave", res["url"])

        derived = matlab_env.derive("matlab-symbolic", imports=["symbolic"])
        res_derived = derived.execute("syms x; diff(x^2)")
        self.assertTrue(res_derived["url"].startswith("https://sagecell.sagemath.org/?z="))
        self.assertNotEqual(res_derived["url"], res["url"])

        res_empty = matlab_env.execute("   ")
        self.assertEqual(res_empty["status"], "error")

    def test_mathematica_env_tio_mathics(self):
        self.assertIsInstance(mathematica_env, ExternalWebEnvironment)
        self.assertIsInstance(mathematica_env, TioRunEnvironment)
        self.assertEqual(mathematica_env.name, "mathematica")
        self.assertEqual(mathematica_env.tio_lang, "mathics")
        self.assertEqual(mathematica_env.base_url, "https://tio.run")

        res = mathematica_env.execute("Fibonacci[10]")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["mode"], "remote")
        self.assertEqual(res["language"], "mathematica")
        self.assertEqual(res["output"], "55")

        derived = mathematica_env.derive("mathematica-pkg", imports=["Combinatorica`"])
        self.assertTrue(derived.build_eval_url("Fibonacci[10]").startswith("https://tio.run/#mathics#code="))
        self.assertNotEqual(derived.build_eval_url("Fibonacci[10]"), res["url"])

        res_empty = mathematica_env.execute("   ")
        self.assertEqual(res_empty["status"], "error")

        # Nettoyage des commentaires (* ... *) et encadrement par Print[...]
        raw = "Table[Fibonacci[k], {k, 0, 50}] (* _Mohammad K. Azarian_, Jul 11 2015 *)"
        self.assertEqual(
            mathematica_env.prepare_code(raw),
            "Print[Table[Fibonacci[k], {k, 0, 50}]]",
        )
        # Ne ré-encadre pas si Print[...] est déjà présent
        with_print = "(* comment *) Print[Fibonacci[10]]"
        self.assertEqual(
            mathematica_env.prepare_code(with_print),
            "Print[Fibonacci[10]]",
        )

        # Vérifie le format exact #mathics#code=...
        import base64
        url = mathematica_env.build_eval_url(raw)
        b64 = url.split("#code=", 1)[1].replace("@", "+")
        b64 += "=" * (-len(b64) % 4)
        decoded = base64.b64decode(b64).decode("utf-8")
        self.assertEqual(decoded, "Print[Table[Fibonacci[k], {k, 0, 50}]]")

    def test_render_matlab_code_block_has_runner(self):
        md = "```matlab\ndisp(1 + 1)\n```"
        rendered = render_sequence_markdown_to_html(md, "A000045", "en")
        self.assertIn('data-lang="matlab"', rendered)
        self.assertIn('class="btn-run"', rendered)

    def test_execute_terms_100_and_1s_timeout(self):
        code = "A000045 = Sequence(math2.fibonacci_gen, math2.fibonacci, math2.is_fibonacci)"
        res = sequence_env.execute_terms(code, max_terms=100, timeout=1.0, seq_id="A000045")
        self.assertEqual(res["status"], "success", res)
        terms = [t.strip() for t in res["output"].strip().split(",")]
        self.assertEqual(len(terms), 100)
        self.assertEqual(terms[:6], ["0", "1", "1", "2", "3", "5"])

        res_timeout = python_env.execute_terms("while True:\n    pass", max_terms=100, timeout=0.3)
        self.assertEqual(res_timeout["status"], "error")
        self.assertIn("timed out", res_timeout["error"])

    def test_sagecell_and_tio_environments(self):
        for env_obj, expected_lang in (
            (sagemath_env, "sage"),
            (maxima_env, "maxima"),
            (pari_env, "gp"),
            (gap_env, "gap"),
            (matlab_env, "octave"),
            (octave_env, "octave"),
            (r_env, "r"),
            (singular_env, "singular"),
            (macaulay2_env, "macaulay2"),
            (html_env, "html"),
        ):
            self.assertIsInstance(env_obj, SageMathCellEnvironment)
            res = env_obj.execute("1+1")
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["mode"], "external")
            self.assertIn("https://sagecell.sagemath.org/?z=", res["url"])
            self.assertIn(f"lang={expected_lang}", res["url"])

        for env_obj in (haskell_env, julia_env, scala_env):
            self.assertIsInstance(env_obj, TioRunEnvironment)
            self.assertTrue(env_obj.build_eval_url("1+1").startswith(f"https://tio.run/#{env_obj.tio_lang}#code="))


if __name__ == "__main__":
    unittest.main()
