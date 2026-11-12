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
    OctaveOnlineEnvironment,
    matlab_env,
    python_env,
    sequence_env,
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
        self.assertTrue(res_matlab["url"].startswith("https://octave-online.net/?code="))

    def test_matlab_env_octave_online(self):
        self.assertIsInstance(matlab_env, OctaveOnlineEnvironment)
        self.assertEqual(matlab_env.name, "matlab")
        self.assertEqual(matlab_env.base_url, "https://octave-online.net")

        res = matlab_env.execute("disp(fibonacci(10))")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["mode"], "external")
        self.assertEqual(res["language"], "matlab")
        self.assertIn("https://octave-online.net/?code=disp", res["url"])

        derived = matlab_env.derive("matlab-symbolic", imports=["symbolic"])
        res_derived = derived.execute("syms x; diff(x^2)")
        self.assertIn("pkg+load+symbolic", res_derived["url"])

        res_empty = matlab_env.execute("   ")
        self.assertEqual(res_empty["status"], "error")

    def test_render_matlab_code_block_has_runner(self):
        md = "```matlab\ndisp(1 + 1)\n```"
        rendered = render_sequence_markdown_to_html(md, "A000045", "en")
        self.assertIn('data-lang="matlab"', rendered)
        self.assertIn('class="btn-run"', rendered)


if __name__ == "__main__":
    unittest.main()
