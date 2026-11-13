"""
Environnements d'exécution locaux pour du code téléchargé (ex. extraits OEIS).
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
import zlib
from typing import Any, Iterable
from urllib.parse import urlencode

import websockets

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = (
    os.path.dirname(SRC_DIR)
    if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), "oeis.db"))
    or os.path.exists(os.path.join(os.path.dirname(SRC_DIR), "seq"))
    else SRC_DIR
)


def _format_import(spec: str) -> str:
    """Normalise une spécification d'import en instruction Python valide."""
    cleaned = spec.strip()
    if cleaned.startswith(("import ", "from ")):
        return cleaned
    return f"import {cleaned}"


class ExecutionEnvironment:
    """
    Définit un environnement d'exécution local sur le serveur pour du code téléchargé.

    Permet de pré-importer un ensemble de bibliothèques, de configurer le répertoire
    de travail, le PYTHONPATH et le délai d'expiration, et de dériver de nouveaux
    environnements par spécialisation.
    """

    def __init__(
        self,
        name: str,
        imports: Iterable[str] = (),
        *,
        timeout: float = 5.0,
        cwd: str | None = None,
        pythonpath: Iterable[str] = (),
        env_vars: dict[str, str] | None = None,
        executable: str | None = None,
    ) -> None:
        self.name = name
        self.imports: tuple[str, ...] = tuple(dict.fromkeys(s.strip() for s in imports if s and s.strip()))
        self.timeout = timeout
        self.cwd = cwd
        self.pythonpath: tuple[str, ...] = tuple(dict.fromkeys(p for p in pythonpath if p))
        self.env_vars = dict(env_vars) if env_vars else {}
        self.executable = executable or sys.executable

    def derive(
        self,
        name: str,
        imports: Iterable[str] = (),
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        pythonpath: Iterable[str] | None = None,
        env_vars: dict[str, str] | None = None,
        executable: str | None = None,
    ) -> "ExecutionEnvironment":
        """Crée une instance dérivée héritant de cet environnement et ajoutant des imports."""
        merged_imports = list(self.imports) + [s for s in imports if s and s.strip()]
        merged_pythonpath = (
            list(self.pythonpath) + list(pythonpath)
            if pythonpath is not None
            else list(self.pythonpath)
        )
        merged_env = dict(self.env_vars)
        if env_vars:
            merged_env.update(env_vars)

        return ExecutionEnvironment(
            name=name,
            imports=merged_imports,
            timeout=self.timeout if timeout is None else timeout,
            cwd=self.cwd if cwd is None else cwd,
            pythonpath=merged_pythonpath,
            env_vars=merged_env,
            executable=self.executable if executable is None else executable,
        )

    def build_prelude(self) -> str:
        """Génère le bloc d'imports Python injecté avant le code téléchargé."""
        if not self.imports:
            return ""
        return "\n".join(_format_import(spec) for spec in self.imports)

    def prepare_code(self, code: str) -> str:
        """Prépare le code complet à exécuter avec les imports de l'environnement."""
        prelude = self.build_prelude()
        if not prelude:
            return code
        return f"{prelude}\n\n{code}"

    def _build_process_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.env_vars)
        if self.pythonpath:
            existing = env.get("PYTHONPATH", "")
            parts = list(self.pythonpath)
            if existing:
                parts.append(existing)
            env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def _run_subprocess(self, full_code: str, timeout: float) -> dict[str, Any]:
        """Exécute un script Python complet dans un sous-processus isolé."""
        try:
            proc = subprocess.run(
                [self.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._build_process_env(),
                cwd=self.cwd,
            )
            output = proc.stdout or ""
            if proc.stderr:
                output = f"{output}\n--- stderr ---\n{proc.stderr}" if output else proc.stderr
            return {
                "status": "success" if proc.returncode == 0 else "error",
                "output": output,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": f"Execution timed out ({timeout}s limit).",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Exécute le code en local sur le serveur dans un sous-processus isolé."""
        if not code or not code.strip():
            return {"status": "error", "error": "No code provided"}

        effective_timeout = self.timeout if timeout is None else timeout
        return self._run_subprocess(self.prepare_code(code), effective_timeout)

    def execute_terms(
        self,
        code: str,
        *,
        max_terms: int = 100,
        timeout: float = 1.0,
        seq_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Exécute le code et évalue les `max_terms` premiers termes avec un timeout strict.
        """
        if not code or not code.strip():
            return {"status": "error", "error": "No code provided"}

        prelude = self.build_prelude()
        wrapper = (
            "import ast as _ast, io as _io, itertools as _it, re as _re, sys as _sys\n"
            f"_USER_CODE = {code!r}\n"
            f"_MAX_TERMS = {int(max_terms)}\n"
            f"_SEQ_ID = {seq_id!r}\n"
            "_ns = globals()\n"
            "_buf = _io.StringIO()\n"
            "_old_out = _sys.stdout\n"
            "_sys.stdout = _buf\n"
            "try:\n"
            "    _tree = _ast.parse(_USER_CODE, mode='exec')\n"
            "    _last_expr = None\n"
            "    if _tree.body and isinstance(_tree.body[-1], _ast.Expr):\n"
            "        _last_expr = _ast.Expression(_tree.body.pop().value)\n"
            "    _before_keys = set(_ns.keys())\n"
            "    exec(compile(_tree, '<snippet>', 'exec'), _ns)\n"
            "    _expr_val = eval(compile(_last_expr, '<snippet>', 'eval'), _ns) if _last_expr is not None else None\n"
            "finally:\n"
            "    _sys.stdout = _old_out\n"
            "_printed = _buf.getvalue().strip()\n"
            "if _printed:\n"
            "    print(_printed)\n"
            "else:\n"
            "    _terms = None\n"
            "    _seq_cls = _ns.get('Sequence')\n"
            "    if _SEQ_ID and _SEQ_ID in _ns:\n"
            "        _cand = _ns[_SEQ_ID]\n"
            "        if _seq_cls and isinstance(_cand, _seq_cls):\n"
            "            _terms = list(_it.islice(_cand, _MAX_TERMS))\n"
            "        elif callable(_cand):\n"
            "            _terms = [_cand(i) for i in range(_MAX_TERMS)]\n"
            "    if _terms is None and _seq_cls is not None:\n"
            "        for _k in reversed(list(_ns.keys())):\n"
            "            if not _k.startswith('_') and isinstance(_ns[_k], _seq_cls):\n"
            "                _terms = list(_it.islice(_ns[_k], _MAX_TERMS))\n"
            "                break\n"
            "    if _terms is None:\n"
            "        for _k in reversed(list(_ns.keys())):\n"
            "            if _k in _before_keys or _k.startswith('_'):\n"
            "                continue\n"
            "            _v = _ns[_k]\n"
            "            if callable(_v) and (_k == 'a' or _re.match(r'^A\\d+$', _k)):\n"
            "                _terms = [_v(i) for i in range(_MAX_TERMS)]\n"
            "                break\n"
            "    if _terms is None and _expr_val is not None:\n"
            "        if isinstance(_expr_val, (list, tuple)):\n"
            "            _terms = list(_expr_val[:_MAX_TERMS])\n"
            "        elif hasattr(_expr_val, '__iter__') and not isinstance(_expr_val, (str, bytes)):\n"
            "            _terms = list(_it.islice(_expr_val, _MAX_TERMS))\n"
            "        else:\n"
            "            _terms = _expr_val\n"
            "    if _terms is not None:\n"
            "        print(', '.join(str(x) for x in _terms) if isinstance(_terms, list) else repr(_terms))\n"
        )
        full_code = f"{prelude}\n\n{wrapper}" if prelude else wrapper
        return self._run_subprocess(full_code, timeout)


# Instance Python standard avec math et itertools
python_env = ExecutionEnvironment(
    name="python",
    imports=["math", "itertools"],
    timeout=5.0,
    cwd=ROOT_DIR,
    pythonpath=(SRC_DIR,),
)

# Instance dérivée pour les objets Sequence avec goulib.math2, goulib.itertools2 et goulib.containers
sequence_env = python_env.derive(
    name="sequence",
    imports=[
        "goulib.math2",
        "goulib.math2 as math2",
        "goulib.itertools2",
        "goulib.itertools2 as itertools2",
        "goulib.containers",
        "from goulib.containers import Sequence",
    ],
)


class ExternalWebEnvironment(ExecutionEnvironment):
    """
    Environnement d'exécution distant déléguant l'évaluation à un service web externe.
    """

    def __init__(
        self,
        name: str,
        imports: Iterable[str] = (),
        *,
        base_url: str,
        query_path: str = "/",
        query_param: str = "code",
        extra_params: dict[str, str] | None = None,
        language: str = "text",
        prelude_template: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(name=name, imports=imports, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.query_path = query_path if query_path.startswith("/") else f"/{query_path}"
        self.query_param = query_param
        self.extra_params = dict(extra_params) if extra_params else {}
        self.language = language
        self.prelude_template = prelude_template

    def build_prelude(self) -> str:
        """Génère les instructions d'initialisation à partir du modèle configuré."""
        if not self.imports or not self.prelude_template:
            return ""
        return "\n".join(self.prelude_template.format(pkg=pkg) for pkg in self.imports)

    def build_eval_url(self, code: str) -> str:
        """Construit l'URL d'évaluation externe avec le code préparé."""
        prepared = self.prepare_code(code)
        params = {self.query_param: prepared, **self.extra_params}
        return f"{self.base_url}{self.query_path}?{urlencode(params)}"

    def derive(
        self,
        name: str,
        imports: Iterable[str] = (),
        *,
        base_url: str | None = None,
        query_path: str | None = None,
        query_param: str | None = None,
        extra_params: dict[str, str] | None = None,
        language: str | None = None,
        prelude_template: str | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> "ExternalWebEnvironment":
        """Crée une instance dérivée héritant de cet environnement externe."""
        merged_imports = list(self.imports) + [s for s in imports if s and s.strip()]
        merged_extra = dict(self.extra_params)
        if extra_params:
            merged_extra.update(extra_params)
        return type(self)(
            name=name,
            imports=merged_imports,
            base_url=self.base_url if base_url is None else base_url,
            query_path=self.query_path if query_path is None else query_path,
            query_param=self.query_param if query_param is None else query_param,
            extra_params=merged_extra,
            language=self.language if language is None else language,
            prelude_template=self.prelude_template if prelude_template is None else prelude_template,
            timeout=self.timeout if timeout is None else timeout,
        )

    def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Prépare l'évaluation distante via l'URL du service web configuré."""
        if not code or not code.strip():
            return {"status": "error", "error": "No code provided"}

        eval_url = self.build_eval_url(code)
        return {
            "status": "success",
            "mode": "external",
            "language": self.language,
            "url": eval_url,
            "output": f"Evaluation on {self.base_url}: {eval_url}",
        }

    def execute_terms(
        self,
        code: str,
        *,
        max_terms: int = 100,
        timeout: float = 1.0,
        seq_id: str | None = None,
    ) -> dict[str, Any]:
        """Délègue l'évaluation des termes au service web externe."""
        return self.execute(code, timeout=timeout)


class OctaveOnlineEnvironment(ExternalWebEnvironment):
    """Environnement d'exécution distant pour Matlab/Octave via https://octave-online.net."""

    def __init__(
        self,
        name: str = "matlab",
        imports: Iterable[str] = (),
        *,
        base_url: str = "https://octave-online.net",
        query_path: str = "/",
        query_param: str = "code",
        language: str = "matlab",
        prelude_template: str | None = "pkg load {pkg};",
        timeout: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            imports=imports,
            base_url=base_url,
            query_path=query_path,
            query_param=query_param,
            language=language,
            prelude_template=prelude_template,
            timeout=timeout,
            **kwargs,
        )


class WolframAlphaEnvironment(ExternalWebEnvironment):
    """Environnement d'exécution distant pour Mathematica via https://www.wolframalpha.com."""

    def __init__(
        self,
        name: str = "mathematica",
        imports: Iterable[str] = (),
        *,
        base_url: str = "https://www.wolframalpha.com",
        query_path: str = "/input",
        query_param: str = "i",
        language: str = "mathematica",
        prelude_template: str | None = 'Needs["{pkg}"];',
        timeout: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            imports=imports,
            base_url=base_url,
            query_path=query_path,
            query_param=query_param,
            language=language,
            prelude_template=prelude_template,
            timeout=timeout,
            **kwargs,
        )


class SageMathCellEnvironment(ExternalWebEnvironment):
    """Environnement d'exécution distant via SageMathCell (https://sagecell.sagemath.org)."""

    def __init__(
        self,
        name: str,
        cell_lang: str,
        imports: Iterable[str] = (),
        *,
        base_url: str = "https://sagecell.sagemath.org",
        prelude_template: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(
            name=name,
            imports=imports,
            base_url=base_url,
            query_path="/",
            query_param="z",
            extra_params={"lang": cell_lang},
            language=name,
            prelude_template=prelude_template,
            timeout=timeout,
        )
        self.cell_lang = cell_lang

    def build_eval_url(self, code: str) -> str:
        """Construit l'URL SageMathCell ?z=<zlib+base64>&lang=<cell_lang>."""
        prepared = self.prepare_code(code)
        compressed = zlib.compress(prepared.encode("utf-8"))
        z_param = base64.urlsafe_b64encode(compressed).decode("ascii")
        params = {"z": z_param, **self.extra_params}
        return f"{self.base_url}{self.query_path}?{urlencode(params)}"

    def _format_kernel_code(self, code: str) -> str:
        prepared = self.prepare_code(code)
        if self.cell_lang and self.cell_lang != "sage":
            return f"%%{self.cell_lang}\n{prepared}"
        return prepared

    async def _execute_ws(self, kernel_code: str, timeout: float) -> tuple[bool, str]:
        data = urlencode({"accepted_tos": "true"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/kernel",
            data=data,
            headers={"User-Agent": "Mozilla/5.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            info = json.loads(resp.read().decode("utf-8"))

        kid = info["id"]
        ws_base = info.get("ws_url", self.base_url.replace("https://", "wss://").replace("http://", "ws://")).rstrip("/")
        ws_url = f"{ws_base}/kernel/{kid}/channels"

        msg_id = str(uuid.uuid4())
        msg = {
            "header": {
                "msg_id": msg_id,
                "username": "user",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": kernel_code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        had_error = False

        async with websockets.connect(ws_url, open_timeout=timeout) as ws:
            await ws.send(json.dumps(msg))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                m = json.loads(raw)
                if m.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                mtype = m.get("msg_type") or m.get("header", {}).get("msg_type")
                content = m.get("content", {})
                if mtype == "stream":
                    text = content.get("text", "")
                    if content.get("name") == "stderr":
                        stderr_parts.append(text)
                    else:
                        stdout_parts.append(text)
                elif mtype in ("execute_result", "display_data"):
                    plain = content.get("data", {}).get("text/plain")
                    if plain:
                        stdout_parts.append(plain + "\n")
                elif mtype == "error":
                    had_error = True
                    ename = content.get("ename", "Error")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}")
                elif mtype == "status" and content.get("execution_state") == "idle":
                    break

        output = "".join(stdout_parts).strip()
        err = "".join(stderr_parts).strip()
        if err:
            output = f"{output}\n--- stderr ---\n{err}" if output else err
        return not had_error, output

    def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Exécute le code de manière transparente sur SageMathCell via WebSocket."""
        if not code or not code.strip():
            return {"status": "error", "error": "No code provided"}

        effective_timeout = max(self.timeout, timeout or 0.0, 15.0)
        kernel_code = self._format_kernel_code(code)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                ok, output = pool.submit(
                    lambda: asyncio.run(self._execute_ws(kernel_code, effective_timeout))
                ).result(timeout=effective_timeout + 5.0)
            return {
                "status": "success" if ok else "error",
                "mode": "remote",
                "language": self.language,
                "output": output,
                "url": self.build_eval_url(code),
            }
        except Exception as e:
            return {"status": "error", "error": f"SageMathCell request failed: {e}"}

    def execute_terms(
        self,
        code: str,
        *,
        max_terms: int = 100,
        timeout: float = 1.0,
        seq_id: str | None = None,
    ) -> dict[str, Any]:
        """Exécute le code sur SageMathCell avec le timeout réseau de l'environnement."""
        return self.execute(code, timeout=self.timeout)

    def derive(
        self,
        name: str,
        imports: Iterable[str] = (),
        *,
        cell_lang: str | None = None,
        base_url: str | None = None,
        prelude_template: str | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> "SageMathCellEnvironment":
        """Crée une instance dérivée héritant de cet environnement SageMathCell."""
        merged_imports = list(self.imports) + [s for s in imports if s and s.strip()]
        return SageMathCellEnvironment(
            name=name,
            cell_lang=self.cell_lang if cell_lang is None else cell_lang,
            imports=merged_imports,
            base_url=self.base_url if base_url is None else base_url,
            prelude_template=self.prelude_template if prelude_template is None else prelude_template,
            timeout=self.timeout if timeout is None else timeout,
        )


class TioRunEnvironment(ExternalWebEnvironment):
    """Environnement d'exécution distant transparent via Try It Online (https://tio.run)."""

    DEFAULT_RUN_URL = "https://tio.run/cgi-bin/static/fb67788fd3d1ebf92e66b295525335af-run"

    def __init__(
        self,
        name: str,
        tio_lang: str,
        imports: Iterable[str] = (),
        *,
        base_url: str = "https://tio.run",
        run_url: str = DEFAULT_RUN_URL,
        prelude_template: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        super().__init__(
            name=name,
            imports=imports,
            base_url=base_url,
            query_path="/",
            query_param="code",
            language=name,
            prelude_template=prelude_template,
            timeout=timeout,
        )
        self.tio_lang = tio_lang
        self.run_url = run_url.rstrip("/")

    def build_eval_url(self, code: str) -> str:
        """Encode le langage et le code en lien stateless https://tio.run/#<lang>#code=..."""
        prepared = self.prepare_code(code)
        b64 = (
            base64.b64encode(prepared.encode("utf-8"))
            .decode("ascii")
            .replace("+", "@")
            .rstrip("=")
        )
        return f"{self.base_url}/#{self.tio_lang}#code={b64}"

    def _build_run_payload(self, prepared_code: str) -> bytes:
        code_bytes = prepared_code.encode("utf-8")
        lang_bytes = self.tio_lang.encode("utf-8")
        raw_state = (
            b"Vlang\x001\x00"
            + lang_bytes
            + b"\x00F.code.tio\x00"
            + str(len(code_bytes)).encode("ascii")
            + b"\x00"
            + code_bytes
            + b"R"
        )
        compressor = zlib.compressobj(level=9, wbits=-15)
        return compressor.compress(raw_state) + compressor.flush()

    @staticmethod
    def _parse_run_response(raw_response: bytes) -> tuple[str, str]:
        if len(raw_response) <= 10:
            return "", "Empty response from tio.run"
        text = zlib.decompress(raw_response[10:], -15).decode("utf-8", errors="replace")
        if len(text) < 16:
            return "", text
        sep = text[:16]
        parts = text[16:].split(sep)
        output = parts[0].rstrip("\n") if parts else ""
        debug_info = parts[1].strip() if len(parts) > 1 else ""
        return output, debug_info

    def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Exécute le code de manière transparente via l'API tio.run et retourne le champ Output."""
        if not code or not code.strip():
            return {"status": "error", "error": "No code provided"}

        prepared = self.prepare_code(code)
        body = self._build_run_payload(prepared)
        endpoint = f"{self.run_url}/{secrets.token_hex(16)}"
        http_timeout = self.timeout if timeout is None else max(self.timeout, timeout)

        try:
            req = urllib.request.Request(endpoint, data=body, method="POST")
            with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                raw_response = resp.read()
            output, debug_info = self._parse_run_response(raw_response)
            if not output and debug_info and "Exit code: 0" not in debug_info:
                return {"status": "error", "error": debug_info, "output": debug_info}
            return {
                "status": "success",
                "mode": "remote",
                "language": self.language,
                "output": output,
                "url": self.build_eval_url(code),
            }
        except urllib.error.URLError as e:
            return {"status": "error", "error": f"tio.run request failed: {e}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def execute_terms(
        self,
        code: str,
        *,
        max_terms: int = 100,
        timeout: float = 1.0,
        seq_id: str | None = None,
    ) -> dict[str, Any]:
        """Exécute le code sur tio.run avec le timeout réseau de l'environnement."""
        return self.execute(code, timeout=self.timeout)

    def derive(
        self,
        name: str,
        imports: Iterable[str] = (),
        *,
        tio_lang: str | None = None,
        base_url: str | None = None,
        run_url: str | None = None,
        prelude_template: str | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> "TioRunEnvironment":
        """Crée une instance dérivée héritant de cet environnement TIO."""
        merged_imports = list(self.imports) + [s for s in imports if s and s.strip()]
        return TioRunEnvironment(
            name=name,
            tio_lang=self.tio_lang if tio_lang is None else tio_lang,
            imports=merged_imports,
            base_url=self.base_url if base_url is None else base_url,
            run_url=self.run_url if run_url is None else run_url,
            prelude_template=self.prelude_template if prelude_template is None else prelude_template,
            timeout=self.timeout if timeout is None else timeout,
        )


# Environnements SageMathCell (SageMath, Maxima, PARI/GP, GAP, Matlab/Octave, R, Singular, Macaulay2, HTML)
sagemath_env = SageMathCellEnvironment(name="sagemath", cell_lang="sage")
maxima_env = SageMathCellEnvironment(name="maxima", cell_lang="maxima")
pari_env = SageMathCellEnvironment(name="pari", cell_lang="gp")
gap_env = SageMathCellEnvironment(name="gap", cell_lang="gap")
matlab_env = SageMathCellEnvironment(name="matlab", cell_lang="octave", prelude_template="pkg load {pkg};")
octave_env = matlab_env.derive("octave")
r_env = SageMathCellEnvironment(name="r", cell_lang="r")
singular_env = SageMathCellEnvironment(name="singular", cell_lang="singular")
macaulay2_env = SageMathCellEnvironment(name="macaulay2", cell_lang="macaulay2")
html_env = SageMathCellEnvironment(name="html", cell_lang="html")

def _strip_mathematica_comments(code: str) -> str:
    """Supprime les commentaires Mathematica/Mathics (* ... *), y compris imbriqués."""
    out: list[str] = []
    i = 0
    n = len(code)
    depth = 0
    in_string = False

    while i < n:
        ch = code[i]
        if depth == 0 and ch == '"':
            in_string = not in_string
            out.append(ch)
            i += 1
        elif depth == 0 and in_string and ch == "\\" and i + 1 < n:
            out.append(code[i : i + 2])
            i += 2
        elif not in_string and code.startswith("(*", i):
            depth += 1
            i += 2
        elif not in_string and depth > 0 and code.startswith("*)", i):
            depth -= 1
            i += 2
        elif depth == 0:
            out.append(ch)
            i += 1
        else:
            i += 1

    return "".join(out).strip()


class MathicsTioEnvironment(TioRunEnvironment):
    """Environnement Try It Online pour Mathics (nettoie les commentaires et encadre par Print[...])."""

    def __init__(
        self,
        name: str = "mathematica",
        tio_lang: str = "mathics",
        imports: Iterable[str] = (),
        *,
        base_url: str = "https://tio.run",
        prelude_template: str | None = 'Needs["{pkg}"];',
        timeout: float = 15.0,
    ) -> None:
        super().__init__(
            name=name,
            tio_lang=tio_lang,
            imports=imports,
            base_url=base_url,
            prelude_template=prelude_template,
            timeout=timeout,
        )

    def prepare_code(self, code: str) -> str:
        """Supprime les commentaires (* ... *) et encadre par Print[...] si absent."""
        cleaned = _strip_mathematica_comments(code)
        if cleaned and "Print[" not in cleaned:
            cleaned = f"Print[{cleaned}]"
        return super().prepare_code(cleaned)

    def derive(
        self,
        name: str,
        imports: Iterable[str] = (),
        *,
        tio_lang: str | None = None,
        base_url: str | None = None,
        prelude_template: str | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> "MathicsTioEnvironment":
        """Crée une instance dérivée héritant de cet environnement Mathics."""
        merged_imports = list(self.imports) + [s for s in imports if s and s.strip()]
        return MathicsTioEnvironment(
            name=name,
            tio_lang=self.tio_lang if tio_lang is None else tio_lang,
            imports=merged_imports,
            base_url=self.base_url if base_url is None else base_url,
            prelude_template=self.prelude_template if prelude_template is None else prelude_template,
            timeout=self.timeout if timeout is None else timeout,
        )


# Environnements Try It Online (Haskell, Julia, Scala, Mathematica/Mathics)
haskell_env = TioRunEnvironment(name="haskell", tio_lang="haskell-ghc")
julia_env = TioRunEnvironment(name="julia", tio_lang="julia")
scala_env = TioRunEnvironment(name="scala", tio_lang="scala")
mathematica_env = MathicsTioEnvironment()
