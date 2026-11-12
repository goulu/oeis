"""
Environnements d'exécution locaux pour du code téléchargé (ex. extraits OEIS).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Iterable
from urllib.parse import urlencode

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

    def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Exécute le code en local sur le serveur dans un sous-processus isolé."""
        if not code or not code.strip():
            return {"status": "error", "error": "No code provided"}

        effective_timeout = self.timeout if timeout is None else timeout
        full_code = self.prepare_code(code)

        try:
            proc = subprocess.run(
                [self.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=self._build_process_env(),
                cwd=self.cwd,
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
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": f"Execution timed out ({effective_timeout}s limit).",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


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


class OctaveOnlineEnvironment(ExecutionEnvironment):
    """
    Environnement d'exécution distant pour du code Matlab/Octave
    utilisant https://octave-online.net pour l'évaluation.
    """

    def __init__(
        self,
        name: str = "matlab",
        imports: Iterable[str] = (),
        *,
        base_url: str = "https://octave-online.net",
        language: str = "matlab",
        timeout: float = 10.0,
    ) -> None:
        super().__init__(name=name, imports=imports, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.language = language

    def build_prelude(self) -> str:
        """Génère les instructions d'initialisation Matlab/Octave (ex. pkg load)."""
        if not self.imports:
            return ""
        return "\n".join(f"pkg load {pkg};" for pkg in self.imports)

    def build_eval_url(self, code: str) -> str:
        """Construit l'URL d'évaluation sur Octave Online avec le code préparé."""
        prepared = self.prepare_code(code)
        return f"{self.base_url}/?{urlencode({'code': prepared})}"

    def derive(
        self,
        name: str,
        imports: Iterable[str] = (),
        *,
        base_url: str | None = None,
        language: str | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> "OctaveOnlineEnvironment":
        """Crée une instance dérivée héritant de cet environnement Matlab/Octave."""
        merged_imports = list(self.imports) + [s for s in imports if s and s.strip()]
        return OctaveOnlineEnvironment(
            name=name,
            imports=merged_imports,
            base_url=self.base_url if base_url is None else base_url,
            language=self.language if language is None else language,
            timeout=self.timeout if timeout is None else timeout,
        )

    def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Prépare l'évaluation distante du code Matlab/Octave via Octave Online."""
        if not code or not code.strip():
            return {"status": "error", "error": "No code provided"}

        eval_url = self.build_eval_url(code)
        return {
            "status": "success",
            "mode": "external",
            "language": self.language,
            "url": eval_url,
            "output": f"Evaluation on Octave Online: {eval_url}",
        }


# Instance Matlab utilisant https://octave-online.net pour l'évaluation
matlab_env = OctaveOnlineEnvironment(
    name="matlab",
    base_url="https://octave-online.net",
    language="matlab",
    timeout=10.0,
)
