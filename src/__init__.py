"""
Package oeis : mini Encyclopedia of Integer Sequences en Python.
"""

from .sequence import Sequence
import os
import sys

# Assure que le dossier src est dans sys.path pour les imports internes
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
