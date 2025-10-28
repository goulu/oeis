"""
tests for OEIS.py and database.py

(OEIS is Neil Sloane's On-Line Encyclopedia of Integer Sequences at https://oeis.org/)
"""

__author__ = "Philippe Guglielmetti"
__copyright__ = "Copyright (c) 2015 Philippe Guglielmetti"
__license__ = "LGPL"
__credits__ = []

import os
import logging
import re

from goulib import decorators
from goulib.tests import *

from oeis import oeis
from database import database

path = os.path.dirname(os.path.abspath(__file__))

slow = []  # list of slow sequences


def test(f, ref, name, timeout=1.):
    ref = list(ref)
    n = len(ref)
    timeout, f.timeout = f.timeout, timeout  # save Sequence's timeout
    liste = []
    try:
        for i, x in enumerate(f):
            if i >= n:
                break
            if x != ref[i]:
                raise ValueError(
                    '%s: term %d: got %s, expected %s' % (name, i, x, ref[i]))
            liste.append(x)

    except decorators.TimeoutError:
        if i < min(10, n / 2):
            slow.append((i, name, f.desc))
            logging.warning('%s timeout after only %d loops' % (name, i))
        else:
            logging.info('%s timeout after %d loops' % (name, i))
    finally:
        f.timeout = timeout  # restore Sequence's timeout
    return liste


def data(s):
    s2 = s[1:]

    # http://stackoverflow.com/questions/807863/how-to-output-list-of-floats-to-a-binary-file-in-python
    try:
        return database[s]
    except IndexError:
        pass

    from urllib.request import urlopen
    try:  # is there a local, patched file ?
        file = open('b%s.txt' % s2, 'r')
        logging.warning('reading b%s.txt' % s2)
    # FileNotFoundError (not defined in Py2.7) download the B-file from OEIS
    except OSError:
        file = urlopen('http://oeis.org/A%s/b%s.txt' % (s2, s2))
        logging.info('downloading b%s.txt' % s2)
    res = []
    for line in file:  # files are iterable
        # urlopen yields bytes in py3, local open() yields str; normalize to str
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='replace')
        c = line[0]
        if c == '#':
            continue  # skip comment
        # use a string regex (works on decoded text)
        m = re.search(r"(\d+)\s+(-?\d+)", line)
        if m:
            m = m.groups()
            if len(m) == 2:
                n = int(m[1])
                res.append(n)

    database[s] = res

    return res


def check(f, s=None):
    try:
        name = f.__name__
    except:
        name, f = f, globals()[f]
    # to break as soon as an error happens
    test(f, s or name, name=name, timeout=None)


if __name__ == "__main__":
    keys = oeis.keys()
    for name in sorted(keys):  # test in sorted order to spot slows faster
        logging.info(name)
        time_limit = 0.1  # second
        ref = list(data(name))
        result = test(oeis[name], ref, name, time_limit)
        print(str(oeis[name]), result)
    print('slower Sequences:')
    slow.sort()
    for (i, name, desc) in slow:
        print(i, name, desc)
