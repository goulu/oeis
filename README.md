# Python OEIS & Collaborative Sequence Platform

A Python framework and modern web application for infinite integer sequences inspired by Neil Sloane's **[The On-Line Encyclopedia of Integer Sequences (OEIS)](https://oeis.org/)**.

This platform combines functional Python abstractions for infinite series with a crowdsourcing-ready architecture, automated Markdown sheet generation, and a high-performance integer search engine implementing the query syntax of **[oeis.org/hints.html](https://oeis.org/hints.html)**.

---

## Table of Contents

- [Core Concept: The Three Pillars of `Sequence`](#core-concept-the-three-pillars-of-sequence)
- [Search Engine Architecture (hints.html & Russ Cox's Design)](#search-engine-architecture)
- [Database Schema (`create.sql`)](#database-schema)
- [On-The-Fly Markdown & Web Server](#on-the-fly-markdown--web-server)
- [Project Directory Structure](#project-directory-structure)
- [Quick Start](#quick-start)
- [References & Academic Context](#references--academic-context)
- [License](#license)

---

## Core Concept: The Three Pillars of `Sequence`

At the heart of the project is the [`Sequence`](src/sequence.py) class. Rather than merely storing static tables of integers, each entry combines three complementary functional representations:

1. **`iterf` (Infinite Generator / Iterator)**: An iterator producing the sequence terms consecutively ($a_0, a_1, a_2, \dots$).
2. **`itemf(n)` (Closed-Form / Direct Access)**: A function computing the $n$-th element directly in $O(1)$ or sub-linear time (e.g., via Binet's formula, recurrence matrix exponentiation, or direct arithmetic).
3. **`containf(x)` (Membership Predicate)**: A boolean function determining whether $x \in S$ (e.g., using integer square tests for Fibonacci numbers without expanding the entire sequence).

Algebraic and set-theoretic operations (`+`, `-`, `*`, `//`, `|`, `&`, `filter`, `apply`) are supported natively between `Sequence` instances.

---

## Search Engine Architecture

Searching an encyclopedia of hundreds of thousands of integer sequences presents unique algorithmic challenges that standard relational databases and basic full-text engines cannot handle.

### The Problem with Naive Relational Queries
A sequence query such as `10946, 17711, 28657` (Fibonacci terms $F_{21}, F_{22}, F_{23}$) cannot be efficiently resolved by SQL `LIKE '%10946, 17711%'` or dynamic recursive Common Table Expressions (CTEs):
* Common integers like `0, 1, 2` appear in nearly every sequence, causing naive index lookups to produce massive candidate sets.
* Dynamic CTE joins across multiple tables degrade severely when matching sequences of 6 to 10 terms.

### The Dual-Engine Solution
Inspired by **Russ Cox's 2005–2010 search engine overhaul of the OEIS**, the search subsystem ([`src/search_engine.py`](src/search_engine.py)) employs a hybrid architecture:

```
                            User Query
                                │
                                ▼
                       Query Parser & Normalizer
             (Detects: Integers, Ranges, Prefixes, Text, Wildcards)
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
     Numerical Inverted Index            Full-Text Index (FTS5)
      (Sliding 3-Grams Hash)              (Porter Stemming)
               │                                 │
               └────────────────┬────────────────┘
                                ▼
                    Intersection & Positional Check
                                ▼
                 Scoring & Ranking Engine (+Score)
```

1. **Numerical 3-Gram Inverted Index**:
   * Each sequence is split into overlapping sliding windows of length 3: $(t_i, t_{i+1}, t_{i+2})$.
   * A 64-bit hash of each triplet points to a postings list of `(sequence_id, offset_pos)`.
   * While individual terms (`1`, `2`) are common, triplets such as `(10946, 17711, 28657)` are highly selective. Candidate sequence IDs are intersected in memory, and relative positional offsets ($\text{pos}_{k} = \text{pos}_0 + k$) are validated in microseconds.
2. **Wildcards and Subsequences (`hints.html` Syntax)**:
   * **`_` (Single term wildcard)**: Matches any single integer at a given index (e.g., `1, 2, _, 4, 5`).
   * **`__` (Subsequence wildcard)**: Matches an arbitrary subsequence, non-necessarily contiguous (e.g., `1, 2, __, 8, 13`).
   * **Prefix queries**: Supports targeted field searches (`id:`, `author:`, `keyword:`, `seq:`, `subseq:`).
3. **Full-Text Search (SQLite FTS5)**:
   * Indexes names, comments, and keywords using the Porter stemmer.
4. **Relevance Scoring (`+Score`)**:
   * Computes matching confidence (+100 for exact contiguous subsequences, with proximity bonuses for matches starting near index 0, combined with BM25 rankings for textual matches).

---

## Database Schema (`create.sql`)

The database ([`create.sql`](create.sql)) is organized into relational tables designed for collaborative editing and rapid index retrieval:

* **`users`**: User authentication, profiles, and roles (`contributor`, `reviewer`, `admin`).
* **`sequences`**: Core catalog entries ($A000001 \dots$) with preview terms, offset, keywords, and a pointer to the active implementation.
* **`implementations`**: Versioned code submissions for each sequence:
  * `definition_code`: Full Python initialization expression.
  * `iterf_code`, `itemf_code`, `containf_code`: Granular source code for each method.
  * `auxiliary_code`: Supporting helper functions (e.g., `def fA051885(n): ...`).
  * `status`: Editorial state (`approved`, `draft`, `submitted`, `rejected`).
  * `test_status`: Automated verification results against official reference $b$-files.
* **`ngram_index`**: Primary numerical index mapping `ngram_hash` to `(sequence_id, offset_pos)`.
* **`sequences_fts`**: SQLite FTS5 virtual table for Porter-stemmed metadata search.
* **`seq`**: Run-length encoded (RLE) compressed storage of historical sequence terms `(id, i, n, repeat)` for backwards compatibility.

---

## On-The-Fly Markdown & Web Server

### Markdown Synchronization
Whenever a sequence implementation or description is modified in the database, [`src/markdown_generator.py`](src/markdown_generator.py) regenerates a standalone Markdown document in the `sequences/` directory (e.g., [`sequences/A000045.md`](sequences/A000045.md)).

Each generated Markdown sheet mirrors the canonical OEIS layout:
* **DATA**: Integer terms.
* **METADATA**: Offset, keywords, original author, current Python implementer.
* **COMMENTS & FORMULA**: Mathematical properties, Binet formulas, generating functions (rendered with KaTeX / MathJax).
* **PYTHON IMPLEMENTATION**: Formatted code block showing the `Sequence` instantiation and method breakdown.
* **CROSSREFERENCES**: Links to adjacent and related sequences.

### On-The-Fly HTML Conversion
The FastAPI server ([`src/server.py`](src/server.py)) reads the static Markdown files and transforms them into modern, responsive HTML pages on-the-fly. It embeds an **interactive Python test console** allowing users to evaluate `itemf(n)` and `containf(x)` directly in their browser.

---

## Project Directory Structure

```text
oeis/
├── src/                          # Application source code
│   ├── __init__.py               # Python package initialization
│   ├── sequence.py               # Core Sequence abstraction (iterf, itemf, containf)
│   ├── oeis.py                   # Catalogue of 270+ implemented OEIS sequences
│   ├── database.py               # Database connector & RLE compression routines
│   ├── search_engine.py          # hints.html query parser, 3-gram & FTS5 search engine
│   ├── markdown_generator.py     # Database-to-Markdown sheet exporter
│   ├── server.py                 # FastAPI web application (on-the-fly Markdown-to-HTML)
│   └── init_db.py                # Database population & AST source code extractor
├── tests/                        # Automated test suite
│   └── test_oeis.py              # Verification against reference OEIS b-files
├── sequences/                    # Generated Markdown sheets (A000045.md, etc.)
├── create.sql                    # Relational and FTS5 database schema
├── design.md                     # Comprehensive architecture and security specification
├── pyproject.toml                # Package configuration & pytest settings
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Initialization

Populate SQLite with the schema and extract implementation code from `oeis.py` via Python AST:

```bash
python src/init_db.py
```

### 3. Run Tests

Verify sequence implementations against reference $b$-files:

```bash
pytest
```

### 4. Start the Web Server

Launch the web interface locally:

```bash
uvicorn src.server:app --reload
```

Navigate to:
* **Home**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Fibonacci Page**: [http://127.0.0.1:8000/A000045](http://127.0.0.1:8000/A000045)
* **Search Demo**: [http://127.0.0.1:8000/search?q=10946%2C+17711%2C+28657](http://127.0.0.1:8000/search?q=10946%2C+17711%2C+28657)
* **Search Hints**: [http://127.0.0.1:8000/hints.html](http://127.0.0.1:8000/hints.html)

---

## References & Academic Context

The design of the search backend and database architecture is directly informed by historical research and engineering overhauls of the OEIS:

1. **Neil J. A. Sloane**, *"The On-Line Encyclopedia of Integer Sequences: From Punched Cards to Wiki in 46 Years"*, Conference on Intelligent Computer Mathematics (CICM / CEUR-WS), 2010.  
   *(Documents the transition from manual flat-files to a collaborative wiki and acknowledges Russ Cox for engineering the custom software and search backend).*
2. **Russ Cox**, *"OEIS Server Features and Search Engine Design"*, OeisWiki Technical Documentation, 2010–2015.  
   *(Details the use of the Porter stemming algorithm, inverted numerical indexing for contiguous sub-sequence lookups, and query relevance scoring).*
3. **Russ Cox & Alex Healy**, *"Accelerating OEIS Sequence Lookups"*, OEIS Foundation Reports, 2005.  
   *(Introduced sub-second algorithmic lookup techniques for large mathematical integer repositories).*
4. **Neil J. A. Sloane**, *"The On-Line Encyclopedia of Integer Sequences"*, *Notices of the American Mathematical Society*, Vol. 65, No. 9, 2018.

---

## License

This software is released under the **GNU Lesser General Public License (LGPL)**. Integer sequence definitions and data are subject to the [OEIS End-User License Agreement](https://oeis.org/wiki/The_OEIS_End-User_License_Agreement).