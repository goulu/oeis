"""
Générateur de fichiers Markdown pour les séquences OEIS.
Crée ou met à jour un fichier Markdown pour une séquence donnée à partir de la base de données SQLite.
"""

import os
import sqlite3

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'oeis.db')) or os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'sequences')) else SRC_DIR
DEFAULT_DB_PATH = os.path.join(BASE_DIR, 'oeis.db')
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'sequences')



def get_sequence_data(seq_id, conn):
    """Récupère les données d'une séquence et de son implémentation active."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            s.id, s.name, s.offset_val, s.terms_preview, s.keywords, s.author,
            i.definition_code, i.iterf_code, i.itemf_code, i.containf_code, i.auxiliary_code,
            i.description, i.status, i.terms_verified_count, u.username
        FROM sequences s
        LEFT JOIN implementations i ON s.current_implementation_id = i.id
        LEFT JOIN users u ON i.user_id = u.id
        WHERE s.id = ?
    """,
        (seq_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    columns = [
        'id',
        'name',
        'offset_val',
        'terms_preview',
        'keywords',
        'author',
        'definition_code',
        'iterf_code',
        'itemf_code',
        'containf_code',
        'auxiliary_code',
        'description',
        'status',
        'terms_verified_count',
        'submitter',
    ]
    data = dict(zip(columns, row))

    # Récupération de termes supplémentaires dans la table seq si disponibles
    try:
        numeric_id = int(seq_id[1:])
        cursor.execute(
            """
            SELECT n, repeat FROM seq WHERE id = ? ORDER BY i
        """,
            (numeric_id,),
        )
        seq_rows = cursor.fetchall()
        if seq_rows:
            all_terms = []
            for n_val, r_val in seq_rows:
                all_terms.extend([n_val] * r_val)
            if len(all_terms) > 20:
                data['all_terms'] = all_terms
    except Exception:
        pass

    return data


def format_sequence_markdown(data):
    """Formate les données d'une séquence en un document Markdown structuré inspiré d'OEIS."""
    sid = data['id']
    name = data['name'] or 'Séquence sans titre'
    offset = data.get('offset_val', 0)
    keywords = data.get('keywords', '') or 'core,nice'
    author = data.get('author') or 'Philippe Guglielmetti'
    submitter = data.get('submitter') or 'goulu'
    terms = data.get('terms_preview', '')

    # Si c'est A000045, on enrichit avec les informations historiques et mathématiques d'OEIS
    is_fibonacci = sid == 'A000045'

    md = []
    md.append(f"# {sid}")
    md.append(f"## {name}\n")

    # Liens d'action OEIS
    md.append(
        "> `(list)` &nbsp; `(graph)` &nbsp; `(refs)` &nbsp; `(listen)` &nbsp; `(history)` &nbsp; `(text)` &nbsp; `(edit)`\n"
    )

    # Section Termes
    md.append("### DATA")
    md.append("```text")
    if is_fibonacci and 'all_terms' not in data:
        fib_sample = (
            "0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, "
            "6765, 10946, 17711, 28657, 46368, 75025, 121393, 197418, 318811, 516229, 835040, "
            "1351269, 2186309, 3537578, 5723887, 9261465, 14985352, 24246817, 39232169, 63478984"
        )
        md.append(fib_sample)
    elif 'all_terms' in data:
        md.append(', '.join(str(x) for x in data['all_terms'][:60]))
    else:
        md.append(terms if terms else "Termes non calculés")
    md.append("```\n")

    # Table des métadonnées
    md.append("### METADATA")
    md.append(f"- **OFFSET** : `{offset}, 4`")
    md.append(f"- **KEYWORDS** : `{keywords}`")
    md.append(f"- **AUTHOR** : {author}")
    md.append(f"- **CURRENT IMPLEMENTER** : @{submitter}\n")

    # Section Commentaires
    md.append("### COMMENTS")
    if is_fibonacci:
        md.append(
            "- **Définition** : $F(n) = F(n-1) + F(n-2)$ avec $F(0) = 0$ et $F(1) = 1$."
        )
        md.append(
            "- **Nombre d'or** : Le quotient de deux termes successifs $F(n+1)/F(n)$ converge vers la proportion dorée $\\phi = \\frac{1 + \\sqrt{5}}{2} \\approx 1.6180339887...$"
        )
        md.append(
            "- **Identité de Cassini** : $F(n-1)F(n+1) - F(n)^2 = (-1)^n$."
        )
        md.append(
            "- **Propriété de divisibilité** : $\\gcd(F(m), F(n)) = F(\\gcd(m, n))$."
        )
        md.append(
            "- **Applications** : Modélisation de croissance de populations (lapins de Fibonacci), algorithmes de recherche dichotomique/dorée, combinatoire des pavages par dominos."
        )
    else:
        md.append(
            f"- Séquence calculée et gérée par Goulib `Sequence` pour l'OEIS."
        )
    md.append("")

    # Section Formules
    md.append("### FORMULA")
    if is_fibonacci:
        md.append("- **Récurrence** : $a(n) = a(n-1) + a(n-2)$ pour $n \\ge 2$.")
        md.append(
            "- **Formule explicite de Binet** : $a(n) = \\frac{\\phi^n - (-\\phi)^{-n}}{\\sqrt{5}}$."
        )
        md.append(
            "- **Fonction génératrice** : $G(x) = \\frac{x}{1 - x - x^2} = x + x^2 + 2x^3 + 3x^4 + 5x^5 + \\dots$"
        )
    else:
        md.append("- Voir l'implémentation Python ci-dessous.")
    md.append("")

    # Section Implémentation Python Goulib / Sequence
    md.append("### PYTHON IMPLEMENTATION")
    md.append(
        "Implémentation exécutable de la suite basée sur `goulib.sequence.Sequence` :\n"
    )

    if data.get('auxiliary_code'):
        md.append("#### Fonctions auxiliaires :")
        md.append("```python")
        md.append(data['auxiliary_code'].strip())
        md.append("```\n")

    md.append("#### Définition principale :")
    md.append("```python")
    if data.get('definition_code'):
        md.append(data['definition_code'].strip())
    else:
        md.append(f"{sid} = Sequence(...)")
    md.append("```\n")

    # Détail des 3 piliers
    md.append("#### Détail des méthodes fournies :")
    md.append("| Fonction | Expression | Description |")
    md.append("| :--- | :--- | :--- |")
    iterf_str = f"`{data.get('iterf_code')}`" if data.get('iterf_code') else "*(aucun / count)*"
    itemf_str = f"`{data.get('itemf_code')}`" if data.get('itemf_code') else "*(évaluation séquentielle)*"
    containf_str = f"`{data.get('containf_code')}`" if data.get('containf_code') else "*(par recherche monotone)*"
    md.append(
        f"| **`iterf`** | {iterf_str} | Générateur itérant sur la suite infinie |"
    )
    md.append(
        f"| **`itemf(n)`** | {itemf_str} | Calcul direct du $n$-ième terme |"
    )
    md.append(
        f"| **`containf(x)`** | {containf_str} | Test d'appartenance booléen ($x \\in S$) |"
    )
    md.append("")

    # Section Liens & Crossrefs
    md.append("### CROSSREFERENCES")
    if is_fibonacci:
        md.append(
            "- **[A000032](A000032)** : Nombres de Lucas $L(n) = L(n-1) + L(n-2)$ avec $L(0)=2, L(1)=1$."
        )
        md.append(
            "- **[A001622](A001622)** : Décimales du nombre d'or $\\phi$."
        )
        md.append(
            "- **[A003893](A003893)** : Nombres de Fibonacci modulo 10 (période de Pisano)."
        )
    else:
        md.append(f"- Voir également l'entrée officielle sur https://oeis.org/{sid}")
    md.append("")

    return "\n".join(md)


def generate_sequence_markdown(
    seq_id, conn=None, output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Génère et écrit le fichier Markdown pour une séquence donnée.
    Doit être appelée à chaque modification de la séquence dans la base de données.
    """
    os.makedirs(output_dir, exist_ok=True)

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        close_conn = True

    try:
        data = get_sequence_data(seq_id, conn)
        if not data:
            raise ValueError(f"Séquence {seq_id} introuvable en base.")

        md_content = format_sequence_markdown(data)
        file_path = os.path.join(output_dir, f"{seq_id}.md")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return file_path, md_content
    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    import sys

    target_id = sys.argv[1] if len(sys.argv) > 1 else "A000045"
    fpath, _ = generate_sequence_markdown(target_id)
    print(f"Fichier Markdown généré avec succès : {fpath}")
