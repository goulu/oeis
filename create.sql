-- Structure de la base de données SQLite pour le clone OEIS

-- 1. Table des utilisateurs
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) DEFAULT '',
    role VARCHAR(20) DEFAULT 'contributor', -- 'contributor', 'reviewer', 'admin'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Table des séquences OEIS
CREATE TABLE IF NOT EXISTS sequences (
    id VARCHAR(7) PRIMARY KEY, -- ex: 'A000045'
    name TEXT NOT NULL DEFAULT '',
    offset_val INTEGER DEFAULT 0,
    terms_preview TEXT DEFAULT '',
    keywords TEXT DEFAULT '',
    author TEXT DEFAULT '',
    inlink_count INTEGER DEFAULT 0,
    current_implementation_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(current_implementation_id) REFERENCES implementations(id)
);

-- 3. Table des implémentations de fonctions et soumissions (drafts / approved)
CREATE TABLE IF NOT EXISTS implementations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id VARCHAR(7) NOT NULL,
    user_id INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'approved', -- 'draft', 'submitted', 'approved', 'rejected'
    definition_code TEXT NOT NULL,         -- Code Python complet de déclaration (ex: Sequence(...))
    iterf_code TEXT,                      -- Code source ou expression de iterf
    itemf_code TEXT,                      -- Code source ou expression de itemf
    containf_code TEXT,                   -- Code source ou expression de containf
    auxiliary_code TEXT,                  -- Fonctions ou classes auxiliaires (ex: def fAXXXXXX)
    description TEXT,
    test_status VARCHAR(20) DEFAULT 'untested', -- 'passed', 'failed', 'timeout'
    test_logs TEXT,
    terms_verified_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(sequence_id) REFERENCES sequences(id),
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_impl_seq_id ON implementations(sequence_id);
CREATE INDEX IF NOT EXISTS idx_impl_user_id ON implementations(user_id);
CREATE INDEX IF NOT EXISTS idx_impl_status ON implementations(status);

-- 4. Table des termes bruts compressés par répétition (compatibilité database.py)
CREATE TABLE IF NOT EXISTS seq (
    id INTEGER NOT NULL,
    i INTEGER NOT NULL,
    n INTEGER NOT NULL,
    repeat INTEGER NOT NULL,
    PRIMARY KEY(id, i, n)
);
CREATE INDEX IF NOT EXISTS idx_seq_n ON seq(n, repeat);

-- 5. Table d'index inversé numérique (3-Grams pour recherche instantanée)
CREATE TABLE IF NOT EXISTS ngram_index (
    ngram_hash INTEGER NOT NULL,
    sequence_id VARCHAR(7) NOT NULL,
    offset_pos INTEGER NOT NULL,
    PRIMARY KEY (ngram_hash, sequence_id, offset_pos)
);
CREATE INDEX IF NOT EXISTS idx_ngram_hash ON ngram_index(ngram_hash);

-- 6. Table virtuelle FTS5 pour recherche plein texte
CREATE VIRTUAL TABLE IF NOT EXISTS sequences_fts USING fts5(
    id UNINDEXED,
    name,
    keywords,
    author,
    tokenize='porter'
);