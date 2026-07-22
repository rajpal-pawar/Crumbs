//! `index/schema.rs` — Database schema initialisation.
//!
//! All DDL lives here.  Every statement uses `CREATE TABLE IF NOT EXISTS` or
//! `CREATE VIRTUAL TABLE IF NOT EXISTS` so `apply_schema` is **idempotent**
//! and safe to call on every startup without manual migration tracking.
//!
//! # Schema design
//!
//! ## `documents` (metadata store)
//! The canonical record for each indexed file.  Both the FTS and vector tables
//! refer back to this table via `doc_id`.
//!
//! ## `docs_fts` (FTS5 full-text index)
//! A content-less FTS5 table that stores the body text for BM25 keyword
//! search.  The `content` option points at `documents` so SQLite can
//! reconstruct snippets without duplicating the text in the FTS index itself.
//!
//! ## `embeddings` (sqlite-vec vector index)
//! A `vec0` virtual table.  The embedding column is declared as `float[384]`
//! for 384-dim text embeddings (BAAI/bge-small-en-v1.5) and a separate
//! `embeddings_images` table stores 512-dim CLIP image embeddings.
//! Dimension validation happens at the application layer in `writer.rs`.
//!
//! ## `schema_version` (migration sentinel)
//! A single-row table holding the current schema version so future migrations
//! can be gated on it.

use rusqlite::Connection;
use tracing::{debug, warn};

use crate::index::DbError;

/// Current schema version.  Increment this when making breaking DDL changes.
///
/// v2: Migrated text embeddings from MiniLM-L6-v2 → BAAI/bge-small-en-v1.5.
///     All existing embeddings are mathematically incompatible and must be
///     regenerated, so the migration drops all data tables.
pub const SCHEMA_VERSION: i64 = 2;

/// Apply all DDL statements to `conn`.
///
/// This function is idempotent — calling it on a fully initialised database
/// is a no-op.
///
/// # Errors
/// Propagates [`DbError::Rusqlite`] for any SQL failure.
pub fn apply_schema(conn: &Connection) -> Result<(), DbError> {
    debug!("applying schema (version {})", SCHEMA_VERSION);

    // -----------------------------------------------------------------
    // Phase 0: Migration guard — hard reset if schema is older than v2.
    //
    // v2 swapped the text embedding model (MiniLM → BGE-small).  The
    // 384-dim vectors produced by BGE live in a different semantic space
    // so every existing embedding must be discarded and regenerated.
    // We also drop `documents` and `docs_fts` to force a clean crawl.
    // -----------------------------------------------------------------
    let needs_reset = match conn.query_row(
        "SELECT version FROM schema_version WHERE id = 1",
        [],
        |row| row.get::<_, i64>(0),
    ) {
        Ok(v) => v < SCHEMA_VERSION,
        // Table doesn't exist yet (fresh database) — no reset needed,
        // the CREATE IF NOT EXISTS statements will handle it.
        Err(_) => false,
    };

    if needs_reset {
        warn!("schema version < {} detected — dropping all data tables for clean re-index", SCHEMA_VERSION);
        // Drop triggers first (they reference `documents`).
        let _ = conn.execute_batch("
            DROP TRIGGER IF EXISTS documents_ai;
            DROP TRIGGER IF EXISTS documents_ad;
            DROP TRIGGER IF EXISTS documents_au;
        ");
        // Drop FTS5 virtual table, then the backing view.
        let _ = conn.execute_batch("DROP TABLE IF EXISTS docs_fts;");
        let _ = conn.execute_batch("DROP VIEW IF EXISTS documents_view;");
        // Drop vec0 virtual tables (outside transaction — same reason as
        // creation: vec0 manages its own shadow tables).
        let _ = conn.execute_batch("DROP TABLE IF EXISTS embeddings;");
        let _ = conn.execute_batch("DROP TABLE IF EXISTS embeddings_images;");
        // Drop the main documents table last.
        let _ = conn.execute_batch("DROP TABLE IF EXISTS documents;");
        debug!("old data tables dropped — will recreate from scratch");
    }

    // -----------------------------------------------------------------
    // Phase 1: Regular tables + FTS5 inside an explicit transaction.
    //
    // Virtual table modules (vec0) must NOT be created inside explicit
    // transactions — they create their own shadow tables internally and
    // can corrupt the WAL when nested inside BEGIN EXCLUSIVE.
    // -----------------------------------------------------------------
    conn.execute_batch("BEGIN EXCLUSIVE;").map_err(DbError::Rusqlite)?;

    if let Err(e) = conn.execute_batch(CORE_SCHEMA_SQL) {
        let _ = conn.execute_batch("ROLLBACK;");
        return Err(DbError::Rusqlite(e));
    }

    // Ensure the schema_version row exists and is up to date.
    if let Err(e) = conn.execute(
        "INSERT INTO schema_version (id, version) VALUES (1, ?1)
         ON CONFLICT(id) DO UPDATE SET version = ?1",
        rusqlite::params![SCHEMA_VERSION],
    ) {
        let _ = conn.execute_batch("ROLLBACK;");
        return Err(DbError::Rusqlite(e));
    }

    if let Err(e) = conn.execute_batch(VEC0_SCHEMA_SQL) {
        let _ = conn.execute_batch("ROLLBACK;");
        return Err(DbError::Rusqlite(e));
    }

    conn.execute_batch("COMMIT;").map_err(DbError::Rusqlite)?;

    // Force a WAL checkpoint so the schema is flushed to the main DB file.
    // This prevents "disk image is malformed" errors from un-checkpointed
    // shadow tables.
    if let Err(e) = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);") {
        warn!(error = %e, "WAL checkpoint failed (non-fatal)");
    }

    debug!("schema applied successfully");
    Ok(())
}

// ---------------------------------------------------------------------------
// DDL — Core tables (safe inside a transaction)
// ---------------------------------------------------------------------------

const CORE_SCHEMA_SQL: &str = "
-- -------------------------------------------------------------------------
-- schema_version — single-row migration sentinel
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),  -- enforces single row
    version INTEGER NOT NULL
);

-- -------------------------------------------------------------------------
-- documents — canonical record per indexed file
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Absolute path to the source file (unique — one row per path).
    path        TEXT NOT NULL UNIQUE,

    -- MIME type detected at index time (e.g. 'text/plain', 'image/png').
    mime_type   TEXT NOT NULL DEFAULT 'application/octet-stream',

    -- File body text (UTF-8).  NULL for non-text files (images, etc.).
    body        TEXT,

    -- SHA-256 hex digest of the file contents at index time.
    -- Used to skip unchanged files on re-index.
    checksum    TEXT NOT NULL DEFAULT '',

    -- File size in bytes at index time.
    size_bytes  INTEGER NOT NULL DEFAULT 0,

    -- Wall-clock timestamps (Unix epoch seconds, UTC).
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at  INTEGER NOT NULL DEFAULT (unixepoch())
);

-- Index for fast lookup by path (already covered by UNIQUE, but explicit
-- for clarity).
CREATE INDEX IF NOT EXISTS idx_documents_path
    ON documents (path);

-- Index for incremental re-index: find files modified since last run.
CREATE INDEX IF NOT EXISTS idx_documents_updated
    ON documents (updated_at);

-- -------------------------------------------------------------------------
-- docs_fts — FTS5 full-text index (BM25 via the built-in `rank` column)
-- -------------------------------------------------------------------------
-- A view to project the exact columns FTS5 expects, as FTS5 `content=` tables
-- must return exactly the matching columns when queried by rowid.
CREATE VIEW IF NOT EXISTS documents_view AS
    SELECT replace(path, rtrim(path, replace(path, '/', '')), '') AS title,
           body,
           id
    FROM documents;

-- `content='documents_view'` tells FTS5 to read body text from the view
-- rather than storing a second copy, saving disk space.
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    title,
    body,
    content     = 'documents_view',
    content_rowid = 'id'
);

-- Triggers to keep the FTS index perfectly in sync with the `documents` table.
-- This prevents FTS corruption, as FTS5 strictly requires the exact OLD values
-- when deleting/updating an external content row.
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO docs_fts(rowid, title, body)
    VALUES (new.id,
            -- extract filename from path for the title
            replace(new.path, rtrim(new.path, replace(new.path, '/', '')), ''),
            new.body);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, title, body)
    VALUES ('delete', old.id,
            replace(old.path, rtrim(old.path, replace(old.path, '/', '')), ''),
            old.body);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, title, body)
    VALUES ('delete', old.id,
            replace(old.path, rtrim(old.path, replace(old.path, '/', '')), ''),
            old.body);
    INSERT INTO docs_fts(rowid, title, body)
    VALUES (new.id,
            replace(new.path, rtrim(new.path, replace(new.path, '/', '')), ''),
            new.body);
END;
";

const VEC0_SCHEMA_SQL: &str = "
-- We use `float[384]` as sqlite-vec strictly requires the exact array dimension.
-- This supports 384-dim f32 text embeddings (BAAI/bge-small-en-v1.5, 1536 bytes).
-- The `doc_id` auxiliary column lets us JOIN back to `documents` after ANN
-- search without a separate lookup.
--
-- CRITICAL: This must be created OUTSIDE explicit transactions (BEGIN/COMMIT).
-- sqlite-vec's vec0 module creates shadow tables internally, and nesting
-- inside an explicit transaction corrupts the database.
CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
    embedding float[384],
    +doc_id   INTEGER,
    +dim      INTEGER
);

-- Separate table for 512-dim image embeddings (CLIP).
CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_images USING vec0(
    embedding float[512],
    +doc_id   INTEGER,
    +dim      INTEGER
);
";
