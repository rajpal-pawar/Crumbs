//! `index/writer.rs` — Upsert logic for the FTS and vector tables.
//!
//! # Transaction contract
//! `upsert` takes a `&Transaction` — NOT a `&Connection`.  Callers own
//! begin/commit/rollback, enabling batched writes (50–100× faster than
//! one transaction per row).
//!
//! # Vector blobs
//! `Vec<f32>` embeddings are stored as raw byte blobs via `zerocopy::AsBytes`
//! — zero allocation, no serialisation overhead.

use std::path::{Path, PathBuf};

use rusqlite::{OptionalExtension, Transaction};
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;
use zerocopy::AsBytes as _;

use crate::config::Config;
use crate::extractor;
use crate::index::DbError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Everything needed to index a single document.
#[derive(Debug)]
pub struct DocumentRecord<'a> {
    pub path:       &'a Path,
    pub title:      &'a str,
    /// Full body text for FTS. `None` for binary/image files.
    pub body:       Option<&'a str>,
    pub mime_type:  &'a str,
    /// SHA-256 hex digest — used to skip unchanged files on re-index.
    pub checksum:   &'a str,
    pub size_bytes: u64,
}

/// A single embedding chunk.
#[derive(Debug)]
pub struct EmbeddingRecord {
    /// Must be exactly 384 (text) or 512 (image) f32 values.
    pub vector: Vec<f32>,
}

impl EmbeddingRecord {
    fn validate(&self) -> Result<(), DbError> {
        let dim = self.vector.len();
        if dim != 384 && dim != 512 {
            return Err(DbError::Schema(format!(
                "unsupported embedding dimension {dim}: expected 384 or 512"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// upsert
// ---------------------------------------------------------------------------

/// Upsert a document and its embeddings within an active `Transaction`.
///
/// Atomically keeps `documents`, `docs_fts`, and `embeddings` in sync.
/// Returns the `documents.id` rowid of the upserted row.
pub fn upsert(
    tx: &Transaction,
    doc: &DocumentRecord<'_>,
    embeddings: &[EmbeddingRecord],
) -> Result<i64, DbError> {
    // Validate all vectors before touching the DB.
    for emb in embeddings {
        emb.validate()?;
    }

    let path_str = doc.path.to_string_lossy();
    debug!(path = %path_str, "upserting document");

    // ------------------------------------------------------------------
    // 1. Upsert `documents`
    // ------------------------------------------------------------------
    tx.execute(
        "INSERT INTO documents (path, body, mime_type, checksum, size_bytes, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, unixepoch())
         ON CONFLICT(path) DO UPDATE SET
             body       = excluded.body,
             mime_type  = excluded.mime_type,
             checksum   = excluded.checksum,
             size_bytes = excluded.size_bytes,
             updated_at = unixepoch()",
        rusqlite::params![
            path_str,
            doc.body,
            doc.mime_type,
            doc.checksum,
            doc.size_bytes as i64,
        ],
    )
    .map_err(DbError::Rusqlite)?;

    let doc_id: i64 = tx
        .query_row(
            "SELECT id FROM documents WHERE path = ?1",
            rusqlite::params![path_str],
            |row| row.get(0),
        )
        .map_err(DbError::Rusqlite)?;
    // ------------------------------------------------------------------
    // 3. Replace embeddings
    // ------------------------------------------------------------------
    if !embeddings.is_empty() {
        tx.execute(
            "DELETE FROM embeddings WHERE doc_id = ?1",
            rusqlite::params![doc_id],
        )
        .map_err(DbError::Rusqlite)?;

        tx.execute(
            "DELETE FROM embeddings_images WHERE doc_id = ?1",
            rusqlite::params![doc_id],
        )
        .map_err(DbError::Rusqlite)?;

        let mut stmt_text = tx
            .prepare(
                "INSERT INTO embeddings (embedding, doc_id, dim) VALUES (?1, ?2, ?3)",
            )
            .map_err(DbError::Rusqlite)?;

        let mut stmt_image = tx
            .prepare(
                "INSERT INTO embeddings_images (embedding, doc_id, dim) VALUES (?1, ?2, ?3)",
            )
            .map_err(DbError::Rusqlite)?;

        for emb in embeddings {
            let dim = emb.vector.len() as i64;
            // zerocopy: Vec<f32> → &[u8], zero allocation.
            let blob: &[u8] = emb.vector.as_bytes();

            if dim == 384 {
                stmt_text.execute(rusqlite::params![blob, doc_id, dim])
                    .map_err(DbError::Rusqlite)?;
            } else if dim == 512 {
                stmt_image.execute(rusqlite::params![blob, doc_id, dim])
                    .map_err(DbError::Rusqlite)?;
            }
        }

        debug!(doc_id, count = embeddings.len(), "embeddings written");
    } else {
        warn!(
            doc_id,
            path = %path_str,
            "upserting document with no embeddings — vector search will not cover this file"
        );
    }

    Ok(doc_id)
}

// ---------------------------------------------------------------------------
// delete
// ---------------------------------------------------------------------------

/// Remove a document and all its FTS / embedding rows.
///
/// Returns `true` if a row was deleted, `false` if the path was not indexed.
pub fn delete(tx: &Transaction, path: &Path) -> Result<bool, DbError> {
    let path_str = path.to_string_lossy();

    let maybe_id: Option<i64> = tx
        .query_row(
            "SELECT id FROM documents WHERE path = ?1",
            rusqlite::params![path_str],
            |row| row.get(0),
        )
        .optional()
        .map_err(DbError::Rusqlite)?;

    let doc_id = match maybe_id {
        Some(id) => id,
        None => return Ok(false),
    };


    // Clean embeddings.
    tx.execute(
        "DELETE FROM embeddings WHERE doc_id = ?1",
        rusqlite::params![doc_id],
    )
    .map_err(DbError::Rusqlite)?;

    tx.execute(
        "DELETE FROM embeddings_images WHERE doc_id = ?1",
        rusqlite::params![doc_id],
    )
    .map_err(DbError::Rusqlite)?;

    // Remove the document row.
    tx.execute(
        "DELETE FROM documents WHERE id = ?1",
        rusqlite::params![doc_id],
    )
    .map_err(DbError::Rusqlite)?;

    debug!(doc_id, path = %path_str, "document deleted from index");
    Ok(true)
}

// ---------------------------------------------------------------------------
// scan_directories
// ---------------------------------------------------------------------------

/// Scan a directory, extract documents, and process them.
pub fn scan_directories(config: &Config, path: &PathBuf) -> Result<(), DbError> {
    info!("Starting scan of directory: {}", path.display());

    let walker = WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file());

    for entry in walker {
        let file_path = entry.path();

        match extractor::extract(file_path, config) {
            Ok(Some(extracted)) => {
                // To fully index these, we would need to run them through the
                // embedding pipeline and open a database transaction.
                // For now, we log the successful extraction.
                info!(
                    path = %file_path.display(),
                    mime_type = %extracted.mime_type(),
                    "successfully extracted file"
                );
            }
            Ok(None) => {
                debug!(path = %file_path.display(), "skipped un-indexable file");
            }
            Err(e) => {
                error!(path = %file_path.display(), error = %e, "extraction error during scan");
            }
        }
    }

    info!("Completed scan of directory: {}", path.display());
    Ok(())
}

