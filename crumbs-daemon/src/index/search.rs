//! `index/search.rs` — Hybrid BM25 + Vector search with Reciprocal Rank Fusion.
//!
//! # Pipeline
//!
//! ```text
//!  query text ──┬──► FTS5 BM25 search  ──► ranked doc list (bm25_hits)
//!               │
//!  query vector ┴──► vec0 ANN search   ──► ranked doc list (vec_hits)
//!                                                 │
//!                              RRF fusion ◄────────┘
//!                                   │
//!                    LIKE fallback ◄─┤ (if both lists are empty)
//!                                   │
//!                              SearchHit list (sorted by rrf_score desc)
//! ```
//!
//! # Reciprocal Rank Fusion (RRF)
//!
//! RRF score for a document *d*:
//!
//! ```text
//! rrf(d) = Σ  1.0 / (k + rank_i(d))
//!         sources
//! ```
//!
//! where *k* = 60 (standard constant) and *rank_i(d)* is the 1-based rank of
//! *d* in source *i* (or ∞ / not present → contributes 0).
//!
//! **CRITICAL:** ranks are cast to `f32` before the division so that integer
//! arithmetic does not floor the result to zero.
//!
//! # Fallback
//!
//! When neither BM25 nor vector search returns results (e.g. the FTS index is
//! empty on first run, or the query doesn't tokenise to any known term), a
//! plain `LIKE` query on `documents.body` and `documents.path` is executed as
//! a best-effort fallback, ensuring the user always sees *something* useful.

use std::collections::HashMap;

use rusqlite::Connection;
use tracing::{debug, warn};
use zerocopy::AsBytes as _;

use crate::index::DbError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single result returned to the caller.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub doc_id:     i64,
    pub path:       String,
    pub title:      String,
    /// Snippet of body text around the matching terms.  `None` for binary
    /// documents or when the body is unavailable.
    pub snippet:    Option<String>,
    /// Combined RRF relevance score (higher = more relevant).
    pub rrf_score:  f32,
    /// Which sources contributed to this hit.
    pub sources:    HitSources,
}

/// Bitmask-style flags indicating which search pipeline contributed a hit.
#[derive(Debug, Clone, Default)]
pub struct HitSources {
    pub bm25:     bool,
    pub vector:   bool,
    pub fallback: bool,
}

/// Input to the search pipeline.
pub struct SearchQuery<'a> {
    /// Raw query string (used for FTS5 MATCH and LIKE fallback).
    pub text: &'a str,

    /// Optional pre-computed query embedding for vector search.
    /// Pass `None` to skip the vector search leg (e.g. if ONNX is not yet
    /// loaded).
    pub embedding: Option<&'a [f32]>,

    /// Maximum number of results to return.
    pub limit: usize,

    /// RRF constant *k*.  Standard value is 60.
    pub rrf_k: f32,
}

impl<'a> SearchQuery<'a> {
    pub fn new(text: &'a str, embedding: Option<&'a [f32]>, limit: usize) -> Self {
        SearchQuery {
            text,
            embedding,
            limit,
            rrf_k: 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Execute the hybrid search pipeline on `conn`.
///
/// This function is synchronous and must be called from within
/// `tokio::task::spawn_blocking` — it will block the current thread while
/// executing SQLite queries.
///
/// # Errors
/// Propagates [`DbError::Rusqlite`] for any SQLite failure.
pub fn search(conn: &Connection, query: &SearchQuery<'_>) -> Result<Vec<SearchHit>, DbError> {
    // Candidate pool: doc_id → accumulated RRF score + metadata.
    let mut pool: HashMap<i64, CandidateEntry> = HashMap::new();

    // ------------------------------------------------------------------
    // Leg 1: BM25 full-text search via FTS5
    // ------------------------------------------------------------------
    let bm25_hits = run_bm25(conn, query.text, query.limit * 2)?;
    let bm25_found = !bm25_hits.is_empty();

    for (rank_0based, hit) in bm25_hits.into_iter().enumerate() {
        // rank is 1-based; cast to f32 BEFORE division to prevent integer flooring.
        let rank = (rank_0based + 1) as f32;
        let contribution = 1.0_f32 / (query.rrf_k + rank);

        let entry = pool.entry(hit.doc_id).or_insert_with(|| CandidateEntry {
            path:    hit.path,
            title:   hit.title,
            snippet: hit.snippet,
            rrf:     0.0,
            sources: HitSources::default(),
        });
        entry.rrf += contribution;
        entry.sources.bm25 = true;
    }

    // ------------------------------------------------------------------
    // Leg 2: Vector ANN search via sqlite-vec
    // ------------------------------------------------------------------
    let vec_found = if let Some(emb) = query.embedding {
        let vec_hits = run_vector(conn, emb, query.limit * 2)?;
        let found = !vec_hits.is_empty();

        for (rank_0based, hit) in vec_hits.into_iter().enumerate() {
            // Cast rank to f32 before division — same guard as above.
            let rank = (rank_0based + 1) as f32;
            let contribution = 1.0_f32 / (query.rrf_k + rank);

            let entry = pool.entry(hit.doc_id).or_insert_with(|| CandidateEntry {
                path:    hit.path,
                title:   hit.title,
                snippet: None,
                rrf:     0.0,
                sources: HitSources::default(),
            });
            entry.rrf += contribution;
            entry.sources.vector = true;
        }

        found
    } else {
        debug!("vector search skipped — no query embedding provided");
        false
    };

    // ------------------------------------------------------------------
    // Leg 3: LIKE fallback (metadata match)
    // ------------------------------------------------------------------
    // Triggered when BOTH BM25 and vector search return nothing.  This
    // ensures the user always gets candidate results even when the index is
    // empty or the query terms are too rare for FTS5 to score.
    if !bm25_found && !vec_found {
        warn!(
            query = query.text,
            "BM25 and vector search both returned 0 results — running LIKE fallback"
        );

        let fallback_hits = run_like_fallback(conn, query.text, query.limit)?;

        for (rank_0based, hit) in fallback_hits.into_iter().enumerate() {
            let rank = (rank_0based + 1) as f32;
            let contribution = 1.0_f32 / (query.rrf_k + rank);

            let entry = pool.entry(hit.doc_id).or_insert_with(|| CandidateEntry {
                path:    hit.path,
                title:   hit.title,
                snippet: hit.snippet,
                rrf:     0.0,
                sources: HitSources::default(),
            });
            entry.rrf += contribution;
            entry.sources.fallback = true;
        }
    }

    // ------------------------------------------------------------------
    // Assemble and sort the final result list
    // ------------------------------------------------------------------
    let mut results: Vec<SearchHit> = pool
        .into_iter()
        .map(|(doc_id, e)| SearchHit {
            doc_id,
            path:      e.path,
            title:     e.title,
            snippet:   e.snippet,
            rrf_score: e.rrf,
            sources:   e.sources,
        })
        .collect();

    // Sort descending by RRF score (higher = more relevant).
    results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(query.limit);

    debug!(
        query = query.text,
        returned = results.len(),
        "search complete"
    );

    Ok(results)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Intermediate candidate accumulator.
struct CandidateEntry {
    path:    String,
    title:   String,
    snippet: Option<String>,
    rrf:     f32,
    sources: HitSources,
}

/// A raw hit from one search leg (before RRF fusion).
struct RawHit {
    doc_id:  i64,
    path:    String,
    title:   String,
    snippet: Option<String>,
}

/// Run FTS5 BM25 search.  Returns up to `limit` rows ordered by rank
/// (best = most negative FTS5 score, ascending = best first).
fn run_bm25(conn: &Connection, text: &str, limit: usize) -> Result<Vec<RawHit>, DbError> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Escape FTS5 special characters so user input doesn't break the query.
    let escaped = escape_fts5(text);

    let mut stmt = conn
        .prepare(
            "SELECT d.id, d.path,
                    snippet(docs_fts, 1, '<b>', '</b>', '…', 20) AS snip
             FROM docs_fts
             JOIN documents d ON d.id = docs_fts.rowid
             WHERE docs_fts MATCH ?1
             ORDER BY bm25(docs_fts, 10.0, 1.0)
             LIMIT ?2",
        )
        .map_err(DbError::Rusqlite)?;

    let hits = stmt
        .query_map(
            rusqlite::params![escaped, limit as i64],
            |row| {
                let path: String = row.get(1)?;
                let title = std::path::Path::new(&path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                Ok(RawHit {
                    doc_id:  row.get(0)?,
                    path,
                    title,
                    snippet: row.get(2)?,
                })
            },
        )
        .map_err(DbError::Rusqlite)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(DbError::Rusqlite)?;

    debug!(query = %text, count = hits.len(), "BM25 hits");
    Ok(hits)
}

/// Run sqlite-vec ANN (approximate nearest neighbour) search.
/// Returns up to `limit` rows ordered by vector distance (ascending = closest).
fn run_vector(
    conn: &Connection,
    embedding: &[f32],
    limit: usize,
) -> Result<Vec<RawHit>, DbError> {
    // Cast &[f32] → &[u8] for the blob parameter.
    let blob: &[u8] = embedding.as_bytes();

    let mut stmt = conn
        .prepare(
            "SELECT e.doc_id, d.path,
                    vec_distance_L2(e.embedding, ?1) AS dist
             FROM embeddings e
             JOIN documents d ON d.id = e.doc_id
             WHERE e.embedding MATCH ?1
               AND k = ?2
             ORDER BY dist ASC",
        )
        .map_err(DbError::Rusqlite)?;

    let hits = stmt
        .query_map(
            rusqlite::params![blob, limit as i64],
            |row| {
                let path: String = row.get(1)?;
                let title = std::path::Path::new(&path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                Ok(RawHit {
                    doc_id:  row.get(0)?,
                    path,
                    title,
                    snippet: None, // vector search doesn't produce text snippets
                })
            },
        )
        .map_err(DbError::Rusqlite)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(DbError::Rusqlite)?;

    debug!(count = hits.len(), "vector ANN hits");
    Ok(hits)
}

/// LIKE-based fallback search against `documents.body` and `documents.path`.
///
/// This is intentionally simple and slow — it only runs when the primary
/// search legs return nothing, and `limit` is always small.
fn run_like_fallback(
    conn: &Connection,
    text: &str,
    limit: usize,
) -> Result<Vec<RawHit>, DbError> {
    let pattern = format!("%{}%", text.replace('%', r"\%").replace('_', r"\_"));

    let mut stmt = conn
        .prepare(
            "SELECT id, path,
                    CASE WHEN body IS NOT NULL
                         THEN substr(body, 1, 200)
                         ELSE NULL
                    END AS snip
             FROM documents
             WHERE body    LIKE ?1 ESCAPE '\\'
                OR path    LIKE ?1 ESCAPE '\\'
             ORDER BY updated_at DESC
             LIMIT ?2",
        )
        .map_err(DbError::Rusqlite)?;

    let hits = stmt
        .query_map(
            rusqlite::params![pattern, limit as i64],
            |row| {
                let path: String = row.get(1)?;
                let title = std::path::Path::new(&path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                Ok(RawHit {
                    doc_id:  row.get(0)?,
                    path,
                    title,
                    snippet: row.get(2)?,
                })
            },
        )
        .map_err(DbError::Rusqlite)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(DbError::Rusqlite)?;

    debug!(count = hits.len(), "LIKE fallback hits");
    Ok(hits)
}

/// Escape characters that have special meaning in FTS5 MATCH expressions.
///
/// FTS5 treats `"`, `*`, `^`, and whitespace specially.  We wrap the whole
/// query in double-quotes to treat it as a phrase query, escaping any
/// embedded double-quotes.
fn escape_fts5(text: &str) -> String {
    // Replace every `"` with `""` (FTS5 double-quote escape).
    let escaped = text.replace('"', "\"\"");
    format!("\"{}\"", escaped)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_cast_nonzero() {
        // Verify that the RRF contribution for rank=1 with k=60 is non-zero.
        // If rank were integer-divided this would be 0 (1 / 61 = 0 in integer math).
        let rank: f32 = 1.0_f32;
        let k: f32 = 60.0;
        let contribution = 1.0_f32 / (k + rank);
        assert!(
            contribution > 0.0,
            "RRF contribution must be positive, got {contribution}"
        );
        // 1/61 ≈ 0.01639
        assert!(
            (contribution - 0.01639).abs() < 0.0001,
            "unexpected RRF value: {contribution}"
        );
    }

    #[test]
    fn test_escape_fts5_quotes() {
        let escaped = escape_fts5(r#"hello "world""#);
        assert_eq!(escaped, r#""hello ""world"""#);
    }

    #[test]
    fn test_escape_fts5_plain() {
        let escaped = escape_fts5("rust async");
        assert_eq!(escaped, r#""rust async""#);
    }
}
