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
    pub text_embedding: Option<&'a [f32]>,
    pub image_embedding: Option<&'a [f32]>,

    /// Maximum number of results to return.
    pub limit: usize,

    /// RRF constant *k*.  Standard value is 60.
    pub rrf_k: f32,
}

impl<'a> SearchQuery<'a> {
    pub fn new(text: &'a str, text_embedding: Option<&'a [f32]>, image_embedding: Option<&'a [f32]>, limit: usize) -> Self {
        SearchQuery {
            text,
            text_embedding,
            image_embedding,
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
    let vec_found = if query.text_embedding.is_some() || query.image_embedding.is_some() {
        match run_vector(conn, query.text_embedding, query.image_embedding, query.limit * 2) {
            Ok(vec_hits) => {
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
            }
            Err(e) => {
                warn!(error = %e, "vector search failed — continuing with BM25 results only");
                false
            }
        }
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

/// Run FTS5 BM25 search.  Returns up to `limit` rows ordered by relevance.
///
/// For **multi-term queries** (2+ terms after stop-word filtering), the
/// function runs one BM25 query per term, then ranks documents by the number
/// of distinct query terms they matched (term coverage).  Ties within the
/// same coverage tier are broken by best BM25 rank across terms.
///
/// This ensures a document matching 3/4 query terms always outranks one that
/// merely repeats a single term many times — directly fixing the App.jsx-vs-PDF
/// ranking issue where OR-based BM25 rewarded term *frequency* over *coverage*.
///
/// For **single-term queries**, a plain FTS5 BM25 query is used (unchanged).
fn run_bm25(conn: &Connection, text: &str, limit: usize) -> Result<Vec<RawHit>, DbError> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let terms = extract_query_terms(text);
    if terms.is_empty() {
        debug!(query = %text, "all query terms filtered as stop-words — skipping BM25");
        return Ok(Vec::new());
    }

    // Single-term: plain BM25 query (no coverage logic needed).
    if terms.len() == 1 {
        let fts_query = format!(r#""{}""#, terms[0]);
        return run_bm25_query(conn, &fts_query, limit);
    }

    // ---------------------------------------------------------------
    // Multi-term: per-term BM25 queries → merge by term coverage
    // ---------------------------------------------------------------
    // HashMap doc_id → (RawHit, term_count, best_bm25_rank)
    let mut doc_info: HashMap<i64, (RawHit, usize, usize)> = HashMap::new();

    for term in &terms {
        let fts_query = format!(r#""{}""#, term);
        match run_bm25_query(conn, &fts_query, limit * 2) {
            Ok(hits) => {
                for (rank_0, hit) in hits.into_iter().enumerate() {
                    let entry = doc_info.entry(hit.doc_id)
                        .or_insert_with(|| (hit, 0, usize::MAX));
                    entry.1 += 1;                        // term_count
                    entry.2 = entry.2.min(rank_0);       // best_bm25_rank
                }
            }
            Err(e) => {
                warn!(term = %term, error = %e, "per-term BM25 query failed");
            }
        }
    }

    // Sort: primary by term_count DESC, secondary by best BM25 rank ASC.
    let mut results: Vec<_> = doc_info.into_values().collect();
    results.sort_by(|a, b| {
        b.1.cmp(&a.1)                   // more matching terms = better
            .then_with(|| a.2.cmp(&b.2)) // lower BM25 rank = better
    });
    results.truncate(limit);

    let hits: Vec<RawHit> = results.into_iter().map(|(hit, _, _)| hit).collect();
    debug!(query = %text, count = hits.len(), "BM25 hits (multi-term coverage)");
    Ok(hits)
}

/// Execute a single FTS5 BM25 query and return raw hits ordered by rank.
fn run_bm25_query(conn: &Connection, fts_query: &str, limit: usize) -> Result<Vec<RawHit>, DbError> {
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
            rusqlite::params![fts_query, limit as i64],
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

    Ok(hits)
}

/// Escape characters that have special meaning in FTS5 MATCH expressions
/// and strip English stop-words to reduce linguistic noise in BM25 scoring.
///
/// # Stop-word filtering
///
/// Common English function words ("a", "the", "is", …) are removed because
/// they appear in almost every document and therefore add no discriminative
/// power to BM25 ranking.  Tokens shorter than 2 characters are also dropped
/// — unless the entire original query is a single short term (e.g. the
/// programming language "R") in which case the lone token is preserved so
/// the search is not silently discarded.
///
/// If filtering removes **all** terms the function returns an empty string,
/// which causes `run_bm25` to skip the FTS leg entirely and rely on the
/// semantic vector search path instead.
#[allow(dead_code)] // used by unit tests
fn escape_fts5(text: &str) -> String {
    let filtered = extract_query_terms(text);
    if filtered.is_empty() {
        return String::new();
    }

    // Wrap each surviving term in quotes and join with OR.
    filtered
        .iter()
        .map(|term| format!(r#""{}""#, term))
        .collect::<Vec<_>>()
        .join(" OR ")
}

// ---------------------------------------------------------------------------
// Query term extraction and stop-word filtering
// ---------------------------------------------------------------------------

/// Strict English stop-word list — common function words that carry
/// almost zero discriminative power for BM25 retrieval.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "of", "for", "with", "by", "about", "to", "from",
    "it", "this", "that",
];

/// Extract and filter query terms from raw user input.
///
/// Non-alphanumeric characters are replaced with spaces, stop-words and
/// single-character tokens are removed (unless the query is a single term).
fn extract_query_terms(text: &str) -> Vec<String> {
    let safe_text: String = text
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();

    let raw_terms: Vec<&str> = safe_text.split_whitespace().collect();

    if raw_terms.len() == 1 {
        // Single-term query — keep it regardless of length / stop-word status
        // so the user always gets *something* from BM25.
        raw_terms.into_iter().map(|s| s.to_string()).collect()
    } else {
        raw_terms
            .into_iter()
            .filter(|t| {
                let lower = t.to_lowercase();
                lower.len() >= 2 && !STOP_WORDS.contains(&lower.as_str())
            })
            .map(|s| s.to_string())
            .collect()
    }
}

/// Run sqlite-vec ANN (approximate nearest neighbour) search.
/// Returns up to `limit` rows ordered by vector distance (ascending = closest).
///
/// sqlite-vec `vec0` KNN query syntax:
/// ```sql
/// SELECT rowid, distance FROM embeddings
///   WHERE embedding MATCH ?1 AND k = ?2;
/// ```
/// The virtual table provides a hidden `distance` column automatically when
/// using MATCH.  We query vec0 alone (no JOINs — vec0 doesn't support them
/// in KNN mode) and then resolve doc_id → path in a second step.
fn run_vector(
    conn: &Connection,
    text_embedding: Option<&[f32]>,
    image_embedding: Option<&[f32]>,
    limit: usize,
) -> Result<Vec<RawHit>, DbError> {
    let mut knn_results: Vec<(i64, f64)> = Vec::new();

    // Query A: Text table
    if let Some(emb) = text_embedding {
        let blob: &[u8] = emb.as_bytes();
        let mut stmt = conn
            .prepare(
                "SELECT doc_id, distance
                 FROM embeddings
                 WHERE embedding MATCH ?1
                   AND k = ?2",
            )
            .map_err(DbError::Rusqlite)?;

        let hits = stmt
            .query_map(
                rusqlite::params![blob, limit as i64],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)),
            )
            .map_err(DbError::Rusqlite)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(DbError::Rusqlite)?;
        knn_results.extend(hits);
    }

    // Query B: Images table
    if let Some(emb) = image_embedding {
        let blob: &[u8] = emb.as_bytes();
        let mut stmt = conn
            .prepare(
                "SELECT doc_id, distance
                 FROM embeddings_images
                 WHERE embedding MATCH ?1
                   AND k = ?2",
            )
            .map_err(DbError::Rusqlite)?;

        let hits = stmt
            .query_map(
                rusqlite::params![blob, limit as i64],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)),
            )
            .map_err(DbError::Rusqlite)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(DbError::Rusqlite)?;
        knn_results.extend(hits);
    }

    // Combine and sort by distance ASC (closest matches first)
    knn_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate in case a doc_id somehow matched both
    let mut unique_results = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (doc_id, dist) in knn_results {
        if !seen.contains(&doc_id) {
            seen.insert(doc_id);
            unique_results.push((doc_id, dist));
        }
    }

    unique_results.truncate(limit);

    // Step 2: Resolve doc_id → path from the documents table.
    let mut hits = Vec::with_capacity(unique_results.len());
    for (doc_id, _distance) in unique_results {
        let path: String = conn
            .query_row(
                "SELECT path FROM documents WHERE id = ?1",
                rusqlite::params![doc_id],
                |row| row.get(0),
            )
            .unwrap_or_default();

        let title = std::path::Path::new(&path)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();

        hits.push(RawHit {
            doc_id,
            path,
            title,
            snippet: None, // vector search doesn't produce text snippets
        });
    }

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
        assert_eq!(escaped, r#""hello" OR "world""#);
    }

    #[test]
    fn test_escape_fts5_plain() {
        let escaped = escape_fts5("rust async");
        assert_eq!(escaped, r#""rust" OR "async""#);
    }

    #[test]
    fn test_escape_fts5_stop_words_stripped() {
        // "a boy holding a drink" → stop-words "a" removed, short tokens dropped
        let escaped = escape_fts5("a boy holding a drink");
        assert_eq!(escaped, r#""boy" OR "holding" OR "drink""#);
    }

    #[test]
    fn test_escape_fts5_all_stop_words_returns_empty() {
        // All terms are stop-words → returns empty so BM25 is skipped.
        let escaped = escape_fts5("the a");
        assert_eq!(escaped, "");
    }

    #[test]
    fn test_escape_fts5_single_short_term_preserved() {
        // A lone single-character query should be kept (edge case: "R").
        let escaped = escape_fts5("R");
        assert_eq!(escaped, r#""R""#);
    }
}
