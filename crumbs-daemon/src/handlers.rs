//! Request handlers — Phase 3.
//!
//! # Architecture
//!
//! ```text
//!  IPC request (async task)
//!       │
//!       │  tokio::task::spawn_blocking
//!       ▼
//!  OS blocking thread
//!       ├─ WalkDir: discover files
//!       ├─ extractor::extract: read + checksum + decode
//!       ├─ Batch accumulation (text docs + image docs)
//!       │
//!       │  When batch full OR walk complete:
//!       ├─ embed::embed_text_batch  → open MiniLM → infer → DROP session
//!       ├─ embed::embed_image_batch → open CLIP   → infer → DROP session
//!       │
//!       └─ db.with_conn: begin TX → writer::upsert × N → commit
//! ```
//!
//! # Tokio starvation rule (CRITICAL)
//! SQLite queries, file I/O, and ONNX inference are all synchronous blocking
//! calls.  The entire reindex pipeline lives inside `spawn_blocking`.  Do NOT
//! move any of it onto async Tokio threads.
//!
//! # Memory budget
//! - Idle:  < 150 MB (no sessions loaded).
//! - Active text batch:  + ~90 MB (MiniLM weights).
//! - Active image batch: + ~350 MB (CLIP weights).
//! Sessions are always dropped before the next is opened, so peak RAM is
//! max(90, 350) MB above idle, well within the 1 GB daemon budget.

use std::path::PathBuf;
use std::sync::Arc;

use image::DynamicImage;
use serde_json::json;
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::embed;
use crate::extractor::{self, Extracted};
use crate::index::{
    search::{SearchHit, SearchQuery},
    writer::{DocumentRecord, EmbeddingRecord},
    Database,
};
use crate::ipc::{Request, Response};

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

/// Handle a `search` request.
///
/// Params: `{ "query": "<text>", "limit": <int> }`.
///
/// Pipeline:
/// 1. Lazily embed the query text with MiniLM (open session, infer, drop).
/// 2. Run hybrid BM25 + vector search via RRF.
/// 3. Fall back to BM25-only if MiniLM model is absent.
pub async fn handle_search(
    req: Request,
    config: &Arc<Config>,
    db: &Arc<Database>,
) -> Response {
    let query_text = match req.params.get("query").and_then(|v| v.as_str()) {
        Some(q) if !q.trim().is_empty() => q.to_owned(),
        _ => return Response::failure(req.id, "missing or empty 'query' parameter"),
    };

    let limit = req
        .params
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(20)
        .min(100) as usize;

    debug!(id = %req.id, query = %query_text, limit, "handling search request");

    let db     = Arc::clone(db);
    let config = Arc::clone(config);

    // spawn_blocking: ONNX inference + SQLite queries are both synchronous.
    let result = tokio::task::spawn_blocking(move || -> Result<Vec<SearchHit>, String> {
        // ------------------------------------------------------------------
        // 1. Lazily embed the query text.
        //    If the model is absent, degrade to BM25-only (embedding = None).
        // ------------------------------------------------------------------
        let query_embedding: Option<Vec<f32>> =
            match embed::embed_text_batch(&[query_text.clone()], &config) {
                Ok(mut vecs) if !vecs.is_empty() => Some(vecs.remove(0)),
                Ok(_) => {
                    warn!("embed_text_batch returned empty results for query");
                    None
                }
                Err(embed::EmbedError::ModelNotFound(_)) => {
                    debug!("MiniLM model absent — falling back to BM25-only search");
                    None
                }
                Err(e) => {
                    warn!(error = %e, "text embedding failed — falling back to BM25-only");
                    None
                }
            };
        // MiniLM session is dropped inside embed_text_batch. ✓

        // ------------------------------------------------------------------
        // 2. Run hybrid search.
        // ------------------------------------------------------------------
        let sq = SearchQuery::new(
            &query_text,
            query_embedding.as_deref(),
            limit,
        );

        db.with_conn(|conn| {
            crate::index::search::search(conn, &sq)
                .map_err(|e| crate::index::DbError::Schema(e.to_string()))
        })
        .map_err(|e| e.to_string())
    })
    .await;

    match result {
        Ok(Ok(hits)) => {
            let total = hits.len();
            let json_hits: Vec<serde_json::Value> =
                hits.into_iter().map(hit_to_json).collect();
            Response::success(req.id, json!({ "hits": json_hits, "total": total }))
        }
        Ok(Err(e))   => Response::failure(req.id, format!("search error: {e}")),
        Err(join_err) => Response::failure(
            req.id,
            format!("spawn_blocking panicked: {join_err}"),
        ),
    }
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

/// Handle a `status` request.
pub async fn handle_status(
    req: Request,
    config: &Arc<Config>,
    db: &Arc<Database>,
) -> Response {
    debug!(id = %req.id, "handling status request");

    let config = Arc::clone(config);
    let db     = Arc::clone(db);

    let result = tokio::task::spawn_blocking(move || {
        db.with_conn(|conn| {
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
                .map_err(crate::index::DbError::Rusqlite)?;
            Ok(count)
        })
    })
    .await;

    let doc_count = match result {
        Ok(Ok(n))  => n,
        Ok(Err(e)) => { warn!(error = %e, "status doc-count query failed"); -1 }
        Err(_)     => -1,
    };

    let minilm_ready = config.model_cache_dir().join("minilm-l6-int8.onnx").exists();
    let clip_ready   = config.model_cache_dir().join("clip-vit-b32-int8.onnx").exists();

    Response::success(
        req.id,
        json!({
            "version":          env!("CARGO_PKG_VERSION"),
            "status":           "idle",
            "data_dir":         config.data_dir.display().to_string(),
            "db_path":          config.db_path().display().to_string(),
            "max_file_bytes":   config.max_file_bytes,
            "embed_batch_size": config.embed_batch_size,
            "onnx_threads":     config.onnx_intra_threads,
            "models": {
                "minilm_ready": minilm_ready,
                "clip_ready":   clip_ready,
            },
            "doc_count":        doc_count,
            "watch_dirs":       config
                                    .watch_dirs
                                    .iter()
                                    .map(|p| p.display().to_string())
                                    .collect::<Vec<_>>(),
        }),
    )
}

// ---------------------------------------------------------------------------
// reindex
// ---------------------------------------------------------------------------

/// Handle a `reindex` request.
///
/// The entire pipeline runs inside `tokio::task::spawn_blocking`:
///
/// ```text
/// for each file in watch_dirs:
///   skip if > max_file_bytes
///   extractor::extract(path) → Extracted::Text | Extracted::Image | None
///   accumulate into text_batch or image_batch
///
///   if batch_size reached:
///     embed_text_batch  → drop MiniLM session
///     embed_image_batch → drop CLIP session
///     db: BEGIN TX
///       writer::upsert × N
///     COMMIT
///     clear batches
///
/// flush remaining batch (same embed + commit sequence)
/// ```
pub async fn handle_reindex(
    req: Request,
    config: &Arc<Config>,
    db: &Arc<Database>,
) -> Response {
    info!(id = %req.id, "reindex requested — dispatching to blocking thread pool");

    let config = Arc::clone(config);
    let db     = Arc::clone(db);

    // CRITICAL: the entire pipeline is synchronous — ONNX + SQLite + file I/O.
    let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        run_reindex_pipeline(&config, &db)
    })
    .await;

    match result {
        Ok(Ok(payload)) => Response::success(req.id, payload),
        Ok(Err(e))      => Response::failure(req.id, format!("reindex error: {e}")),
        Err(join_err)   => Response::failure(
            req.id,
            format!("spawn_blocking panicked: {join_err}"),
        ),
    }
}

// ---------------------------------------------------------------------------
// Reindex pipeline (synchronous — runs on spawn_blocking thread)
// ---------------------------------------------------------------------------

/// One pending item in the extraction batch.
struct PendingItem {
    path:       PathBuf,
    title:      String,
    checksum:   String,
    mime_type:  String,
    size_bytes: u64,
    /// Present for text documents.
    body:       Option<String>,
    /// Present for image documents.
    image:      Option<DynamicImage>,
}

pub fn run_reindex_pipeline(
    config: &Config,
    db: &Database,
) -> Result<serde_json::Value, String> {
    let mut stats = ReindexStats::default();

    // Accumulated batch of pending items (mixed text + image).
    let mut text_batch:  Vec<PendingItem> = Vec::with_capacity(config.embed_batch_size);
    let mut image_batch: Vec<PendingItem> = Vec::with_capacity(config.embed_batch_size);

    // -----------------------------------------------------------------------
    // Walk all watch directories.
    // -----------------------------------------------------------------------
    for dir in &config.watch_dirs {
        if !dir.exists() {
            debug!(path = %dir.display(), "watch dir does not exist — skipping");
            continue;
        }

        let walker = walkdir::WalkDir::new(dir)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file());

        for entry in walker {
            stats.scanned += 1;

            // ---------------------------------------------------------------
            // Size gate.
            // ---------------------------------------------------------------
            let meta = match entry.metadata() {
                Ok(m)  => m,
                Err(e) => {
                    warn!(path = %entry.path().display(), error = %e, "metadata error — skipping");
                    stats.skipped += 1;
                    continue;
                }
            };

            if meta.len() > config.max_file_bytes {
                debug!(
                    path = %entry.path().display(),
                    bytes = meta.len(),
                    limit = config.max_file_bytes,
                    "file exceeds size limit — skipping"
                );
                stats.skipped += 1;
                continue;
            }

            // ---------------------------------------------------------------
            // Extract.
            // ---------------------------------------------------------------
            let path = entry.path().to_path_buf();
            let extracted = match extractor::extract(&path, config) {
                Ok(Some(e)) => e,
                Ok(None)    => { stats.skipped += 1; continue; }
                Err(e)      => {
                    warn!(path = %path.display(), error = %e, "extraction error — skipping");
                    stats.errors += 1;
                    continue;
                }
            };

            let title      = path.file_name()
                                 .map(|n| n.to_string_lossy().into_owned())
                                 .unwrap_or_default();
            let checksum   = extracted.checksum().to_owned();
            let mime_type  = extracted.mime_type().to_owned();
            let size_bytes = meta.len();

            match extracted {
                Extracted::Text { body, .. } => {
                    text_batch.push(PendingItem {
                        path, title, checksum, mime_type, size_bytes,
                        body: Some(body),
                        image: None,
                    });
                }
                Extracted::Image { image, .. } => {
                    image_batch.push(PendingItem {
                        path, title, checksum, mime_type, size_bytes,
                        body: None,
                        image: Some(image),
                    });
                }
            }

            // ---------------------------------------------------------------
            // Flush when batch is full.
            // ---------------------------------------------------------------
            if text_batch.len() >= config.embed_batch_size {
                if let Err(e) = flush_text_batch(&mut text_batch, config, db, &mut stats) {
                    warn!(error = %e, "failed to flush text batch — skipping batch");
                    stats.errors += text_batch.len() as u64;
                    text_batch.clear();
                }
            }
            if image_batch.len() >= config.embed_batch_size {
                if let Err(e) = flush_image_batch(&mut image_batch, config, db, &mut stats) {
                    warn!(error = %e, "failed to flush image batch — skipping batch");
                    stats.errors += image_batch.len() as u64;
                    image_batch.clear();
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Flush remaining items.
    // -----------------------------------------------------------------------
    if !text_batch.is_empty() {
        if let Err(e) = flush_text_batch(&mut text_batch, config, db, &mut stats) {
            warn!(error = %e, "failed to flush remaining text batch — skipping");
            stats.errors += text_batch.len() as u64;
            text_batch.clear();
        }
    }
    if !image_batch.is_empty() {
        if let Err(e) = flush_image_batch(&mut image_batch, config, db, &mut stats) {
            warn!(error = %e, "failed to flush remaining image batch — skipping");
            stats.errors += image_batch.len() as u64;
            image_batch.clear();
        }
    }

    info!(
        scanned  = stats.scanned,
        indexed  = stats.indexed,
        skipped  = stats.skipped,
        errors   = stats.errors,
        "reindex complete"
    );

    Ok(json!({
        "files_scanned": stats.scanned,
        "files_indexed": stats.indexed,
        "files_skipped": stats.skipped,
        "files_errored": stats.errors,
    }))
}

// ---------------------------------------------------------------------------
// Batch flush helpers
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ReindexStats {
    scanned: u64,
    indexed: u64,
    skipped: u64,
    errors:  u64,
}

/// Embed a text batch and commit all upserts in a single transaction.
///
/// # RAM lifecycle
/// 1. `embed_text_batch` opens MiniLM → infers → **drops the session**.
/// 2. `db.with_conn` begins a transaction, upserts N rows, commits.
/// 3. `text_batch` is cleared — all `String` bodies are freed.
fn flush_text_batch(
    batch: &mut Vec<PendingItem>,
    config: &Config,
    db: &Database,
    stats: &mut ReindexStats,
) -> Result<(), FlushError> {
    if batch.is_empty() {
        return Ok(());
    }

    // Collect body text refs for the embedder.
    let texts: Vec<String> = batch
        .iter()
        .filter_map(|item| item.body.clone())
        .collect();

    // Embed the text documents using the shared session.
    let embeddings_or_err = embed::embed_text_batch(&texts, config);

    let embeddings: Option<Vec<Vec<f32>>> = match embeddings_or_err {
        Ok(vecs) => Some(vecs),
        Err(embed::EmbedError::ModelNotFound(_)) => {
            warn!("MiniLM absent — upserting text docs without embeddings");
            None
        }
        Err(e) => {
            warn!(error = %e, "text embedding failed — upserting without embeddings");
            None
        }
    };
    // MiniLM session is dropped inside embed_text_batch. ✓

    // Commit all upserts in a single transaction.
    db.with_conn(|conn| {
        let tx = conn.unchecked_transaction()
            .map_err(crate::index::DbError::Rusqlite)?;

        for (i, item) in batch.iter().enumerate() {
            let emb_records: Vec<EmbeddingRecord> = embeddings
                .as_ref()
                .and_then(|vecs| vecs.get(i))
                .map(|v| vec![EmbeddingRecord { vector: v.clone() }])
                .unwrap_or_default();

            let doc = DocumentRecord {
                path:       &item.path,
                title:      &item.title,
                body:       item.body.as_deref(),
                mime_type:  &item.mime_type,
                checksum:   &item.checksum,
                size_bytes: item.size_bytes,
            };

            crate::index::writer::upsert(&tx, &doc, &emb_records)?;
        }

        tx.commit().map_err(crate::index::DbError::Rusqlite)?;
        Ok(())
    })
    .map_err(FlushError::Db)?;

    stats.indexed += batch.len() as u64;
    batch.clear(); // free all String bodies
    Ok(())
}

/// Embed an image batch and commit all upserts in a single transaction.
///
/// # RAM lifecycle
/// 1. `embed_image_batch` opens CLIP → infers → **drops the session**.
/// 2. `db.with_conn` begins a transaction, upserts N rows, commits.
/// 3. `image_batch` is cleared — all `DynamicImage` pixel buffers are freed.
fn flush_image_batch(
    batch: &mut Vec<PendingItem>,
    config: &Config,
    db: &Database,
    stats: &mut ReindexStats,
) -> Result<(), FlushError> {
    if batch.is_empty() {
        return Ok(());
    }

    let images: Vec<DynamicImage> = batch
        .iter()
        .filter_map(|item| item.image.as_ref().map(|img| img.clone()))
        .collect();

    // Open CLIP, embed, drop session.
    let embeddings_or_err = embed::embed_image_batch(&images, config);
    // CLIP session dropped inside embed_image_batch. ✓

    let embeddings: Option<Vec<Vec<f32>>> = match embeddings_or_err {
        Ok(vecs) => Some(vecs),
        Err(embed::EmbedError::ModelNotFound(_)) => {
            warn!("CLIP model absent — upserting image docs without embeddings");
            None
        }
        Err(e) => {
            warn!(error = %e, "image embedding failed — upserting without embeddings");
            None
        }
    };

    db.with_conn(|conn| {
        let tx = conn.unchecked_transaction()
            .map_err(crate::index::DbError::Rusqlite)?;

        for (i, item) in batch.iter().enumerate() {
            let emb_records: Vec<EmbeddingRecord> = embeddings
                .as_ref()
                .and_then(|vecs| vecs.get(i))
                .map(|v| vec![EmbeddingRecord { vector: v.clone() }])
                .unwrap_or_default();

            let doc = DocumentRecord {
                path:       &item.path,
                title:      &item.title,
                body:       None,       // images have no body text
                mime_type:  &item.mime_type,
                checksum:   &item.checksum,
                size_bytes: item.size_bytes,
            };

            crate::index::writer::upsert(&tx, &doc, &emb_records)?;
        }

        tx.commit().map_err(crate::index::DbError::Rusqlite)?;
        Ok(())
    })
    .map_err(FlushError::Db)?;

    stats.indexed += batch.len() as u64;
    batch.clear(); // free all DynamicImage pixel buffers
    Ok(())
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
enum FlushError {
    #[error("database error: {0}")]
    Db(#[from] crate::index::DbError),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hit_to_json(h: SearchHit) -> serde_json::Value {
    json!({
        "doc_id":  h.doc_id,
        "path":    h.path,
        "title":   h.title,
        "snippet": h.snippet,
        "score":   h.rrf_score,
        "sources": {
            "bm25":     h.sources.bm25,
            "vector":   h.sources.vector,
            "fallback": h.sources.fallback,
        }
    })
}
