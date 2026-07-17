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

use crate::config::{AtomicConfig, Config, DirState, DirStatusRegistry};
use crate::embed;
use crate::extractor::{self, Extracted};
use crate::index::{
    search::{SearchHit, SearchQuery},
    writer::{DocumentRecord, EmbeddingRecord},
    Database,
};
use crate::ipc::{Request, Response};

// ---------------------------------------------------------------------------
// update_config (synchronous — no spawn_blocking needed)
// ---------------------------------------------------------------------------

/// Handle an `update_config` request.
///
/// Params: `{ "batch_size": <int>, "threads": <int> }`.
/// Both fields are optional; only provided fields are updated.
///
/// The atomic values are updated in-place — the indexing loop reads them
/// on the next batch iteration with no restart.
pub fn handle_update_config(
    req: Request,
    atomic_config: &Arc<AtomicConfig>,
) -> Response {
    if let Some(bs) = req.params.get("batch_size").and_then(|v| v.as_u64()) {
        atomic_config.set_batch_size(bs as usize);
    }
    if let Some(t) = req.params.get("threads").and_then(|v| v.as_i64()) {
        atomic_config.set_threads(t as i16);
    }
    if let Some(p) = req.params.get("paused").and_then(|v| v.as_bool()) {
        crate::state::get_model_manager().set_engine_paused(p);
    }
    info!(
        batch_size = atomic_config.batch_size(),
        threads = atomic_config.threads(),
        paused = crate::state::get_model_manager().is_engine_paused(),
        "engine config updated"
    );
    Response::success(
        req.id,
        json!({
            "batch_size": atomic_config.batch_size(),
            "threads": atomic_config.threads(),
            "paused": crate::state::get_model_manager().is_engine_paused(),
        }),
    )
}

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
        crate::state::get_model_manager().increment_search();
        let search_res = (|| {
            // 1. Lazily embed the query text.
            let query_embedding: Option<Vec<f32>> =
                match embed::embed_text_batch(&[query_text.clone()], &config) {
                    Ok(mut vecs) if !vecs.is_empty() => {
                        let mut chunks = vecs.remove(0);
                        if !chunks.is_empty() {
                            Some(chunks.remove(0))
                        } else {
                            None
                        }
                    },
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

            let clip_embedding: Option<Vec<f32>> =
                match embed::embed_clip_text(&query_text, &config) {
                    Ok(vec) => Some(vec),
                    Err(embed::EmbedError::ModelNotFound(_)) => {
                        debug!("CLIP text model absent — skipping image search");
                        None
                    }
                    Err(e) => {
                        warn!(error = %e, "CLIP text embedding failed — skipping image search");
                        None
                    }
                };

            // 2. Run hybrid search.
            let sq = SearchQuery::new(
                &query_text,
                query_embedding.as_deref(),
                clip_embedding.as_deref(),
                limit,
            );

            db.with_read_conn(|conn| {
                crate::index::search::search(conn, &sq)
                    .map_err(|e| crate::index::DbError::Schema(e.to_string()))
            })
            .map_err(|e| e.to_string())
        })();
        crate::state::get_model_manager().decrement_search();
        search_res
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
    atomic_config: &Arc<AtomicConfig>,
) -> Response {
    debug!(id = %req.id, "handling status request");

    let config = Arc::clone(config);
    let db     = Arc::clone(db);

    let result = tokio::task::spawn_blocking(move || {
        db.with_read_conn(|conn| {
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
    let clip_ready   = config.model_cache_dir().join("clip-vision-int8.onnx").exists();

    let db_size = std::fs::metadata(config.db_path()).map(|m| m.len()).unwrap_or(0);
    let onnx_memory = crate::state::get_model_manager().get_onnx_memory_footprint();

    let (failed_files, skipped_files) = crate::state::get_model_manager().get_indexing_issues();
    tracing::info!("handle_status: sending {} failed_files and {} skipped_files", failed_files.len(), skipped_files.len());

    let failed_json: Vec<serde_json::Value> = failed_files.iter().take(50).map(|(p, r)| {
        json!({"path": p, "reason": r})
    }).collect();
    let skipped_json: Vec<serde_json::Value> = skipped_files.iter().take(50).map(|(p, r)| {
        json!({"path": p, "reason": r})
    }).collect();

    let status_str = if crate::state::get_model_manager().is_indexer_active() {
        "indexing"
    } else {
        "idle"
    };

    let watch_dirs_paths = atomic_config.watch_dirs.read()
        .map(|d| d.clone())
        .unwrap_or_default();

    Response::success(
        req.id,
        json!({
            "version":          env!("CARGO_PKG_VERSION"),
            "status":           status_str,
            "data_dir":         config.data_dir.display().to_string(),
            "db_path":          config.db_path().display().to_string(),
            "db_size":          db_size,
            "onnx_memory":      onnx_memory,
            "paused":           crate::state::get_model_manager().is_engine_paused(),
            "max_file_bytes":   config.max_file_bytes,
            "embed_batch_size": config.embed_batch_size,
            "onnx_threads":     config.onnx_intra_threads,
            "models": {
                "minilm_ready": minilm_ready,
                "clip_ready":   clip_ready,
            },
            "doc_count":        doc_count,
            "watch_dirs":       watch_dirs_paths
                                    .iter()
                                    .map(|p| p.display().to_string())
                                    .collect::<Vec<_>>(),
            "failed_files":     failed_json,
            "skipped_files":    skipped_json,
        }),
    )
}

// ---------------------------------------------------------------------------
// reindex
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU64, Ordering};
use crate::ipc::SharedWriter;

pub async fn handle_reindex(
    req: Request,
    config: &Arc<Config>,
    db: &Arc<Database>,
    writer: SharedWriter,
) -> Response {
    info!(id = %req.id, "reindex requested — starting MPSC pipeline");

    let config_clone = Arc::clone(config);
    let db_clone = Arc::clone(db);
    
    let result = tokio::task::spawn_blocking(move || {
        run_reindex_pipeline_internal(&config_clone, &db_clone, Some(writer))
    }).await;

    match result {
        Ok(Ok(stats)) => Response::success(req.id, json!({
            "files_scanned": stats.scanned,
            "files_indexed": stats.indexed,
            "files_skipped": stats.skipped,
            "files_errored": stats.errors,
        })),
        Ok(Err(e)) => Response::failure(req.id, format!("reindex failed: {}", e)),
        Err(e) => Response::failure(req.id, format!("reindex task panicked: {}", e)),
    }
}

pub fn run_reindex_pipeline(config: &Arc<Config>, db: &Arc<Database>) -> Result<(), String> {
    run_reindex_pipeline_internal(config, db, None).map(|_| ())
}

pub fn run_reindex_pipeline_internal(
    config: &Arc<Config>,
    db: &Arc<Database>,
    writer: Option<SharedWriter>,
) -> Result<ReindexStats, String> {
    crate::state::get_model_manager().set_indexer_active(true, config);
    let res = run_reindex_pipeline_internal_impl(config, db, writer);
    crate::state::get_model_manager().set_indexer_active(false, config);
    res
}

fn run_reindex_pipeline_internal_impl(
    config: &Arc<Config>,
    db: &Arc<Database>,
    writer: Option<SharedWriter>,
) -> Result<ReindexStats, String> {
    let (path_tx, path_rx) = std::sync::mpsc::sync_channel::<PathBuf>(1000);
    
    let scanned = Arc::new(AtomicU64::new(0));

    // Create a directory status registry for per-directory state tracking.
    let dir_registry = Arc::new(DirStatusRegistry::new(&config.watch_dirs));
    
    let producer_scanned = Arc::clone(&scanned);
    let producer_registry = Arc::clone(&dir_registry);
    let config_clone = Arc::clone(config);
    let producer = std::thread::spawn(move || {
        for dir in &config_clone.watch_dirs {
            // Task 1.1: Ensure the crawler only runs on paths explicitly present in watch_dirs
            if !config_clone.watch_dirs.contains(dir) {
                continue;
            }
            if !dir.exists() { continue; }

            // Transition directory to Scanning state.
            producer_registry.set_state(dir, DirState::Scanning);

            // Task 2: Completely block hidden files and folders across Linux and Windows
            let walker = walkdir::WalkDir::new(dir)
                .into_iter()
                .filter_entry(|e| {
                    let file_name = e.file_name().to_string_lossy();
                    if file_name.starts_with('.') {
                        return false;
                    }
                    if e.file_type().is_dir() {
                        let lower = file_name.to_lowercase();
                        !matches!(
                            lower.as_str(),
                            "node_modules" | "target" | "appdata" | "temp"
                            | "windows" | "system32" | "usr" | "bin" | "lib"
                            | "snap" | "flatpak" | "venv" | "env"
                            | "__pycache__" | "build" | "dist"
                        )
                    } else {
                        true
                    }
                });

            for entry_res in walker {
                while crate::state::get_model_manager().is_engine_paused() {
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
                let entry = match entry_res {
                    Ok(e) => e,
                    Err(_) => continue, // Explicitly ignore PermissionDenied or unreadable folders
                };

                if !entry.file_type().is_file() {
                    continue;
                }

                let path = entry.path().to_path_buf();

                // Task 1.2: Absolute boundary assertion check before pushing path to MPSC queue
                if !config_clone.watch_dirs.iter().any(|allowed_dir| path.starts_with(allowed_dir)) {
                    continue; // Reject instantly if it strayed out of bounds
                }

                producer_scanned.fetch_add(1, Ordering::Relaxed);
                if path_tx.send(path).is_err() {
                    break;
                }
            }

            // Transition directory to Indexing state (file discovery complete for this dir).
            producer_registry.set_state(dir, DirState::Indexing);
        }

        // After all directories have been walked, mark any still in Indexing
        // as Completed (the consumer will handle final flush).
        // Note: we only mark Scanning→Indexing here; Completed happens after
        // the consumer finishes processing all files from this dir.
    });

    let config_clone = Arc::clone(config);
    let db_clone = Arc::clone(db);
    let consumer_scanned = Arc::clone(&scanned);
    let consumer_registry = Arc::clone(&dir_registry);
    
    let consumer = std::thread::spawn(move || {
        let mut stats = ReindexStats::default();
        let batch_size = config_clone.embed_batch_size;
        let mut text_batch = Vec::with_capacity(batch_size);
        let mut image_batch = Vec::with_capacity(batch_size);
        
        let rt_handle = tokio::runtime::Handle::try_current().ok();

        let report_progress = |stats: &ReindexStats, writer_opt: &Option<SharedWriter>, scanned_arc: &Arc<AtomicU64>, rt: &Option<tokio::runtime::Handle>, registry: &Arc<DirStatusRegistry>| {
            let s = scanned_arc.load(Ordering::Relaxed);
            let dir_snapshot = registry.snapshot();
            let dirs_json: Vec<serde_json::Value> = dir_snapshot.iter().map(|d| {
                json!({"path": d.path, "state": d.state})
            }).collect();
            let processed = stats.indexed + stats.skipped + stats.errors;

            // Build per-file details for the UI (last 50 of each to avoid huge payloads)
            let failed_json: Vec<serde_json::Value> = stats.failed_files.iter().rev().take(50).map(|(p, r)| {
                json!({"path": p, "reason": r})
            }).collect();
            let skipped_json: Vec<serde_json::Value> = stats.skipped_files.iter().rev().take(50).map(|(p, r)| {
                json!({"path": p, "reason": r})
            }).collect();

            println!(
                "{}",
                serde_json::json!({
                    "status": "indexing",
                    "indexed": stats.indexed,
                    "processed": processed,
                    "errors": stats.errors,
                    "skipped": stats.skipped,
                    "total": s,
                    "directories": dirs_json,
                    "failed_files": failed_json,
                    "skipped_files": skipped_json
                })
            );

            if let (Some(w), Some(h)) = (writer_opt, rt) {
                let w = w.clone();
                let event = serde_json::json!({
                    "method": "progress",
                    "params": {
                        "scanned": s,
                        "indexed": stats.indexed,
                        "errors": stats.errors
                    }
                });
                h.spawn(async move {
                    crate::ipc::write_raw_event(&w, event).await;
                });
            }
        };

        while let Ok(path) = path_rx.recv() {
            while crate::state::get_model_manager().is_engine_paused() {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            if crate::state::get_model_manager().has_pending_searches() {
                debug!("Search request pending. Indexer yielding/pausing to prioritize UI response time.");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "Failed to read metadata — skipping");
                    stats.skipped += 1;
                    stats.skipped_files.push((path.display().to_string(), format!("metadata error: {}", e)));
                    continue;
                }
            };

            let is_semantic = {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                crate::extractor::TEXT_EXTENSIONS.contains(&ext.as_str()) || crate::extractor::IMAGE_EXTENSIONS.contains(&ext.as_str())
            };

            if is_semantic && meta.len() > config_clone.max_file_bytes {
                tracing::warn!(path = %path.display(), size = meta.len(), limit = config_clone.max_file_bytes, "File size exceeds limit — skipping");
                stats.skipped += 1;
                stats.skipped_files.push((path.display().to_string(), format!("file too large: {} bytes (limit {})", meta.len(), config_clone.max_file_bytes)));
                continue;
            }

            let extracted = match extractor::extract(&path, &config_clone) {
                Ok(Some(e)) => e,
                Ok(None) => {
                    tracing::info!(path = %path.display(), "File skipped by extractor");
                    stats.skipped += 1;
                    stats.skipped_files.push((path.display().to_string(), "unsupported format or binary content".to_string()));
                    continue;
                }
                Err(e) => {
                    tracing::error!(path = %path.display(), error = ?e, "Extraction failed — skipping");
                    stats.errors += 1;
                    stats.failed_files.push((path.display().to_string(), format!("{}", e)));
                    continue;
                }
            };

            let title = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let checksum = extracted.checksum().to_owned();
            let mime_type = extracted.mime_type().to_owned();
            let size_bytes = meta.len();

            match extracted {
                Extracted::Text { chunks, .. } => {
                    text_batch.push(PendingItem {
                        path, title, checksum, mime_type, size_bytes,
                        chunks: Some(chunks), image: None,
                    });
                }
                Extracted::Image { image, .. } => {
                    image_batch.push(PendingItem {
                        path, title, checksum, mime_type, size_bytes,
                        chunks: None, image: Some(image),
                    });
                }
            }

            let mut flushed = false;
            if text_batch.len() >= batch_size {
                if let Err(e) = flush_text_batch(&mut text_batch, &config_clone, &db_clone, &mut stats) {
                    let err_msg = format!("{}", e);
                    for item in &text_batch {
                        stats.failed_files.push((item.path.display().to_string(), err_msg.clone()));
                    }
                    stats.errors += text_batch.len() as u64;
                    text_batch.clear();
                }
                flushed = true;
            }
            if image_batch.len() >= batch_size {
                if let Err(e) = flush_image_batch(&mut image_batch, &config_clone, &db_clone, &mut stats) {
                    let err_msg = format!("{}", e);
                    for item in &image_batch {
                        stats.failed_files.push((item.path.display().to_string(), err_msg.clone()));
                    }
                    stats.errors += image_batch.len() as u64;
                    image_batch.clear();
                }
                flushed = true;
            }

            if flushed {
                report_progress(&stats, &writer, &consumer_scanned, &rt_handle, &consumer_registry);
                if crate::state::get_model_manager().has_pending_searches() {
                    info!("Search query detected. Indexer pausing between batches...");
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        }

        let mut final_flushed = false;
        if !text_batch.is_empty() {
            if let Err(e) = flush_text_batch(&mut text_batch, &config_clone, &db_clone, &mut stats) {
                let err_msg = format!("{}", e);
                for item in &text_batch {
                    stats.failed_files.push((item.path.display().to_string(), err_msg.clone()));
                }
                stats.errors += text_batch.len() as u64;
                text_batch.clear();
            }
            final_flushed = true;
        }
        if !image_batch.is_empty() {
            if let Err(e) = flush_image_batch(&mut image_batch, &config_clone, &db_clone, &mut stats) {
                let err_msg = format!("{}", e);
                for item in &image_batch {
                    stats.failed_files.push((item.path.display().to_string(), err_msg.clone()));
                }
                stats.errors += image_batch.len() as u64;
                image_batch.clear();
            }
            final_flushed = true;
        }
        
        if final_flushed {
            report_progress(&stats, &writer, &consumer_scanned, &rt_handle, &consumer_registry);
        }

        stats.scanned = consumer_scanned.load(Ordering::Relaxed);
        stats
    });

    let _ = producer.join();

    // Mark all directories as Completed now that the producer is done.
    for dir in &config.watch_dirs {
        dir_registry.set_state(dir, DirState::Completed);
    }

    let stats = consumer.join().map_err(|_| "Consumer thread panicked".to_string())?;

    // Emit a final progress event with all directories completed.
    let final_dirs: Vec<serde_json::Value> = dir_registry.snapshot().iter().map(|d| {
        json!({"path": d.path, "state": d.state})
    }).collect();
    let processed = stats.indexed + stats.skipped + stats.errors;

    let failed_json: Vec<serde_json::Value> = stats.failed_files.iter().map(|(p, r)| {
        json!({"path": p, "reason": r})
    }).collect();
    let skipped_json: Vec<serde_json::Value> = stats.skipped_files.iter().map(|(p, r)| {
        json!({"path": p, "reason": r})
    }).collect();

    println!(
        "{}",
        serde_json::json!({
            "status": "completed",
            "indexed": stats.indexed,
            "processed": processed,
            "errors": stats.errors,
            "skipped": stats.skipped,
            "total": stats.scanned,
            "directories": final_dirs,
            "failed_files": failed_json,
            "skipped_files": skipped_json
        })
    );

    crate::state::get_model_manager().set_indexing_issues(stats.failed_files.clone(), stats.skipped_files.clone());

    Ok(stats)
}

// ---------------------------------------------------------------------------
// update_folders — add/remove watched directories with DB cleanup
// ---------------------------------------------------------------------------

/// Handle an `update_folders` request.
///
/// Params: `{ "folders": ["<abs path>", ...], "is_onboarded": true }`.
///
/// Saves the new folder list to config, persists to disk, and optionally
/// removes embeddings for paths no longer in the monitored list.
pub async fn handle_update_folders(
    req: Request,
    config: &Arc<Config>,
    db: &Arc<Database>,
    atomic_config: &Arc<AtomicConfig>,
    writer: SharedWriter,
) -> Response {
    let folders: Vec<String> = match req.params.get("folders").and_then(|v| v.as_array()) {
        Some(arr) => arr.iter().filter_map(|v| v.as_str().map(|s| s.to_owned())).collect(),
        None => return Response::failure(req.id, "missing 'folders' array parameter"),
    };

    let is_onboarded = req.params.get("is_onboarded")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let new_dirs: Vec<PathBuf> = folders.iter().map(PathBuf::from).collect();

    info!(folders = ?new_dirs, is_onboarded, "updating monitored folders");

    // Determine removed directories for cleanup.
    let old_dirs = atomic_config.watch_dirs.read()
        .map(|d| d.clone())
        .unwrap_or_default();
    let removed_dirs: Vec<PathBuf> = old_dirs.iter()
        .filter(|old| !new_dirs.contains(old))
        .cloned()
        .collect();

    // Update the atomic config.
    if let Ok(mut dirs) = atomic_config.watch_dirs.write() {
        *dirs = new_dirs.clone();
    }
    atomic_config.is_onboarded.store(is_onboarded, Ordering::Relaxed);

    // Persist to disk by creating a new Config snapshot.
    let mut persisted_config = config.as_ref().clone();
    persisted_config.watch_dirs = new_dirs.clone();
    persisted_config.is_onboarded = is_onboarded;
    if let Err(e) = persisted_config.save_persisted() {
        warn!("Failed to persist config: {}", e);
    }

    // Cleanup: remove documents/embeddings for paths under removed directories.
    if !removed_dirs.is_empty() {
        let db_clone = Arc::clone(db);
        let removed = removed_dirs.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Err(e) = cleanup_removed_dirs(&db_clone, &removed) {
                warn!("Cleanup of removed dirs failed: {}", e);
            }
        }).await;
    }

    // Kick off a reindex if we have folders and are onboarded.
    if is_onboarded && !new_dirs.is_empty() {
        let mut reindex_config = config.as_ref().clone();
        reindex_config.watch_dirs = new_dirs;
        reindex_config.is_onboarded = true;
        let reindex_config = Arc::new(reindex_config);
        let db_clone = Arc::clone(db);
        let writer_clone = writer;

        tokio::task::spawn_blocking(move || {
            if let Err(e) = run_reindex_pipeline_internal(&reindex_config, &db_clone, Some(writer_clone)) {
                warn!("Reindex after folder update failed: {}", e);
            }
        });
    }

    Response::success(
        req.id,
        json!({
            "folders": folders,
            "is_onboarded": is_onboarded,
        }),
    )
}

/// Remove all documents and embeddings whose paths fall under removed directories.
fn cleanup_removed_dirs(db: &Database, removed_dirs: &[PathBuf]) -> Result<(), String> {
    db.with_conn(|conn| {
        let tx = conn.unchecked_transaction()
            .map_err(crate::index::DbError::Rusqlite)?;

        for dir in removed_dirs {
            let prefix = format!("{}%", dir.display());
            info!("Cleaning up documents under removed dir: {}", dir.display());

            // Find doc IDs under this path prefix.
            let mut stmt = tx.prepare(
                "SELECT id FROM documents WHERE path LIKE ?1"
            ).map_err(crate::index::DbError::Rusqlite)?;

            let doc_ids: Vec<i64> = stmt.query_map(
                rusqlite::params![prefix],
                |row| row.get(0),
            )
            .map_err(crate::index::DbError::Rusqlite)?
            .filter_map(|r| r.ok())
            .collect();

            for doc_id in &doc_ids {
                tx.execute(
                    "DELETE FROM embeddings WHERE doc_id = ?1",
                    rusqlite::params![doc_id],
                ).map_err(crate::index::DbError::Rusqlite)?;

                tx.execute(
                    "DELETE FROM embeddings_images WHERE doc_id = ?1",
                    rusqlite::params![doc_id],
                ).map_err(crate::index::DbError::Rusqlite)?;
            }

            tx.execute(
                "DELETE FROM documents WHERE path LIKE ?1",
                rusqlite::params![prefix],
            ).map_err(crate::index::DbError::Rusqlite)?;

            info!("Cleaned up {} documents under {}", doc_ids.len(), dir.display());
        }

        tx.commit().map_err(crate::index::DbError::Rusqlite)?;
        Ok(())
    }).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// get_config — read onboarding status and watch_dirs
// ---------------------------------------------------------------------------

/// Handle a `get_config` request.
///
/// Returns the current onboarding status and list of watched directories.
pub fn handle_get_config(
    req: Request,
    config: &Arc<Config>,
    atomic_config: &Arc<AtomicConfig>,
) -> Response {
    let is_onboarded = atomic_config.is_onboarded.load(Ordering::Relaxed);
    let watch_dirs = atomic_config.watch_dirs.read()
        .map(|d| d.iter().map(|p| p.display().to_string()).collect::<Vec<_>>())
        .unwrap_or_default();

    Response::success(
        req.id,
        json!({
            "is_onboarded": is_onboarded,
            "watch_dirs": watch_dirs,
            "data_dir": config.data_dir.display().to_string(),
        }),
    )
}

// ---------------------------------------------------------------------------
// list_documents — return indexed files from the database
// ---------------------------------------------------------------------------

/// Handle a `list_documents` request.
///
/// Returns a list of all indexed documents with their path, title, mime_type,
/// and size_bytes.  Limited to 500 results to avoid large payloads.
pub async fn handle_list_documents(
    req: Request,
    db: &Arc<Database>,
) -> Response {
    debug!(id = %req.id, "handling list_documents request");

    let db = Arc::clone(db);

    let result = tokio::task::spawn_blocking(move || {
        db.with_read_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT path, replace(path, rtrim(path, replace(path, '/', '')), '') AS title, mime_type, size_bytes FROM documents ORDER BY path ASC LIMIT 500"
            ).map_err(crate::index::DbError::Rusqlite)?;

            let docs: Vec<serde_json::Value> = stmt.query_map([], |row| {
                let path: String = row.get(0)?;
                let title: String = row.get(1)?;
                let mime: String = row.get(2)?;
                let size: i64 = row.get(3)?;
                Ok(json!({
                    "path": path,
                    "title": title,
                    "mime_type": mime,
                    "size_bytes": size,
                }))
            })
            .map_err(crate::index::DbError::Rusqlite)?
            .filter_map(|r| r.ok())
            .collect();

            Ok(docs)
        })
    })
    .await;

    match result {
        Ok(Ok(docs)) => {
            let total = docs.len();
            Response::success(req.id, json!({ "documents": docs, "total": total }))
        }
        Ok(Err(e)) => Response::failure(req.id, format!("list_documents error: {e}")),
        Err(e) => Response::failure(req.id, format!("spawn_blocking panicked: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Reindex pipeline
// ---------------------------------------------------------------------------

struct PendingItem {
    path:       PathBuf,
    title:      String,
    checksum:   String,
    mime_type:  String,
    size_bytes: u64,
    chunks:     Option<Vec<String>>,
    image:      Option<DynamicImage>,
}

// ---------------------------------------------------------------------------
// Batch flush helpers
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ReindexStats {
    pub scanned: u64,
    pub indexed: u64,
    pub skipped: u64,
    pub errors:  u64,
    /// (path, reason) for each file that errored during extraction/indexing.
    pub failed_files:  Vec<(String, String)>,
    /// (path, reason) for each file that was deliberately skipped.
    pub skipped_files: Vec<(String, String)>,
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

    // Collect non-empty body text refs for the embedder.
    let mut texts_to_embed = Vec::new();
    let mut batch_to_text_idx = vec![Vec::new(); batch.len()];

    for (i, item) in batch.iter().enumerate() {
        if let Some(chunks) = &item.chunks {
            for chunk in chunks {
                if !chunk.trim().is_empty() {
                    batch_to_text_idx[i].push(texts_to_embed.len());
                    texts_to_embed.push(chunk.clone());
                }
            }
        }
    }

    // Embed the text documents using the shared session.
    let embeddings: Option<Vec<Vec<Vec<f32>>>> = if texts_to_embed.is_empty() {
        None
    } else {
        match embed::embed_text_batch(&texts_to_embed, config) {
            Ok(vecs) => Some(vecs),
            Err(embed::EmbedError::ModelNotFound(_)) => {
                warn!("MiniLM absent — upserting text docs without embeddings");
                None
            }
            Err(e) => {
                warn!(error = %e, "text embedding failed — upserting without embeddings");
                None
            }
        }
    };
    // MiniLM session is dropped inside embed_text_batch. ✓

    // Commit all upserts in a single transaction.
    db.with_conn(|conn| {
        let tx = conn.unchecked_transaction()
            .map_err(crate::index::DbError::Rusqlite)?;

        for (i, item) in batch.iter().enumerate() {
            let mut emb_records = Vec::new();
            if let Some(embeddings) = &embeddings {
                for &idx in &batch_to_text_idx[i] {
                    if let Some(vecs) = embeddings.get(idx) {
                        for v in vecs {
                            emb_records.push(EmbeddingRecord { vector: v.clone() });
                        }
                    }
                }
            }

            let full_body = item.chunks.as_ref()
                .filter(|c| !c.is_empty())
                .map(|c| c.join("\n"));
            let doc = DocumentRecord {
                path:       &item.path,
                title:      &item.title,
                body:       full_body.as_deref(),
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

// ---------------------------------------------------------------------------
// Background File Watcher Task
// ---------------------------------------------------------------------------

pub fn start_background_watcher(
    config: Arc<Config>,
    _db: Arc<Database>,
    rx: std::sync::mpsc::Receiver<notify::Event>,
    watcher: notify::RecommendedWatcher,
) {
    use notify::event::{ModifyKind, RenameMode};
    use notify::EventKind;
    use tracing::error;

    std::thread::spawn(move || {
        // Keep the watcher alive in this thread context
        let _watcher = watcher;

        let conn = match rusqlite::Connection::open(config.db_path()) {
            Ok(c) => c,
            Err(e) => {
                error!("Background watcher failed to open SQLite connection: {}", e);
                return;
            }
        };

        if let Err(e) = conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;"
        ) {
            error!("Background watcher failed to set connection PRAGMAs: {}", e);
            return;
        }

        if let Err(e) = conn.busy_timeout(std::time::Duration::from_secs(5)) {
            error!("Background watcher failed to set busy timeout: {}", e);
            return;
        }

        loop {
            match rx.recv() {
                Ok(event) => {
                    match event.kind {
                        EventKind::Modify(ModifyKind::Name(RenameMode::Both)) => {
                            if event.paths.len() == 2 {
                                let old_path = &event.paths[0];
                                let new_path = &event.paths[1];

                                let old_allowed = config.watch_dirs.iter().any(|allowed| old_path.starts_with(allowed));
                                let new_allowed = config.watch_dirs.iter().any(|allowed| new_path.starts_with(allowed));

                                if old_allowed && new_allowed {
                                    info!(old = %old_path.display(), new = %new_path.display(), "Handling file rename");
                                    let old_path_str = old_path.to_string_lossy();
                                    let new_path_str = new_path.to_string_lossy();

                                    match conn.execute(
                                        "UPDATE documents SET path = ?1, updated_at = unixepoch() WHERE path = ?2",
                                        rusqlite::params![new_path_str, old_path_str],
                                    ) {
                                        Ok(rows) => {
                                            if rows > 0 {
                                                info!(rows, "Updated path in DB for renamed file");
                                            } else {
                                                debug!("Renamed old path not found in DB, indexing new path");
                                                crate::state::get_model_manager().set_indexer_active(true, &config);
                                                if let Err(e) = index_single_file(&conn, new_path, &config) {
                                                    error!(path = %new_path.display(), error = %e, "Failed to index renamed destination path");
                                                }
                                                crate::state::get_model_manager().set_indexer_active(false, &config);
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to update database for rename: {}", e);
                                        }
                                    }
                                } else if old_allowed {
                                    info!(path = %old_path.display(), "File moved out of watched directory, deleting");
                                    match conn.unchecked_transaction() {
                                        Ok(tx) => {
                                            match crate::index::writer::delete(&tx, old_path) {
                                                Ok(deleted) => {
                                                    if deleted {
                                                        let _ = tx.commit();
                                                        info!(path = %old_path.display(), "Deleted document from DB");
                                                    }
                                                }
                                                Err(e) => {
                                                    error!(path = %old_path.display(), error = %e, "Failed to delete document from DB");
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to begin delete transaction: {}", e);
                                        }
                                    }
                                } else if new_allowed {
                                    info!(path = %new_path.display(), "File moved into watched directory, indexing");
                                    crate::state::get_model_manager().set_indexer_active(true, &config);
                                    if let Err(e) = index_single_file(&conn, new_path, &config) {
                                        error!(path = %new_path.display(), error = %e, "Failed to index new file");
                                    }
                                    crate::state::get_model_manager().set_indexer_active(false, &config);
                                }
                            }
                        }
                        EventKind::Create(_) | EventKind::Modify(_) => {
                            for path in &event.paths {
                                if !config.watch_dirs.iter().any(|allowed| path.starts_with(allowed)) {
                                    continue;
                                }
                                if !path.exists() {
                                    continue;
                                }

                                info!(path = %path.display(), "Handling file create/modify");
                                crate::state::get_model_manager().set_indexer_active(true, &config);
                                if let Err(e) = index_single_file(&conn, path, &config) {
                                    error!(path = %path.display(), error = %e, "Failed to index file");
                                }
                                crate::state::get_model_manager().set_indexer_active(false, &config);
                            }
                        }
                        EventKind::Remove(_) => {
                            for path in &event.paths {
                                if !config.watch_dirs.iter().any(|allowed| path.starts_with(allowed)) {
                                    continue;
                                }
                                info!(path = %path.display(), "Handling file removal");
                                match conn.unchecked_transaction() {
                                    Ok(tx) => {
                                        match crate::index::writer::delete(&tx, path) {
                                            Ok(deleted) => {
                                                if deleted {
                                                    let _ = tx.commit();
                                                    info!(path = %path.display(), "Deleted document from DB");
                                                }
                                            }
                                            Err(e) => {
                                                error!(path = %path.display(), error = %e, "Failed to delete document from DB");
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to begin delete transaction: {}", e);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Err(_) => break, // Channel closed
            }
        }
    });
}

fn index_single_file(
    conn: &rusqlite::Connection,
    path: &std::path::Path,
    config: &Config,
) -> Result<(), String> {
    use crate::index::writer::{DocumentRecord, EmbeddingRecord};
    use crate::extractor;
    use crate::embed;

    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;

    let is_semantic = {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        crate::extractor::TEXT_EXTENSIONS.contains(&ext.as_str()) || crate::extractor::IMAGE_EXTENSIONS.contains(&ext.as_str())
    };

    if is_semantic && meta.len() > config.max_file_bytes {
        return Ok(());
    }

    let extracted = match extractor::extract(path, config) {
        Ok(Some(e)) => e,
        Ok(None) => return Ok(()),
        Err(e) => return Err(e.to_string()),
    };

    let title = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let checksum = extracted.checksum().to_owned();
    let mime_type = extracted.mime_type().to_owned();
    let size_bytes = meta.len();

    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;

    match extracted {
        extractor::Extracted::Text { chunks, .. } => {
            let mut texts_to_embed = Vec::new();
            for chunk in &chunks {
                if !chunk.trim().is_empty() {
                    texts_to_embed.push(chunk.clone());
                }
            }

            let embeddings = if texts_to_embed.is_empty() {
                None
            } else {
                match embed::embed_text_batch(&texts_to_embed, config) {
                    Ok(vecs) => Some(vecs),
                    Err(e) => {
                        warn!("embedding failed for single file: {}", e);
                        None
                    }
                }
            };

            let mut emb_records = Vec::new();
            if let Some(embeddings) = embeddings {
                for vecs in embeddings {
                    for v in vecs {
                        emb_records.push(EmbeddingRecord { vector: v });
                    }
                }
            }

            let full_body = if chunks.is_empty() {
                None
            } else {
                Some(chunks.join("\n"))
            };

            let doc = DocumentRecord {
                path,
                title: &title,
                body: full_body.as_deref(),
                mime_type: &mime_type,
                checksum: &checksum,
                size_bytes,
            };

            crate::index::writer::upsert(&tx, &doc, &emb_records).map_err(|e| e.to_string())?;
        }
        extractor::Extracted::Image { image, .. } => {
            let embeddings = match embed::embed_image_batch(&[image], config) {
                Ok(mut vecs) => {
                    if vecs.is_empty() {
                        None
                    } else {
                        Some(vecs.remove(0))
                    }
                }
                Err(e) => {
                    warn!("CLIP embedding failed for single image: {}", e);
                    None
                }
            };

            let mut emb_records = Vec::new();
            if let Some(v) = embeddings {
                emb_records.push(EmbeddingRecord { vector: v });
            }

            let doc = DocumentRecord {
                path,
                title: &title,
                body: None,
                mime_type: &mime_type,
                checksum: &checksum,
                size_bytes,
            };

            crate::index::writer::upsert(&tx, &doc, &emb_records).map_err(|e| e.to_string())?;
        }
    }

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}
