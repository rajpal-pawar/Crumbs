//! NDJSON IPC protocol over `stdin` / `stdout`.
//!
//! # Wire format
//!
//! Every message is a single UTF-8 JSON object terminated by `\n`.
//!
//! **Request** (Tauri → daemon, via stdin):
//! ```json
//! {"id": "<uuid4>", "method": "search", "params": { "query": "rust async" }}
//! ```
//!
//! **Response** (daemon → Tauri, via stdout):
//! ```json
//! {"id": "<uuid4>", "ok": true, "result": { ... }}
//! {"id": "<uuid4>", "ok": false, "error": "description"}
//! ```
//!
//! # Concurrency model
//! - The **read loop** runs on a single task and owns `stdin`.
//! - Each incoming request spawns a new `tokio::task` for the handler.
//! - All handlers share a `Mutex<Stdout>` writer so that JSON lines from
//!   concurrent tasks are never interleaved on the wire.
//!
//! # STDOUT IS SACRED
//! The only thing that may write to `stdout` is [`write_response`].
//! No `println!` anywhere in the codebase.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::config::{Config, AtomicConfig};
use crate::handlers;
use crate::index::Database;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// A request arriving from the Tauri host via stdin.
#[derive(Debug, Deserialize)]
pub struct Request {
    /// Correlation ID (UUID v4).  The response echoes this verbatim.
    pub id: String,
    /// Method name: "search" | "status" | "reindex".
    pub method: String,
    /// Method-specific parameters.  Absent params → `null`.
    #[serde(default)]
    pub params: serde_json::Value,
}

/// A response written to stdout for the Tauri host to read.
#[derive(Debug, Serialize)]
pub struct Response {
    /// Echo of the request's correlation ID.
    pub id: String,
    /// `true` on success, `false` on handler error.
    pub ok: bool,
    /// Present on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Present on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Response {
    pub fn success(id: impl Into<String>, result: serde_json::Value) -> Self {
        Response {
            id: id.into(),
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    pub fn failure(id: impl Into<String>, error: impl std::fmt::Display) -> Self {
        Response {
            id: id.into(),
            ok: false,
            result: None,
            error: Some(error.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared writer type alias
// ---------------------------------------------------------------------------

/// A mutex-guarded async handle to stdout.
///
/// Shared between all handler tasks so that NDJSON lines are never
/// interleaved — each `write_response` call locks, writes exactly one line,
/// and unlocks.
pub type SharedWriter = Arc<Mutex<tokio::io::Stdout>>;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// The main IPC run-loop.
///
/// Reads NDJSON lines from `stdin` until EOF (i.e. the Tauri host closes the
/// pipe, which happens when the host exits).  For each valid request line a
/// new Tokio task is spawned to handle it concurrently.
///
/// # Errors
/// Returns on unrecoverable I/O failure.  A malformed JSON line is logged and
/// skipped — it does not terminate the loop.
pub async fn run_loop(config: Config, db: Arc<Database>) -> Result<(), IpcError> {
    let atomic_config = Arc::new(AtomicConfig::new(config.clone()));
    let config = Arc::new(config);
    // db is already Arc<Database> — no re-wrap needed.

    // Wrap stdout in a shared mutex.  Only this writer may touch stdout.
    let writer: SharedWriter = Arc::new(Mutex::new(tokio::io::stdout()));

    // Wrap stdin in an async buffered reader.
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    info!("IPC run-loop ready — waiting for requests");

    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                info!("IPC Request received: {:?}", line);
                let line = line.trim().to_owned();
                if line.is_empty() {
                    continue; // ignore blank lines
                }

                debug!(raw = %line, "received IPC line");

                // Parse the request.
                let request: Request = match serde_json::from_str(&line) {
                    Ok(r) => r,
                    Err(e) => {
                        warn!(error = %e, raw = %line, "failed to parse IPC request — skipping");
                        continue;
                    }
                };

                // Spawn a task per request so the read loop is never blocked
                // by handler work.
                let writer = Arc::clone(&writer);
                let config = Arc::clone(&config);
                let db     = Arc::clone(&db);
                let atomic = Arc::clone(&atomic_config);
                tokio::spawn(async move {
                    dispatch(request, config, db, writer, atomic).await;
                });
            }

            Ok(None) => {
                // EOF — the Tauri host has closed stdin.
                info!("stdin closed (host exited) — shutting down IPC loop");
                break;
            }

            Err(e) => {
                error!(error = %e, "stdin read error");
                return Err(IpcError::Io(e));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Request dispatcher
// ---------------------------------------------------------------------------

async fn dispatch(req: Request, config: Arc<Config>, db: Arc<Database>, writer: SharedWriter, atomic_config: Arc<AtomicConfig>) {
    let id = req.id.clone();
    let method = req.method.as_str();

    let response = match method {
        "search" => {
            handlers::handle_search(req, &config, &db).await
        }
        "status" => {
            handlers::handle_status(req, &config, &db).await
        }
        "reindex" => {
            handlers::handle_reindex(req, &config, &db, writer.clone()).await
        }
        "update_config" => {
            handlers::handle_update_config(req, &atomic_config)
        }
        unknown => {
            warn!(method = %unknown, "unknown IPC method");
            Response::failure(&id, format!("unknown method: {unknown}"))
        }
    };

    write_response(&writer, response).await;
}

// ---------------------------------------------------------------------------
// Response writer — the ONE place that writes to stdout
// ---------------------------------------------------------------------------

/// Serialise `response` as a single NDJSON line and write it to stdout.
///
/// Acquires the mutex, writes `<json>\n`, and releases.  The mutex guarantees
/// that lines from concurrent tasks never interleave.
pub async fn write_response(writer: &SharedWriter, response: Response) {
    let mut line = match serde_json::to_string(&response) {
        Ok(s) => s,
        Err(e) => {
            error!(error = %e, "failed to serialise response — dropping");
            return;
        }
    };
    line.push('\n');

    use tokio::io::AsyncWriteExt;
    let mut stdout_guard = writer.lock().await;
    let stdout = &mut *stdout_guard;
    if let Err(e) = stdout.write_all(line.as_bytes()).await {
        error!(error = %e, "failed to write response to stdout");
    }
    // Flush ensures the Tauri host receives the line promptly.
    if let Err(e) = stdout.flush().await {
        error!(error = %e, "failed to flush stdout");
    }
}

pub async fn write_raw_event(writer: &SharedWriter, event: serde_json::Value) {
    let mut line = match serde_json::to_string(&event) {
        Ok(s) => s,
        Err(e) => {
            error!(error = %e, "failed to serialise raw event — dropping");
            return;
        }
    };
    line.push('\n');

    use tokio::io::AsyncWriteExt;
    let mut stdout_guard = writer.lock().await;
    let stdout = &mut *stdout_guard;
    let _ = stdout.write_all(line.as_bytes()).await;
    let _ = stdout.flush().await;
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    #[error("stdin I/O error: {0}")]
    Io(#[from] std::io::Error),
}
