//! Daemon lifecycle management and stdout IPC router.
//!
//! # Responsibilities
//!
//! - [`launch`] — spawns the `crumbs-daemon` sidecar binary via Tauri's
//!   `Shell` plugin, wires its `stdin`/`stdout`, and starts the router task.
//! - **Stdout router** — a background Tokio task reads NDJSON lines from the
//!   daemon's stdout, parses them, and routes each response to the waiting
//!   caller via a one-shot channel keyed by the correlation UUID.
//! - [`send_request`] — serialises a request, writes it to the daemon's
//!   stdin, registers a one-shot receiver, and waits up to 30 seconds for
//!   the daemon to reply.
//!
//! # Concurrency model
//!
//! ```text
//!  Tauri command (async task)
//!       │
//!       │ send_request(method, params)
//!       │   1. Generates UUID
//!       │   2. Inserts oneshot::Sender into PENDING map
//!       │   3. Writes NDJSON line to daemon stdin
//!       │   4. Awaits oneshot::Receiver (30 s timeout)
//!       ▼
//!  ┌──────────────┐   stdout   ┌─────────────────────┐
//!  │ crumbs-daemon│ ─────────► │  stdout router task │
//!  └──────────────┘            │  (background loop)  │
//!                              │  Parses JSON, looks  │
//!                              │  up UUID in PENDING, │
//!                              │  sends on oneshot    │
//!                              └─────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Manager};
use tauri_plugin_shell::ShellExt;
use tokio::sync::{oneshot, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Wire types (must mirror crumbs-daemon's ipc.rs)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Request {
    id: String,
    method: String,
    params: Value,
}

#[derive(Deserialize, Debug)]
pub struct Response {
    pub id: String,
    pub ok: bool,
    pub result: Option<Value>,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Shared daemon state
// ---------------------------------------------------------------------------

/// Map of pending request IDs → one-shot senders waiting for a response.
type PendingMap = Arc<Mutex<HashMap<String, oneshot::Sender<Response>>>>;

/// Handle to the daemon's stdin writer and the pending-response map.
/// Stored in Tauri's managed state so commands can call `send_request`.
pub struct DaemonHandle {
    /// The daemon child process (guarded by mutex for shared access to its write method).
    child: Mutex<tauri_plugin_shell::process::CommandChild>,
    /// In-flight requests waiting for a response from the daemon.
    pending: PendingMap,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Launch the `crumbs-daemon` sidecar, start the stdout router, and register
/// the [`DaemonHandle`] in Tauri's managed state.
///
/// This function is called once from `lib.rs`'s setup hook.
pub async fn launch(app: AppHandle) -> Result<(), DaemonError> {
    info!("launching crumbs-daemon sidecar");

    let shell = app.shell();

    // `sidecar("crumbs-daemon")` resolves to the binary that was placed in
    // `src-tauri/binaries/crumbs-daemon-<target-triple>[.exe]` by the build
    // script and referenced in `tauri.conf.json`'s `bundle.externalBin`.
    let (mut rx, child) = shell
        .sidecar("crumbs-daemon")
        .map_err(|e| DaemonError::Spawn(e.to_string()))?
        .spawn()
        .map_err(|e| DaemonError::Spawn(e.to_string()))?;

    info!("crumbs-daemon sidecar spawned");

    let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
    let pending_router = Arc::clone(&pending);

    // -----------------------------------------------------------------------
    // Stdout router — runs for the lifetime of the daemon.
    // -----------------------------------------------------------------------
    tokio::spawn(async move {
        use tauri_plugin_shell::process::CommandEvent;

        info!("stdout router started");

        loop {
            match rx.recv().await {
                Some(CommandEvent::Stdout(line_bytes)) => {
                    let line = String::from_utf8_lossy(&line_bytes);
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    debug!(raw = %line, "daemon stdout line received");

                    match serde_json::from_str::<Response>(line) {
                        Ok(response) => {
                            let id = response.id.clone();
                            let mut map = pending_router.lock().await;
                            if let Some(sender) = map.remove(&id) {
                                let _ = sender.send(response);
                            } else {
                                warn!(id = %id, "received response for unknown request ID");
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, raw = %line, "failed to parse daemon stdout as NDJSON");
                        }
                    }
                }

                Some(CommandEvent::Stderr(line_bytes)) => {
                    // Daemon logs arrive here — forward to our tracing.
                    let line = String::from_utf8_lossy(&line_bytes);
                    info!(daemon_log = %line.trim(), "daemon stderr");
                }

                Some(CommandEvent::Terminated(status)) => {
                    error!(status = ?status, "daemon process terminated unexpectedly");
                    break;
                }

                Some(_) => {} // ignore other events (Error, etc.)

                None => {
                    info!("daemon stdout stream closed");
                    break;
                }
            }
        }

        info!("stdout router exiting");
    });

    // -----------------------------------------------------------------------
    // Register the DaemonHandle in Tauri managed state.
    // -----------------------------------------------------------------------
    // We wrap the CommandChild in a Mutex so multiple command invocations can share it.
    app.manage(Arc::new(DaemonHandle {
        child: Mutex::new(child),
        pending,
    }));

    info!("DaemonHandle registered in Tauri managed state");
    Ok(())
}

/// Send a request to the daemon and await its response (30 s timeout).
///
/// # Errors
/// - [`DaemonError::Timeout`]  — no response within 30 seconds.
/// - [`DaemonError::Send`]     — failed to write to stdin.
/// - [`DaemonError::Cancelled`]— the one-shot channel was dropped (router exited).
pub async fn send_request(
    handle: &Arc<DaemonHandle>,
    method: &str,
    params: Value,
) -> Result<Response, DaemonError> {
    let id = Uuid::new_v4().to_string();

    let request = Request {
        id: id.clone(),
        method: method.to_owned(),
        params,
    };

    let mut line = serde_json::to_string(&request)
        .map_err(|e| DaemonError::Send(e.to_string()))?;
    line.push('\n');

    // Register the one-shot channel BEFORE writing to stdin to avoid a race
    // where the daemon replies before we've registered the receiver.
    let (tx, rx) = oneshot::channel::<Response>();
    {
        let mut map = handle.pending.lock().await;
        map.insert(id.clone(), tx);
    }

    // Write the request to the daemon's stdin.
    {
        handle
            .child
            .lock()
            .await
            .write(line.as_bytes())
            .map_err(|e| DaemonError::Send(e.to_string()))?;
    }

    debug!(id = %id, method = %method, "request sent to daemon");

    // Wait for the router to deliver the response — up to 30 seconds.
    match tokio::time::timeout(Duration::from_secs(30), rx).await {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(_)) => Err(DaemonError::Cancelled),
        Err(_) => {
            // Clean up the orphaned entry from the pending map.
            let mut map = handle.pending.lock().await;
            map.remove(&id);
            Err(DaemonError::Timeout)
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    #[error("failed to spawn daemon sidecar: {0}")]
    Spawn(String),

    #[error("failed to send request to daemon: {0}")]
    Send(String),

    #[error("daemon did not respond within 30 seconds")]
    Timeout,

    #[error("daemon response channel was cancelled (daemon may have exited)")]
    Cancelled,
}
