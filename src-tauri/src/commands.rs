//! Tauri command handlers — the bridge between the WebView front-end and the
//! daemon IPC layer.
//!
//! Each `#[tauri::command]` function is exposed to the front-end via
//! `window.__TAURI__.core.invoke("command_name", { ...params })`.
//!
//! All commands delegate to [`daemon::send_request`] and map errors to
//! `String` for the WebView (Tauri serialises the `Result<T, String>` for us).

use std::sync::Arc;

use serde_json::{json, Value};
use tauri::State;

use crate::daemon::{self, DaemonHandle};

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

/// Search the semantic index.
///
/// **Front-end usage:**
/// ```typescript
/// const results = await invoke<SearchResult[]>("search", { query: "rust async" });
/// ```
#[tauri::command]
pub async fn search(
    query: String,
    handle: State<'_, Arc<DaemonHandle>>,
) -> Result<Value, String> {
    let response = daemon::send_request(
        &handle,
        "search",
        json!({ "query": query }),
    )
    .await
    .map_err(|e| e.to_string())?;

    if response.ok {
        Ok(response.result.unwrap_or(Value::Null))
    } else {
        Err(response.error.unwrap_or_else(|| "unknown daemon error".to_string()))
    }
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

/// Fetch daemon health and configuration snapshot.
///
/// **Front-end usage:**
/// ```typescript
/// const status = await invoke<DaemonStatus>("status");
/// ```
#[tauri::command]
pub async fn status(
    handle: State<'_, Arc<DaemonHandle>>,
) -> Result<Value, String> {
    let response = daemon::send_request(&handle, "status", Value::Null)
        .await
        .map_err(|e| e.to_string())?;

    if response.ok {
        Ok(response.result.unwrap_or(Value::Null))
    } else {
        Err(response.error.unwrap_or_else(|| "unknown daemon error".to_string()))
    }
}

// ---------------------------------------------------------------------------
// reindex
// ---------------------------------------------------------------------------

/// Trigger a full re-index of all configured watch directories.
///
/// This command returns immediately after the daemon acknowledges the request.
/// Progress updates will be delivered via Tauri events in a later phase.
///
/// **Front-end usage:**
/// ```typescript
/// await invoke("reindex");
/// ```
#[tauri::command]
pub async fn reindex(
    handle: State<'_, Arc<DaemonHandle>>,
) -> Result<Value, String> {
    let response = daemon::send_request(&handle, "reindex", Value::Null)
        .await
        .map_err(|e| e.to_string())?;

    if response.ok {
        Ok(response.result.unwrap_or(Value::Null))
    } else {
        Err(response.error.unwrap_or_else(|| "unknown daemon error".to_string()))
    }
}

// ---------------------------------------------------------------------------
// open_file
// ---------------------------------------------------------------------------

/// Opens a file using the operating system's default application.
///
/// **Front-end usage:**
/// ```typescript
/// await invoke("open_file", { path: "/path/to/file.txt" });
/// ```
#[tauri::command]
pub fn open_file(path: String) -> Result<(), String> {
    tracing::info!("Opening file via OS: {}", path);

    #[cfg(target_os = "windows")]
    let result = std::process::Command::new("cmd")
        .args(["/C", "start", "", &path])
        .spawn();

    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open")
        .arg(&path)
        .spawn();

    #[cfg(target_os = "linux")]
    let result = std::process::Command::new("xdg-open")
        .arg(&path)
        .spawn();

    result.map(|_| ()).map_err(|e| e.to_string())
}
