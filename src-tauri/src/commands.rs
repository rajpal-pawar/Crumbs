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
use tauri::AppHandle;

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

// ---------------------------------------------------------------------------
// update_engine_config
// ---------------------------------------------------------------------------

/// Update runtime engine tuning parameters (batch size, CPU threads).
///
/// These are forwarded to the daemon via IPC.  The daemon applies
/// the new values to its AtomicConfig, which the indexing loop reads
/// on every batch iteration — no restart required.
///
/// **Front-end usage:**
/// ```typescript
/// await invoke("update_engine_config", { batchSize: 10, threads: 4 });
/// ```
#[tauri::command]
pub async fn update_engine_config(
    batch_size: Option<u32>,
    threads: Option<u32>,
    handle: State<'_, Arc<DaemonHandle>>,
) -> Result<Value, String> {
    let mut params = serde_json::Map::new();
    if let Some(bs) = batch_size {
        params.insert("batch_size".into(), json!(bs));
    }
    if let Some(t) = threads {
        params.insert("threads".into(), json!(t));
    }

    let response = daemon::send_request(
        &handle,
        "update_config",
        Value::Object(params),
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
// get_onboarding_status
// ---------------------------------------------------------------------------

/// Check if the user has completed onboarding.
///
/// Returns `{ is_onboarded: bool, watch_dirs: string[] }`.
///
/// **Front-end usage:**
/// ```typescript
/// const status = await invoke<{ is_onboarded: boolean, watch_dirs: string[] }>('get_onboarding_status');
/// ```
#[tauri::command]
pub async fn get_onboarding_status(
    handle: State<'_, Arc<DaemonHandle>>,
) -> Result<Value, String> {
    let response = daemon::send_request(&handle, "get_config", Value::Null)
        .await
        .map_err(|e| e.to_string())?;

    if response.ok {
        Ok(response.result.unwrap_or(Value::Null))
    } else {
        Err(response.error.unwrap_or_else(|| "unknown daemon error".to_string()))
    }
}

// ---------------------------------------------------------------------------
// update_monitored_folders
// ---------------------------------------------------------------------------

/// Update the list of monitored folders and persist to daemon config.
///
/// This command triggers a reindex of the new folder set and cleans up
/// embeddings for removed folders.
///
/// **Front-end usage:**
/// ```typescript
/// await invoke('update_monitored_folders', { folders: ['/home/user/Documents', '/home/user/Projects'], isOnboarded: true });
/// ```
#[tauri::command]
pub async fn update_monitored_folders(
    folders: Vec<String>,
    is_onboarded: bool,
    handle: State<'_, Arc<DaemonHandle>>,
) -> Result<Value, String> {
    let response = daemon::send_request(
        &handle,
        "update_folders",
        json!({
            "folders": folders,
            "is_onboarded": is_onboarded,
        }),
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
// select_folders_dialog
// ---------------------------------------------------------------------------

/// Open a native OS folder-picker dialog and return the selected paths.
///
/// Returns an array of absolute path strings, or an empty array if the user
/// cancelled the dialog.
///
/// **Front-end usage:**
/// ```typescript
/// const paths: string[] = await invoke('select_folders_dialog');
/// ```
#[tauri::command]
pub async fn select_folders_dialog(
    app: AppHandle,
) -> Result<Vec<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let (tx, rx) = tokio::sync::oneshot::channel();

    app.dialog()
        .file()
        .set_title("Select Folders to Index")
        .pick_folders(move |folders| {
            let paths = match folders {
                Some(paths) => paths
                    .into_iter()
                    .filter_map(|p| {
                        p.into_path()
                            .ok()
                            .map(|path| path.to_string_lossy().into_owned())
                    })
                    .collect::<Vec<_>>(),
                None => vec![],
            };
            let _ = tx.send(paths);
        });

    rx.await.map_err(|_| "dialog channel closed".to_string())
}
