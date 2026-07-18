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
    paused: Option<bool>,
    handle: State<'_, Arc<DaemonHandle>>,
) -> Result<Value, String> {
    let mut params = serde_json::Map::new();
    if let Some(bs) = batch_size {
        params.insert("batch_size".into(), json!(bs));
    }
    if let Some(t) = threads {
        params.insert("threads".into(), json!(t));
    }
    if let Some(p) = paused {
        params.insert("paused".into(), json!(p));
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
// list_indexed_files
// ---------------------------------------------------------------------------

/// List all indexed documents from the daemon's database.
///
/// Returns `{ documents: [...], total: N }`.
///
/// **Front-end usage:**
/// ```typescript
/// const result = await invoke<{ documents: IndexedFile[], total: number }>('list_indexed_files');
/// ```
#[tauri::command]
pub async fn list_indexed_files(
    handle: State<'_, Arc<DaemonHandle>>,
) -> Result<Value, String> {
    let response = daemon::send_request(&handle, "list_documents", Value::Null)
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

// ---------------------------------------------------------------------------
// app_models_dir  (private helper)
// ---------------------------------------------------------------------------

/// Returns the platform-appropriate models directory:
///   Windows : `%LOCALAPPDATA%\com.crumbs.app\models\`
///   Linux   : `$XDG_DATA_HOME/com.crumbs.app/models/`  (fallback: `~/.local/share/…`)
///   macOS   : `~/Library/Application Support/com.crumbs.app/models/`
fn app_models_dir() -> Result<std::path::PathBuf, String> {
    // `dirs` is already available transitively through Tauri; we use its own
    // data_local_dir() which maps correctly on every platform.
    let base = dirs::data_local_dir()
        .ok_or_else(|| "Cannot determine system data directory".to_string())?;
    Ok(base.join("com.crumbs.app").join("models"))
}

#[tauri::command]
pub async fn download_models(url: String, app: AppHandle) -> Result<String, String> {
    match do_download_models(url, &app).await {
        Ok(_) => Ok("done".to_string()),
        Err(e) => {
            use tauri::Emitter;
            let _ = app.emit("download-error", e.clone());
            Err(e)
        }
    }
}

async fn do_download_models(url: String, app: &AppHandle) -> Result<(), String> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    use tauri::Emitter;

    tracing::info!("download_models: starting — {}", url);

    let models_dir = app_models_dir()?;
    tokio::fs::create_dir_all(&models_dir)
        .await
        .map_err(|e| format!("Cannot create models directory: {e}"))?;

    let tmp_zip = models_dir.with_file_name("models.zip.tmp");

    let client = reqwest::Client::builder()
        .user_agent("Crumbs/1.0")
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Server returned {}: {}", response.status().as_u16(), response.status().canonical_reason().unwrap_or("unknown")));
    }

    let total_bytes = response.content_length();
    tracing::info!("download_models: response OK — total bytes: {:?}", total_bytes);

    let mut file = tokio::fs::File::create(&tmp_zip)
        .await
        .map_err(|e| format!("Cannot create temp file: {e}"))?;

    let mut downloaded: u64 = 0;
    let mut last_pct: f32 = 0.0;
    let mut stream = response.bytes_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Download stream error: {e}"))?;
        file.write_all(&chunk).await.map_err(|e| format!("File write error: {e}"))?;
        downloaded += chunk.len() as u64;

        if let Some(total) = total_bytes {
            let pct = (downloaded as f32 / total as f32) * 100.0;
            if pct - last_pct >= 1.0 {
                last_pct = pct;
                let _ = app.emit("download-progress", pct);
            }
        } else {
            let _ = app.emit("download-progress", -1.0f32);
        }
    }
    file.flush().await.map_err(|e| format!("File flush error: {e}"))?;

    tracing::info!("download_models: download complete — extracting to {:?}", models_dir);

    let tmp_zip_clone = tmp_zip.clone();
    let models_dir_clone = models_dir.clone();

    tokio::task::spawn_blocking(move || -> Result<(), String> {
        let zip_file = std::fs::File::open(&tmp_zip_clone)
            .map_err(|e| format!("Cannot open archive for extraction: {e}"))?;
        let mut archive = zip::ZipArchive::new(zip_file)
            .map_err(|e| format!("Invalid zip archive: {e}"))?;
        archive.extract(&models_dir_clone)
            .map_err(|e| format!("Extraction failed: {e}"))?;

        let nested_models_dir = models_dir_clone.join("models");
        if nested_models_dir.exists() && nested_models_dir.is_dir() {
            tracing::info!("Flattening nested 'models' directory...");
            if let Ok(entries) = std::fs::read_dir(&nested_models_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(file_name) = path.file_name() {
                        let dest = models_dir_clone.join(file_name);
                        if let Err(e) = std::fs::rename(&path, &dest) {
                            tracing::warn!("Failed to move {:?} to {:?}: {}", path, dest, e);
                        }
                    }
                }
            }
            if let Err(e) = std::fs::remove_dir(&nested_models_dir) {
                tracing::warn!("Failed to remove empty nested directory: {}", e);
            }
        }

        Ok(())
    })
    .await
    .map_err(|e| format!("spawn_blocking (extract) failed: {e}"))??;

    tracing::info!("download_models: extraction complete");

    if let Err(e) = tokio::fs::remove_file(&tmp_zip).await {
        tracing::warn!("download_models: failed to delete temp zip: {e}");
    }

    let _ = app.emit("download-progress", 100.0f32);
    tracing::info!("download_models: done — models at {:?}", models_dir);
    Ok(())
}

// ---------------------------------------------------------------------------
// check_models_exist
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn check_models_exist() -> Result<bool, String> {
    let dir = app_models_dir()?;
    if !dir.exists() {
        return Ok(false);
    }
    let mut has_files = false;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.path().is_file() {
                has_files = true;
                break;
            }
        }
    }
    Ok(has_files)
}

// ---------------------------------------------------------------------------
// start_model_download
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_model_download(app: AppHandle) -> Result<String, String> {
    let url = "https://github.com/rajpal-pawar/Crumbs/releases/download/crumbs-v1.0.0/models.zip".to_string();
    download_models(url, app).await
}

