//! `model_download.rs` — Auto-download models from GitHub releases.
//!
//! On daemon startup, if the models directory is missing required files,
//! this module downloads `models.zip` from the Crumbs GitHub release,
//! extracts it, and places the files in the platform model cache directory.
//!
//! This is the **same** zip that the Tauri desktop app downloads via
//! `start_model_download`, ensuring a single source of truth for model
//! assets across development and production flows.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn, error};

/// GitHub release URL for the models zip bundle.
const MODELS_ZIP_URL: &str =
    "https://github.com/rajpal-pawar/Crumbs/releases/download/crumbs-v1.0.0/models.zip";

/// Required model files that must be present for the daemon to function.
/// If any of these are missing, a download is triggered.
const REQUIRED_FILES: &[&str] = &[
    crate::embed::BGE_FILENAME,
    crate::embed::TOKENIZER_FILENAME,
    crate::embed::CLIP_FILENAME,
    crate::embed::CLIP_TEXT_FILENAME,
    crate::embed::CLIP_TOKENIZER_FILENAME,
];

/// Check whether all required model files exist in `models_dir`.
pub fn models_present(models_dir: &Path) -> bool {
    if !models_dir.exists() {
        return false;
    }
    REQUIRED_FILES.iter().all(|f| models_dir.join(f).exists())
}

/// Download and extract the models zip if any required files are missing.
///
/// This function is idempotent — if all files already exist it returns
/// immediately.  On failure it logs the error and returns `Err` so the
/// caller can degrade gracefully (search will work via BM25 only).
pub async fn ensure_models(models_dir: &Path) -> Result<(), String> {
    if models_present(models_dir) {
        info!("All required model files present — skipping download");
        return Ok(());
    }

    info!("Model files missing — downloading from GitHub release…");
    info!("URL: {}", MODELS_ZIP_URL);

    tokio::fs::create_dir_all(models_dir)
        .await
        .map_err(|e| format!("Cannot create models directory: {e}"))?;

    let tmp_zip = models_dir.with_file_name("models-daemon.zip.tmp");

    // ------------------------------------------------------------------
    // 1. Stream-download the zip to a temp file
    // ------------------------------------------------------------------
    let client = reqwest::Client::builder()
        .user_agent("Crumbs/1.0")
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let response = client
        .get(MODELS_ZIP_URL)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned {}: {}",
            response.status().as_u16(),
            response.status().canonical_reason().unwrap_or("unknown")
        ));
    }

    let total_bytes = response.content_length();
    info!("Response OK — total bytes: {:?}", total_bytes);

    let mut file = tokio::fs::File::create(&tmp_zip)
        .await
        .map_err(|e| format!("Cannot create temp file: {e}"))?;

    let mut downloaded: u64 = 0;
    let mut last_pct: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Download stream error: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("File write error: {e}"))?;
        downloaded += chunk.len() as u64;

        // Log progress every 10%.
        if let Some(total) = total_bytes {
            if total > 0 {
                let pct = (downloaded * 100) / total;
                if pct >= last_pct + 10 {
                    last_pct = pct;
                    info!(
                        "Download progress: {}% ({:.1} / {:.1} MB)",
                        pct,
                        downloaded as f64 / 1_048_576.0,
                        total as f64 / 1_048_576.0
                    );
                }
            }
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("File flush error: {e}"))?;

    info!("Download complete — extracting…");

    // ------------------------------------------------------------------
    // 2. Extract the zip on a blocking thread
    // ------------------------------------------------------------------
    let tmp_zip_clone = tmp_zip.clone();
    let models_dir_owned = models_dir.to_path_buf();

    tokio::task::spawn_blocking(move || -> Result<(), String> {
        let zip_file = std::fs::File::open(&tmp_zip_clone)
            .map_err(|e| format!("Cannot open archive: {e}"))?;
        let mut archive = zip::ZipArchive::new(zip_file)
            .map_err(|e| format!("Invalid zip archive: {e}"))?;

        archive
            .extract(&models_dir_owned)
            .map_err(|e| format!("Extraction failed: {e}"))?;

        // Flatten nested `models/` directory if the zip contains one.
        let nested = models_dir_owned.join("models");
        if nested.exists() && nested.is_dir() {
            info!("Flattening nested 'models' directory…");
            if let Ok(entries) = std::fs::read_dir(&nested) {
                for entry in entries.flatten() {
                    let src = entry.path();
                    if let Some(name) = src.file_name() {
                        let dest = models_dir_owned.join(name);
                        if let Err(e) = std::fs::rename(&src, &dest) {
                            warn!("Failed to move {:?} → {:?}: {}", src, dest, e);
                        }
                    }
                }
            }
            let _ = std::fs::remove_dir(&nested);
        }

        Ok(())
    })
    .await
    .map_err(|e| format!("spawn_blocking failed: {e}"))??;

    // ------------------------------------------------------------------
    // 3. Clean up temp file
    // ------------------------------------------------------------------
    if let Err(e) = tokio::fs::remove_file(&tmp_zip).await {
        warn!("Failed to delete temp zip: {e}");
    }

    info!("Models extracted successfully to {:?}", models_dir);
    Ok(())
}
