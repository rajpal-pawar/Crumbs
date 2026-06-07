//! Configuration loading and platform data-directory resolution.
//!
//! # Memory budget
//! The target hardware has 8 GB RAM.  We reserve ≈1 GB for the daemon
//! process (SQLite page cache + ONNX model tensors + working set).  To
//! prevent a single large file from blowing the budget we enforce a
//! **15 MB per-file cap** before feeding anything into the embedding pipeline.
//!
//! Files larger than [`Config::max_file_bytes`] are skipped with a warning
//! during indexing.
//!
//! # Data directories
//! | Platform | Default path |
//! |---|---|
//! | Windows  | `%APPDATA%\Crumbs`  (`C:\Users\<user>\AppData\Roaming\Crumbs`) |
//! | Linux    | `$XDG_DATA_HOME/crumbs` (falls back to `~/.local/share/crumbs`) |
//! | macOS    | `~/Library/Application Support/crumbs` |

use std::path::PathBuf;

/// 15 MB — files above this limit are skipped during indexing.
///
/// Rationale: the ONNX model for our embedding pipeline processes text in
/// 512-token chunks.  A 15 MB text file is already extremely large; reading
/// and chunking anything bigger risks OOM on the i7-6500U / 8 GB target.
pub const MAX_FILE_BYTES: u64 = 15 * 1024 * 1024; // 15 MiB

/// Runtime configuration for the daemon.
#[derive(Debug, Clone)]
pub struct Config {
    /// Root directory for all Crumbs data (database, model cache, index).
    pub data_dir: PathBuf,

    /// Per-file ingestion size cap in bytes.
    pub max_file_bytes: u64,

    /// Maximum bytes to read from a text file for the embedding pipeline.
    /// Reading beyond this is wasteful: MiniLM has a 512-token context window
    /// which corresponds to roughly 100 KB of prose.  Default: 128 KB.
    pub text_read_limit_bytes: usize,

    /// Number of intra-op threads for the ONNX runtime.
    /// On the i7-6500U (2 cores / 4 threads), 2 is the sweet spot:
    /// enough parallelism to utilise both physical cores without completely
    /// starving the IPC and UI threads.
    pub onnx_intra_threads: i16,

    /// Batch size for the embedding pipeline (documents per ONNX session).
    /// Larger batches amortise session startup cost; smaller batches reduce
    /// peak RAM.  Default: 32.
    pub embed_batch_size: usize,

    /// Directories the watcher should monitor for changes.
    pub watch_dirs: Vec<PathBuf>,

    /// Maximum number of concurrent indexing tasks.
    /// Kept low on the target hardware to leave room for the Tauri UI.
    pub index_parallelism: usize,
}

impl Config {
    /// Load configuration.
    ///
    /// Currently reads from a hard-coded set of defaults.  In a later phase
    /// this will merge in values from a TOML file inside `data_dir`.
    pub fn load() -> Result<Self, ConfigError> {
        let data_dir = resolve_data_dir()?;

        // Ensure the data directory exists so later code can write files
        // without having to handle the creation themselves.
        std::fs::create_dir_all(&data_dir).map_err(|e| {
            ConfigError::IoError {
                path: data_dir.clone(),
                source: e,
            }
        })?;

        Ok(Config {
            data_dir,
            max_file_bytes: MAX_FILE_BYTES,
            // 128 KB covers the full context window of MiniLM (512 tokens ≈
            // 2000 words ≈ ~12 KB, so 128 KB is generous without being wasteful).
            text_read_limit_bytes: 128 * 1024,
            // 2 intra-op threads: uses both physical cores of the i7-6500U
            // while leaving the HT siblings for IPC / OS tasks.
            onnx_intra_threads: 2,
            // 32 docs/batch: ~few MB peak RAM per batch, fast enough
            // amortisation of ONNX session startup (~200 ms).
            embed_batch_size: 32,
            watch_dirs: default_watch_dirs(),
            index_parallelism: 1,
        })
    }

    /// Convenience: path to the SQLite database file.
    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("crumbs.db")
    }

    /// Convenience: directory where downloaded ONNX models are cached.
    pub fn model_cache_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn resolve_data_dir() -> Result<PathBuf, ConfigError> {
    // Hardcoding to your project directory to avoid permission/lock issues
    Ok(PathBuf::from("/home/rajpalsinghpanwar/Crumbs/db"))
}

fn default_watch_dirs() -> Vec<PathBuf> {
    // Only target the folders you actually need
    vec![
        PathBuf::from("/home/rajpalsinghpanwar/Crumbs/data"),
    ]
}
// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot determine platform data directory")]
    NoDataDir,

    #[error("I/O error at {path}: {source}")]
    IoError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

// Make ConfigError implement Display for use in main.rs with %e formatting.
// (thiserror already derives Display via the #[error(...)] attrs above.)
