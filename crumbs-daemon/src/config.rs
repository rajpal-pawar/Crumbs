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
//! # Dynamic configuration
//! [`AtomicConfig`] wraps the hot-path tuning knobs (`embed_batch_size`,
//! `onnx_intra_threads`) in atomics so the frontend can update them at
//! runtime without restarting the daemon.
//!
//! # Data directories
//! | Platform | Default path |
//! |---|---|
//! | Windows  | `%APPDATA%\Crumbs`  (`C:\Users\<user>\AppData\Roaming\Crumbs`) |
//! | Linux    | `$XDG_DATA_HOME/crumbs` (falls back to `~/.local/share/crumbs`) |
//! | macOS    | `~/Library/Application Support/crumbs` |

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicI16, Ordering};
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};
use tracing::info;

/// 15 MB — files above this limit are skipped during indexing.
///
/// Rationale: the ONNX model for our embedding pipeline processes text in
/// 512-token chunks.  A 15 MB text file is already extremely large; reading
/// and chunking anything bigger risks OOM on the i7-6500U / 8 GB target.
pub const MAX_FILE_BYTES: u64 = 15 * 1024 * 1024; // 15 MiB

/// Runtime configuration for the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Root directory for all Crumbs data (database, model cache, index).
    pub data_dir: PathBuf,

    /// Per-file ingestion size cap in bytes.
    pub max_file_bytes: u64,

    /// Maximum bytes to read from a text file for the embedding pipeline.
    /// Reading beyond this is wasteful: BGE-small-en-v1.5 has a 512-token context window
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

    /// Whether the user has completed the first-run onboarding flow.
    /// When `false`, the daemon skips automatic crawling and waits for
    /// explicit IPC commands to set up folders.
    pub is_onboarded: bool,
}

// ---------------------------------------------------------------------------
// Atomic runtime configuration (hot-path knobs)
// ---------------------------------------------------------------------------

/// Thread-safe wrapper around the tuning parameters that the frontend can
/// update via the `update_engine_config` Tauri command.  The indexing loops
/// read these atomics on every batch iteration, so changes take effect
/// immediately without restarting the daemon.
pub struct AtomicConfig {
    pub embed_batch_size: AtomicUsize,
    pub onnx_intra_threads: AtomicI16,
    pub index_parallelism: AtomicUsize,
    pub is_onboarded: AtomicBool,
    /// Mutable watch_dirs — updated when user adds/removes folders.
    pub watch_dirs: RwLock<Vec<PathBuf>>,
    /// The base (immutable) config for paths, limits, etc.
    pub base: Config,
}

impl AtomicConfig {
    pub fn new(config: Config) -> Self {
        AtomicConfig {
            embed_batch_size: AtomicUsize::new(config.embed_batch_size),
            onnx_intra_threads: AtomicI16::new(config.onnx_intra_threads),
            index_parallelism: AtomicUsize::new(config.index_parallelism),
            is_onboarded: AtomicBool::new(config.is_onboarded),
            watch_dirs: RwLock::new(config.watch_dirs.clone()),
            base: config,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.embed_batch_size.load(Ordering::Relaxed)
    }

    pub fn threads(&self) -> i16 {
        self.onnx_intra_threads.load(Ordering::Relaxed)
    }

    pub fn set_batch_size(&self, val: usize) {
        let clamped = val.clamp(1, 50);
        self.embed_batch_size.store(clamped, Ordering::Relaxed);
        info!("batch_size updated to {}", clamped);
    }

    pub fn set_threads(&self, val: i16) {
        let clamped = val.clamp(1, 16);
        self.onnx_intra_threads.store(clamped, Ordering::Relaxed);
        info!("onnx_intra_threads updated to {}", clamped);
    }
}

impl std::fmt::Debug for AtomicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicConfig")
            .field("embed_batch_size", &self.batch_size())
            .field("onnx_intra_threads", &self.threads())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Per-directory state tracking
// ---------------------------------------------------------------------------

/// Lifecycle state of a single watched root directory during indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DirState {
    Queued,
    Scanning,
    Indexing,
    Completed,
}

impl std::fmt::Display for DirState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirState::Queued    => write!(f, "queued"),
            DirState::Scanning  => write!(f, "scanning"),
            DirState::Indexing  => write!(f, "indexing"),
            DirState::Completed => write!(f, "completed"),
        }
    }
}

/// Status entry for a single watched directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirStatus {
    pub path: String,
    pub state: DirState,
}

/// Thread-safe directory status registry.
/// The producer thread updates individual entries; the progress reporter
/// reads a snapshot for the JSON payload.
pub struct DirStatusRegistry {
    dirs: RwLock<Vec<DirStatus>>,
}

impl DirStatusRegistry {
    /// Create a new registry pre-populated with all watch dirs in `Queued` state.
    pub fn new(watch_dirs: &[PathBuf]) -> Self {
        let dirs = watch_dirs
            .iter()
            .map(|p| DirStatus {
                path: Self::display_path(p),
                state: DirState::Queued,
            })
            .collect();
        DirStatusRegistry {
            dirs: RwLock::new(dirs),
        }
    }

    /// Update the state of a specific directory by path.
    pub fn set_state(&self, path: &PathBuf, state: DirState) {
        let display = Self::display_path(path);
        if let Ok(mut dirs) = self.dirs.write() {
            if let Some(entry) = dirs.iter_mut().find(|d| d.path == display) {
                entry.state = state;
            }
        }
    }

    /// Take a snapshot of all directory statuses for JSON serialisation.
    pub fn snapshot(&self) -> Vec<DirStatus> {
        self.dirs.read().map(|d| d.clone()).unwrap_or_default()
    }

    /// Convert a PathBuf to a tilde-prefixed display string.
    fn display_path(p: &PathBuf) -> String {
        if let Some(home) = dirs::home_dir() {
            if let Ok(suffix) = p.strip_prefix(&home) {
                return format!("~/{}", suffix.display());
            }
        }
        p.display().to_string()
    }
}

/// Persisted user preferences (watch_dirs + onboarding flag).
/// Stored as JSON in `<data_dir>/config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedConfig {
    watch_dirs: Vec<PathBuf>,
    is_onboarded: bool,
}

impl Config {
    /// Load configuration.
    ///
    /// Reads persistent user preferences (watch_dirs, is_onboarded) from
    /// `<data_dir>/config.json`, falling back to safe defaults if the file
    /// doesn't exist yet.
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

        // Load persisted user preferences or create defaults.
        let config_path = data_dir.join("config.json");
        let persisted = if config_path.exists() {
            match std::fs::read_to_string(&config_path) {
                Ok(contents) => {
                    match serde_json::from_str::<PersistedConfig>(&contents) {
                        Ok(p) => {
                            info!("Loaded persisted config: {:?}", p);
                            p
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse config.json, using defaults: {}", e);
                            PersistedConfig { watch_dirs: vec![], is_onboarded: false }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read config.json, using defaults: {}", e);
                    PersistedConfig { watch_dirs: vec![], is_onboarded: false }
                }
            }
        } else {
            info!("No config.json found — first run, watch_dirs empty, is_onboarded=false");
            PersistedConfig { watch_dirs: vec![], is_onboarded: false }
        };

        // Task 3: Enforce configuration isolation. Read explicitly from the saved config.json
        // and do NOT merge or fall back onto standard system roots if the array contains at least one item.
        let watch_dirs = if !persisted.watch_dirs.is_empty() {
            persisted.watch_dirs
        } else {
            vec![]
        };

        info!("Watch dirs: {:?}, onboarded: {}", watch_dirs, persisted.is_onboarded);

        Ok(Config {
            data_dir,
            max_file_bytes: MAX_FILE_BYTES,
            // 128 KB covers the full context window of BGE-small-en-v1.5 (512 tokens ≈
            // 2000 words ≈ ~12 KB, so 128 KB is generous without being wasteful).
            text_read_limit_bytes: 128 * 1024,
            // 2 intra-op threads: uses both physical cores of the i7-6500U
            // while leaving the HT siblings for IPC / OS tasks.
            onnx_intra_threads: 2,
            // 32 docs/batch: ~few MB peak RAM per batch, fast enough
            // amortisation of ONNX session startup (~200 ms).
            embed_batch_size: 5,
            watch_dirs,
            index_parallelism: 1,
            is_onboarded: persisted.is_onboarded,
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

    /// Path to the persisted config JSON file.
    pub fn config_file_path(&self) -> PathBuf {
        self.data_dir.join("config.json")
    }

    /// Persist the current watch_dirs and is_onboarded to disk.
    pub fn save_persisted(&self) -> Result<(), ConfigError> {
        let persisted = PersistedConfig {
            watch_dirs: self.watch_dirs.clone(),
            is_onboarded: self.is_onboarded,
        };
        let json = serde_json::to_string_pretty(&persisted)
            .map_err(|e| ConfigError::IoError {
                path: self.config_file_path(),
                source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
            })?;
        std::fs::write(self.config_file_path(), json).map_err(|e| {
            ConfigError::IoError {
                path: self.config_file_path(),
                source: e,
            }
        })?;
        info!("Persisted config saved to {:?}", self.config_file_path());
        Ok(())
    }

    /// Returns true if the daemon should perform automatic crawling.
    /// Crawling is skipped when not onboarded or when watch_dirs is empty.
    pub fn should_crawl(&self) -> bool {
        self.is_onboarded && !self.watch_dirs.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn resolve_data_dir() -> Result<PathBuf, ConfigError> {
    if let Ok(dir_str) = std::env::var("CRUMBS_DATA_DIR") {
        return Ok(PathBuf::from(dir_str));
    }
    if let Some(base) = dirs::data_local_dir() {
        Ok(base.join("com.crumbs.app"))
    } else {
        Ok(PathBuf::from("/var/lib/com.crumbs.app"))
    }
}

fn default_watch_dirs() -> Vec<PathBuf> {
    // No automatic home directory indexing — user must explicitly select
    // folders during the onboarding flow.
    vec![]
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
