//! crumbs-daemon — Entry point.
//!
//! # STDOUT IS SACRED
//! Nothing in this binary may write to `stdout` except the NDJSON IPC layer
//! defined in [`ipc`].  All diagnostic output uses [`tracing`], which is
//! configured here to emit **exclusively to stderr**.
//!
//! # Startup sequence
//! 1. Register the `sqlite-vec` extension globally (must happen before any
//!    `Connection` is opened).
//! 2. Initialise `tracing` → stderr.
//! 3. Load [`Config`] from the platform data directory.
//! 4. Open the [`index::Database`].
//! 5. Apply process throttling via [`throttle::apply`].
//! 6. Enter [`ipc::run_loop`] — this blocks until stdin is closed (i.e. the
//!    Tauri host exits).

mod config;
mod embed;
mod extractor;
mod handlers;
mod index;
mod ipc;
mod model_download;
mod throttle;
mod state;

use std::sync::Arc;

use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() {
    // -----------------------------------------------------------------------
    // 1. Register sqlite-vec BEFORE any Connection is opened.
    //    sqlite3_auto_extension must be called on the main thread before
    //    the Tokio runtime spawns additional threads that might open
    //    connections.
    // -----------------------------------------------------------------------
    index::register_vec_extension();

    // -----------------------------------------------------------------------
    // 2. Tracing → stderr
    //    IMPORTANT: `fmt::init()` defaults to stdout.  We explicitly route to
    //    stderr so that IPC over stdout is never polluted.
    // -----------------------------------------------------------------------
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("crumbs_daemon=info,warn"));

    fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_writer(std::io::stderr) // <-- CRITICAL: stderr, not stdout
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!("crumbs-daemon starting up");

    // -----------------------------------------------------------------------
    // 2a. Initialize ONNX Runtime globally (Synchronous)
    // -----------------------------------------------------------------------
    info!("Initializing ONNX Runtime globally...");
    let _ = ort::init().with_name("crumbs-embed").commit();
    info!("ONNX Runtime initialized.");

    // -----------------------------------------------------------------------
    // 2. Load configuration
    // -----------------------------------------------------------------------
    let config = match config::Config::load() {
        Ok(cfg) => {
            info!(
                data_dir = %cfg.data_dir.display(),
                max_file_bytes = cfg.max_file_bytes,
                "configuration loaded"
            );
            info!("[CRITICAL STARTUP] Resolving watch directory to: {:?}", cfg.watch_dirs);
            cfg
        }
        Err(e) => {
            error!(error = %e, "failed to load configuration — aborting");
            std::process::exit(1);
        }
    };

    // -----------------------------------------------------------------------
    // 4. Open the index database
    // -----------------------------------------------------------------------
    let db_path = config.db_path();
    let db = match tokio::task::spawn_blocking(move || index::Database::open(&db_path)).await.expect("Database open task panicked") {
        Ok(d) => {
            info!(path = %config.db_path().display(), "index database opened");
            Arc::new(d)
        }
        Err(e) => {
            error!(error = %e, "failed to open index database — aborting");
            std::process::exit(1);
        }
    };

    // -----------------------------------------------------------------------
    // 4b. Auto-download models if missing
    //     Downloads models.zip from the GitHub release and extracts it to
    //     the platform model cache directory.  Same zip the Tauri app uses.
    // -----------------------------------------------------------------------
    {
        let models_dir = config.model_cache_dir();
        info!("Checking for models in {:?}…", models_dir);
        if let Err(e) = model_download::ensure_models(&models_dir).await {
            warn!("Model auto-download failed: {} — search will degrade to BM25-only", e);
        }
    }

    // -----------------------------------------------------------------------
    // 4c. BGE-small ONNX session initialization (BLOCKING)
    //     We AWAIT this before starting the crawl so that the initial index
    //     run actually generates embeddings.  The IPC loop is spawned first
    //     (below) so the UI is responsive while the model loads.
    // -----------------------------------------------------------------------
    {
        let init_config = config.clone();
        info!("Loading BGE-small ONNX model (this may take a few seconds)...");
        let _ = tokio::task::spawn_blocking(move || {
            let _ = state::get_model_manager().get_minilm(&init_config);
        }).await;
        info!("BGE-small initialization complete (model ready = {})", embed::is_minilm_ready());
    }

    // -----------------------------------------------------------------------
    // 5. Throttle process priority
    // -----------------------------------------------------------------------
    if let Err(e) = throttle::apply() {
        // Non-fatal: warn and continue.  A failure here means we run at normal
        // priority rather than background, which is not ideal on the i7-6500U
        // but does not prevent correct operation.
        tracing::warn!(error = %e, "process throttling failed (non-fatal)");
    } else {
        info!("process throttling applied successfully");
    }

    // -----------------------------------------------------------------------
    // 6. IPC run-loop and Crawler
    // -----------------------------------------------------------------------
    info!("entering IPC run-loop");
    let crawl_config = Arc::new(config.clone());
    let crawl_db = Arc::clone(&db);

    let ipc_handle = tokio::spawn(async move {
        if let Err(e) = ipc::run_loop(config, db).await {
            error!(error = %e, "IPC run-loop terminated with error");
            std::process::exit(1);
        }
    });

    // Spawn the directory crawler *after* the IPC loop is already running,
    // but ONLY if the user has completed onboarding and selected folders.
    tokio::spawn(async move {
        if !crawl_config.should_crawl() {
            info!(
                "Skipping initial crawl: is_onboarded={}, watch_dirs={}",
                crawl_config.is_onboarded,
                crawl_config.watch_dirs.len()
            );
            info!("Daemon is waiting for user to select folders via the onboarding flow.");
            return;
        }

        info!("Performing initial scan of watch_dirs in background...");
        let _ = tokio::task::spawn_blocking({
            let crawl_config = Arc::clone(&crawl_config);
            let crawl_db = Arc::clone(&crawl_db);
            move || {
                if let Err(e) = handlers::run_reindex_pipeline(&crawl_config, &crawl_db) {
                    error!(error = %e, "initial crawl failed");
                } else {
                    info!("initial crawl completed successfully");
                }
            }
        }).await;

        // -------------------------------------------------------------------
        // 7. Background File Watcher
        // -------------------------------------------------------------------
        // Now that the initial crawl is done, watch for real-time file changes.
        use notify::{Watcher, RecursiveMode};
        use std::sync::mpsc::channel;

        let (tx, rx) = channel::<notify::Event>();
        let mut watcher = match notify::RecommendedWatcher::new(
            move |res| {
                if let Ok(event) = res {
                    let _ = tx.send(event);
                }
            },
            notify::Config::default()
        ) {
            Ok(w) => w,
            Err(e) => {
                error!(error = %e, "failed to initialize file watcher");
                return;
            }
        };

        for dir in &crawl_config.watch_dirs {
            if dir.exists() {
                if let Err(e) = watcher.watch(dir, RecursiveMode::NonRecursive) {
                    warn!(path = %dir.display(), error = %e, "failed to watch directory");
                } else {
                    info!(path = %dir.display(), "watching directory for changes (non-recursive)");
                }
            }
        }

        // Spawn dedicated background watcher task
        handlers::start_background_watcher(crawl_config, crawl_db, rx, watcher);
    });

    let _ = ipc_handle.await;
    info!("crumbs-daemon shutting down gracefully");
}
