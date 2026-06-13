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
mod throttle;

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
    // 4b. MiniLM ONNX session initialization (BLOCKING)
    //     We AWAIT this before starting the crawl so that the initial index
    //     run actually generates embeddings.  The IPC loop is spawned first
    //     (below) so the UI is responsive while the model loads.
    // -----------------------------------------------------------------------
    {
        let init_config = config.clone();
        info!("Loading MiniLM ONNX model (this may take a few seconds)...");
        let _ = tokio::task::spawn_blocking(move || {
            embed::eagerly_init_minilm(&init_config);
        }).await;
        info!("MiniLM initialization complete (model ready = {})", embed::is_minilm_ready());
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
    let crawl_config = config.clone();
    let crawl_db = Arc::clone(&db);

    let ipc_handle = tokio::spawn(async move {
        if let Err(e) = ipc::run_loop(config, db).await {
            error!(error = %e, "IPC run-loop terminated with error");
            std::process::exit(1);
        }
    });

    // Spawn the directory crawler *after* the IPC loop is already running.
    tokio::spawn(async move {
        info!("Performing initial scan of watch_dirs in background...");
        let _ = tokio::task::spawn_blocking({
            let crawl_config = crawl_config.clone();
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
        use notify::{Watcher, RecursiveMode, EventKind};
        use std::sync::mpsc::channel;
        use std::time::Duration;

        let (tx, rx) = channel();
        let mut watcher = match notify::RecommendedWatcher::new(
            move |res| { let _ = tx.send(res); },
            notify::Config::default()
        ) {
            Ok(w) => w,
            Err(e) => {
                error!(error = %e, "failed to initialize file watcher");
                return;
            }
        };

        let watch_dir = dirs::home_dir()
            .map(|mut p| { p.push("Crumbs"); p.push("data"); p })
            .unwrap_or_else(|| std::path::PathBuf::from("/var/lib/crumbs/data"));
            
        std::fs::create_dir_all(&watch_dir).ok();

        if watch_dir.exists() {
            if let Err(e) = watcher.watch(&watch_dir, RecursiveMode::Recursive) {
                warn!(path = %watch_dir.display(), error = %e, "failed to watch directory");
            } else {
                info!(path = %watch_dir.display(), "watching directory for changes");
            }
        }

        // Loop to listen for file changes
        tokio::task::spawn_blocking(move || {
            // Keep the watcher alive by moving it into this closure
            let _watcher = watcher;
            
            loop {
                match rx.recv() {
                    Ok(Ok(event)) => {
                        // Only trigger reindex for actual content changes
                        match event.kind {
                            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) => {
                                // Wait a bit to let file writes finish and debounce multiple events
                                std::thread::sleep(Duration::from_millis(500));
                                
                                // Drain any other events that fired in the meantime
                                while let Ok(_) = rx.try_recv() {}

                                info!("detected file changes, triggering reindex...");
                                if let Err(e) = handlers::run_reindex_pipeline(&crawl_config, &crawl_db) {
                                    error!(error = %e, "file watcher reindex failed");
                                } else {
                                    info!("file watcher reindex completed");
                                }
                            }
                            _ => {}
                        }
                    }
                    Ok(Err(e)) => warn!("watch error: {}", e),
                    Err(_) => break, // Channel closed
                }
            }
        });
    });

    let _ = ipc_handle.await;
    info!("crumbs-daemon shutting down gracefully");
}
