//! `index/mod.rs` — Public façade for the Crumbs index layer.
//!
//! # Overview
//!
//! This module exposes a single [`Database`] struct that wraps a
//! `Mutex<rusqlite::Connection>`.  The mutex is required because
//! `rusqlite::Connection` is `!Send + !Sync` (it holds a raw `*mut sqlite3`
//! pointer).  Wrapping it in a `Mutex` makes the struct `Send + Sync` so it
//! can be stored in `Arc<Database>` and shared across async tasks.
//!
//! All heavy SQLite work is expected to be dispatched via
//! `tokio::task::spawn_blocking` in the callers (`handlers.rs`).  The methods
//! on `Database` are intentionally synchronous — they must NOT be called
//! directly from an async context without `spawn_blocking`.
//!
//! # Extension loading
//!
//! `sqlite-vec` is loaded globally via `sqlite3_auto_extension` **before**
//! the first connection is opened.  Call [`register_vec_extension`] exactly
//! once at process startup (in `main.rs`).

pub mod schema;
pub mod search;
pub mod writer;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use rusqlite::Connection;
use tracing::info;

use crate::index::schema::apply_schema;

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

/// Register the `sqlite-vec` extension globally so every subsequent
/// `Connection::open*` call automatically includes it.
///
/// # Safety
/// This function uses FFI (`sqlite3_auto_extension`) and must be called
/// **once** before any SQLite connection is opened.  Calling it multiple
/// times is harmless but redundant.
///
/// # Panics
/// Does not panic — `sqlite3_auto_extension` always succeeds for a valid
/// entry point.
pub fn register_vec_extension() {
    // SAFETY: `sqlite3_vec_init` is a valid C-compatible entry point exported
    // by the `sqlite-vec` static library.  We transmute it to the opaque
    // function-pointer type that `sqlite3_auto_extension` expects.
    unsafe {
        // SAFETY: `sqlite3_vec_init` has the exact signature that
        // `sqlite3_auto_extension` expects.  The explicit transmute
        // from/to types documents the ABI contract.
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *const ::std::os::raw::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> ::std::os::raw::c_int,
        >(sqlite_vec::sqlite3_vec_init as *const ())));
    }
    info!("sqlite-vec extension registered via sqlite3_auto_extension");
}

// ---------------------------------------------------------------------------
// Database wrapper
// ---------------------------------------------------------------------------

/// Thread-safe wrapper around a connection pool of `rusqlite::Connection`s.
///
/// WAL mode allows one writer and multiple concurrent readers.
/// We hold one write connection and a pool of read connections.
pub struct Database {
    write_conn: Mutex<Connection>,
    read_pool: Vec<Arc<Mutex<Connection>>>,
    read_idx: AtomicUsize,
}

impl Database {
    /// Open (or create) the Crumbs database at `path`, apply the schema, and
    /// enable WAL journal mode for better concurrent read performance.
    ///
    /// # Errors
    /// Returns [`DbError`] if the file cannot be opened or the schema
    /// migration fails.
    pub fn open(path: &Path) -> Result<Self, DbError> {
        // Create parent directory if it doesn't exist.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(DbError::Io)?;
        }

        // Open write connection
        let conn = Connection::open(path).map_err(DbError::Rusqlite)?;

        // WAL mode — allows reads to proceed concurrently with a single
        // writer, which matters when the indexer is writing while the UI
        // issues a search query.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;"
        ).map_err(DbError::Rusqlite)?;

        // Apply the schema (idempotent — uses CREATE IF NOT EXISTS).
        apply_schema(&conn)?;

        // Open a pool of read connections (e.g. 4)
        let mut read_pool = Vec::new();
        for _ in 0..4 {
            let r_conn = Connection::open(path).map_err(DbError::Rusqlite)?;
            r_conn.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=NORMAL;
                 PRAGMA foreign_keys=ON;"
            ).map_err(DbError::Rusqlite)?;
            read_pool.push(Arc::new(Mutex::new(r_conn)));
        }

        info!(path = %path.display(), "database opened (with read connection pool) and schema applied");

        Ok(Database {
            write_conn: Mutex::new(conn),
            read_pool,
            read_idx: AtomicUsize::new(0),
        })
    }

    /// Acquire a lock on the dedicated write connection and call `f` with a reference.
    pub fn with_conn<F, T>(&self, f: F) -> Result<T, DbError>
    where
        F: FnOnce(&Connection) -> Result<T, DbError>,
    {
        let conn = self.write_conn.lock().map_err(|_| DbError::PoisonedLock)?;
        f(&conn)
    }

    /// Acquire a lock on a separate read connection from the pool and call `f` with a reference.
    /// This allows concurrent read-only queries (like searches) without blocking the indexer writer.
    pub fn with_read_conn<F, T>(&self, f: F) -> Result<T, DbError>
    where
        F: FnOnce(&Connection) -> Result<T, DbError>,
    {
        let idx = self.read_idx.fetch_add(1, Ordering::Relaxed) % self.read_pool.len();
        let conn_arc = &self.read_pool[idx];
        let conn = conn_arc.lock().map_err(|_| DbError::PoisonedLock)?;
        f(&conn)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("SQLite error: {0}")]
    Rusqlite(#[from] rusqlite::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("database mutex was poisoned")]
    PoisonedLock,

    #[error("schema error: {0}")]
    Schema(String),
}
