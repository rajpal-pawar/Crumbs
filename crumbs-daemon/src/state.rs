use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use ort::session::Session;
use tokenizers::Tokenizer;
use crate::config::Config;
use crate::embed::EmbedError;

// ---------------------------------------------------------------------------
// RAII guard — ensures `indexer_count` is always decremented, even on panic.
// ---------------------------------------------------------------------------

/// Increment the indexer counter on creation; decrement (and maybe cleanup)
/// when the guard is dropped — including unwinding from a panic.
pub struct IndexerGuard;

impl IndexerGuard {
    /// Create a new guard, incrementing the active-indexer counter.
    pub fn new() -> Self {
        get_model_manager().increment_indexer();
        Self
    }
}

impl Drop for IndexerGuard {
    fn drop(&mut self) {
        get_model_manager().decrement_indexer();
    }
}

pub struct BgeModel {
    pub session: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

pub struct CLIPTextModel {
    pub session: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

pub struct ModelManager {
    bge: RwLock<Option<Arc<BgeModel>>>,
    clip_vision: RwLock<Option<Arc<Mutex<Session>>>>,
    clip_text: RwLock<Option<Arc<CLIPTextModel>>>,
    
    active_search_count: AtomicUsize,
    /// Number of concurrently active indexer operations (reindex pipeline,
    /// single-file index from the background watcher, etc.).
    indexer_count: AtomicUsize,
    is_paused: AtomicBool,
    
    last_failed_files: RwLock<Vec<(String, String)>>,
    last_skipped_files: RwLock<Vec<(String, String)>>,
}

pub static MODEL_MANAGER: std::sync::OnceLock<ModelManager> = std::sync::OnceLock::new();

pub fn get_model_manager() -> &'static ModelManager {
    MODEL_MANAGER.get_or_init(ModelManager::new)
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            bge: RwLock::new(None),
            clip_vision: RwLock::new(None),
            clip_text: RwLock::new(None),
            active_search_count: AtomicUsize::new(0),
            indexer_count: AtomicUsize::new(0),
            is_paused: AtomicBool::new(false),
            last_failed_files: RwLock::new(Vec::new()),
            last_skipped_files: RwLock::new(Vec::new()),
        }
    }

    /// Increment the indexer counter.  Prefer using [`IndexerGuard`] instead
    /// of calling this directly.
    pub fn increment_indexer(&self) {
        self.indexer_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement the indexer counter and trigger cleanup if everything is idle.
    /// Prefer using [`IndexerGuard`] instead of calling this directly.
    pub fn decrement_indexer(&self) {
        let prev = self.indexer_count.fetch_sub(1, Ordering::SeqCst);
        // Guard against underflow (shouldn't happen, but be safe).
        if prev == 0 {
            tracing::warn!("decrement_indexer called when count was already 0");
            self.indexer_count.store(0, Ordering::SeqCst);
        }
        self.maybe_cleanup();
    }

    pub fn is_engine_paused(&self) -> bool {
        self.is_paused.load(Ordering::SeqCst)
    }

    pub fn set_engine_paused(&self, paused: bool) {
        self.is_paused.store(paused, Ordering::SeqCst);
    }

    pub fn is_indexer_active(&self) -> bool {
        self.indexer_count.load(Ordering::SeqCst) > 0
    }

    pub fn set_indexing_issues(&self, failed: Vec<(String, String)>, skipped: Vec<(String, String)>) {
        if let Ok(mut lock) = self.last_failed_files.write() {
            *lock = failed;
        }
        if let Ok(mut lock) = self.last_skipped_files.write() {
            *lock = skipped;
        }
    }

    pub fn get_indexing_issues(&self) -> (Vec<(String, String)>, Vec<(String, String)>) {
        let failed = self.last_failed_files.read().map(|l| l.clone()).unwrap_or_default();
        let skipped = self.last_skipped_files.read().map(|l| l.clone()).unwrap_or_default();
        (failed, skipped)
    }

    pub fn get_onnx_memory_footprint(&self) -> u64 {
        let mut onnx_memory = 0;
        if let Ok(lock) = self.bge.read() {
            if lock.is_some() {
                onnx_memory += 90 * 1024 * 1024;
            }
        }
        if let Ok(lock) = self.clip_vision.read() {
            if lock.is_some() {
                onnx_memory += 350 * 1024 * 1024;
            }
        }
        if let Ok(lock) = self.clip_text.read() {
            if lock.is_some() {
                onnx_memory += 350 * 1024 * 1024;
            }
        }
        onnx_memory
    }

    pub fn increment_search(&self) {
        self.active_search_count.fetch_add(1, Ordering::SeqCst);
    }

    pub fn decrement_search(&self) {
        self.active_search_count.fetch_sub(1, Ordering::SeqCst);
        self.maybe_cleanup();
    }

    pub fn has_pending_searches(&self) -> bool {
        self.active_search_count.load(Ordering::SeqCst) > 0
    }

    pub fn is_bge_ready(&self) -> bool {
        if let Ok(lock) = self.bge.read() {
            lock.is_some()
        } else {
            false
        }
    }

    fn maybe_cleanup(&self) {
        let search_active = self.active_search_count.load(Ordering::SeqCst) > 0;
        let indexer_active = self.indexer_count.load(Ordering::SeqCst) > 0;

        
        if !search_active && !indexer_active {
            tracing::info!("Both search and indexer are idle. Freeing ONNX model weights from RAM.");
            if let Ok(mut lock) = self.bge.write() {
                *lock = None;
            }
            if let Ok(mut lock) = self.clip_vision.write() {
                *lock = None;
            }
            if let Ok(mut lock) = self.clip_text.write() {
                *lock = None;
            }
        }
    }

    pub fn get_bge(&self, config: &Config) -> Result<Arc<BgeModel>, EmbedError> {
        {
            let lock = self.bge.read().map_err(|e| EmbedError::Ort(format!("RwLock read error: {}", e)))?;
            if let Some(ref model) = *lock {
                return Ok(Arc::clone(model));
            }
        }
        let mut lock = self.bge.write().map_err(|e| EmbedError::Ort(format!("RwLock write error: {}", e)))?;
        if let Some(ref model) = *lock {
            return Ok(Arc::clone(model));
        }

        tracing::info!("Loading BGE-small-en-v1.5 session and tokenizer...");
        let models_dir = config.model_cache_dir();
        let model_path = models_dir.join(crate::embed::BGE_FILENAME);
        let tokenizer_path = models_dir.join(crate::embed::TOKENIZER_FILENAME);

        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound(model_path));
        }
        if !tokenizer_path.exists() {
            return Err(EmbedError::ModelNotFound(tokenizer_path));
        }

        let session = Session::builder()
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbedError::Ort(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let model = Arc::new(BgeModel { session: Mutex::new(session), tokenizer });
        *lock = Some(Arc::clone(&model));
        Ok(model)
    }

    pub fn get_clip_vision(&self, config: &Config) -> Result<Arc<Mutex<Session>>, EmbedError> {
        {
            let lock = self.clip_vision.read().map_err(|e| EmbedError::Ort(format!("RwLock read error: {}", e)))?;
            if let Some(ref model) = *lock {
                return Ok(Arc::clone(model));
            }
        }
        let mut lock = self.clip_vision.write().map_err(|e| EmbedError::Ort(format!("RwLock write error: {}", e)))?;
        if let Some(ref model) = *lock {
            return Ok(Arc::clone(model));
        }

        tracing::info!("Loading CLIP vision session...");
        let model_path = config.model_cache_dir().join(crate::embed::CLIP_FILENAME);
        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound(model_path));
        }

        let session = Session::builder()
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .with_intra_threads(config.onnx_intra_threads as usize)
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbedError::Ort(e.to_string()))?;

        let model = Arc::new(Mutex::new(session));
        *lock = Some(Arc::clone(&model));
        Ok(model)
    }

    pub fn get_clip_text(&self, config: &Config) -> Result<Arc<CLIPTextModel>, EmbedError> {
        {
            let lock = self.clip_text.read().map_err(|e| EmbedError::Ort(format!("RwLock read error: {}", e)))?;
            if let Some(ref model) = *lock {
                return Ok(Arc::clone(model));
            }
        }
        let mut lock = self.clip_text.write().map_err(|e| EmbedError::Ort(format!("RwLock write error: {}", e)))?;
        if let Some(ref model) = *lock {
            return Ok(Arc::clone(model));
        }

        tracing::info!("Loading CLIP text session and tokenizer...");
        let models_dir = config.model_cache_dir();
        let model_path = models_dir.join(crate::embed::CLIP_TEXT_FILENAME);
        let tokenizer_path = models_dir.join(crate::embed::CLIP_TOKENIZER_FILENAME);

        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound(model_path));
        }
        if !tokenizer_path.exists() {
            return Err(EmbedError::ModelNotFound(tokenizer_path));
        }

        let session = Session::builder()
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .with_intra_threads(config.onnx_intra_threads as usize)
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| EmbedError::Ort(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbedError::Ort(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let model = Arc::new(CLIPTextModel { session: Mutex::new(session), tokenizer });
        *lock = Some(Arc::clone(&model));
        Ok(model)
    }
}

