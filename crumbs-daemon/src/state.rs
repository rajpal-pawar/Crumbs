use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use ort::session::Session;
use tokenizers::Tokenizer;
use crate::config::Config;
use crate::embed::EmbedError;

pub struct MiniLMModel {
    pub session: Session,
    pub tokenizer: Tokenizer,
}

pub struct CLIPTextModel {
    pub session: Session,
    pub tokenizer: Tokenizer,
}

pub struct ModelManager {
    minilm: RwLock<Option<Arc<MiniLMModel>>>,
    clip_vision: RwLock<Option<Arc<Session>>>,
    clip_text: RwLock<Option<Arc<CLIPTextModel>>>,
    
    active_search_count: AtomicUsize,
    is_indexer_active: AtomicBool,
}

pub static MODEL_MANAGER: std::sync::OnceLock<ModelManager> = std::sync::OnceLock::new();

pub fn get_model_manager() -> &'static ModelManager {
    MODEL_MANAGER.get_or_init(ModelManager::new)
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            minilm: RwLock::new(None),
            clip_vision: RwLock::new(None),
            clip_text: RwLock::new(None),
            active_search_count: AtomicUsize::new(0),
            is_indexer_active: AtomicBool::new(false),
        }
    }

    pub fn set_indexer_active(&self, active: bool, config: &Config) {
        self.is_indexer_active.store(active, Ordering::SeqCst);
        if !active {
            self.maybe_cleanup();
        }
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

    pub fn is_minilm_ready(&self) -> bool {
        if let Ok(lock) = self.minilm.read() {
            lock.is_some()
        } else {
            false
        }
    }

    fn maybe_cleanup(&self) {
        let search_active = self.active_search_count.load(Ordering::SeqCst) > 0;
        let indexer_active = self.is_indexer_active.load(Ordering::SeqCst);
        
        if !search_active && !indexer_active {
            tracing::info!("Both search and indexer are idle. Freeing ONNX model weights from RAM.");
            if let Ok(mut lock) = self.minilm.write() {
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

    pub fn get_minilm(&self, config: &Config) -> Result<Arc<MiniLMModel>, EmbedError> {
        {
            let lock = self.minilm.read().map_err(|e| EmbedError::Ort(format!("RwLock read error: {}", e)))?;
            if let Some(ref model) = *lock {
                return Ok(Arc::clone(model));
            }
        }
        let mut lock = self.minilm.write().map_err(|e| EmbedError::Ort(format!("RwLock write error: {}", e)))?;
        if let Some(ref model) = *lock {
            return Ok(Arc::clone(model));
        }

        tracing::info!("Loading MiniLM session and tokenizer...");
        let models_dir = config.model_cache_dir();
        let model_path = models_dir.join("minilm-l6-int8.onnx");
        let tokenizer_path = models_dir.join("tokenizer.json");

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

        let model = Arc::new(MiniLMModel { session, tokenizer });
        *lock = Some(Arc::clone(&model));
        Ok(model)
    }

    pub fn get_clip_vision(&self, config: &Config) -> Result<Arc<Session>, EmbedError> {
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
        let model_path = config.model_cache_dir().join("clip-vision-int8.onnx");
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

        let model = Arc::new(session);
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
        let model_path = models_dir.join("clip-text-int8.onnx");
        let tokenizer_path = models_dir.join("clip-tokenizer.json");

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

        let model = Arc::new(CLIPTextModel { session, tokenizer });
        *lock = Some(Arc::clone(&model));
        Ok(model)
    }
}
