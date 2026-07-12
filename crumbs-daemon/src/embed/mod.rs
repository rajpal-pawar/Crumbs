//! `embed/mod.rs` — Lazy ONNX session execution for text and image embeddings.
//!
//! # RAM budget strategy (i7-6500U, 8 GB)
//!
//! Loading both MiniLM (≈ 90 MB) and CLIP (≈ 350 MB) simultaneously would
//! consume ~440 MB just for model weights.  We stay under 1 GB by strictly
//! sequential lazy loading:
//!
//! ```text
//!  embed_text_batch(texts, config)   ← load tokenizer → open MiniLM → infer → drop
//!  embed_image_batch(images, config) ← open CLIP → infer → drop
//! ```
//!
//! Each function creates and immediately drops its session after returning.
//!
//! # ONNX thread throttling
//!
//! Both sessions enforce:
//! - `with_intra_threads(config.onnx_intra_threads)` — 2 on i7-6500U
//! - `with_inter_threads(1)` — prevents graph-level parallelism
//!
//! # Model + tokenizer paths
//!
//! All files live in `<data_dir>/models/`:
//! | File                      | Purpose                        |
//! |---------------------------|--------------------------------|
//! | `minilm-l6-int8.onnx`     | MiniLM sentence encoder        |
//! | `tokenizer.json`          | HuggingFace WordPiece tokenizer|
//! | `clip-vit-b32-int8.onnx`  | CLIP visual encoder            |
//!
//! If any required file is absent the function returns
//! `Err(EmbedError::ModelNotFound)` so the caller can degrade gracefully.
//!
//! # Input / output shapes
//!
//! | Model   | Input                              | Output shape   |
//! |---------|------------------------------------|----------------|
//! | MiniLM  | `input_ids [1, ≤512]` i64          | `[1, seq, 384]`|
//! | CLIP    | `pixel_values [1, 3, 224, 224]` f32| `[1, 512]` f32 |

use std::path::PathBuf;

use std::sync::Arc;

use image::{DynamicImage, GenericImageView};
use ndarray::{Array, Array2, Array3, Array4, ArrayViewD, Axis};
use ort::session::Session;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::config::Config;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
fn l2_normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

/// INT8-quantized MiniLM from Xenova/all-MiniLM-L6-v2.
/// Reduces model RAM from ~90 MB → ~23 MB.
const MINILM_FILENAME:    &str = "minilm-l6-int8.onnx";
const TOKENIZER_FILENAME: &str = "tokenizer.json";
/// INT8-quantized CLIP visual encoder from Xenova/clip-vit-base-patch32.
/// Reduces model RAM from ~350 MB → ~87 MB.
const CLIP_FILENAME: &str = "clip-vision-int8.onnx";
const CLIP_TEXT_FILENAME: &str = "clip-text-int8.onnx";
const CLIP_TOKENIZER_FILENAME: &str = "clip-tokenizer.json";

/// Maximum token sequence length fed to MiniLM.
/// MiniLM-L6-v2 supports up to 512, but 128 covers most short documents
/// and halves the tensor allocation.
const MINILM_SEQ_LEN: usize = 512;

/// CLIP ViT-B/32 expected spatial resolution.
const CLIP_IMAGE_SIZE: u32 = 224;

// ---------------------------------------------------------------------------
// Public API — Text
// ---------------------------------------------------------------------------

pub fn eagerly_init_minilm(config: &Config) {
    let _ = crate::state::get_model_manager().get_minilm(config);
}

pub fn is_minilm_ready() -> bool {
    crate::state::get_model_manager().is_minilm_ready()
}

pub fn embed_text_batch(
    texts: &[String],
    config: &Config,
) -> Result<Vec<Vec<Vec<f32>>>, EmbedError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let model = crate::state::get_model_manager().get_minilm(config)?;
    let mut results = Vec::with_capacity(texts.len());

    for (doc_idx, text) in texts.iter().enumerate() {
        let chunks = chunk_text(text, 300, 50);
        let mut doc_embeddings = Vec::with_capacity(chunks.len());
        
        for chunk in chunks {
            let encoding = model.tokenizer
                .encode(chunk.as_str(), true)
                .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

            let (ids, mask, type_ids) = encode_to_tensors(&encoding, MINILM_SEQ_LEN)?;

            let inputs = ort::inputs![
                "input_ids"      => ort::value::Tensor::from_array(ids).map_err(|e| EmbedError::Ort(e.to_string()))?,
                "attention_mask" => ort::value::Tensor::from_array(mask.clone()).map_err(|e| EmbedError::Ort(e.to_string()))?,
                "token_type_ids" => ort::value::Tensor::from_array(type_ids).map_err(|e| EmbedError::Ort(e.to_string()))?,
            ];

            let mut session_guard = model.session.lock().map_err(|e| EmbedError::Ort(format!("Mutex lock error: {}", e)))?;
            let outputs = session_guard.run(inputs).map_err(|e| EmbedError::Ort(e.to_string()))?;

            let hidden = outputs["last_hidden_state"]
                .try_extract_array::<f32>()
                .map_err(|e| EmbedError::Ort(e.to_string()))?;

            let real_mask: Vec<bool> = mask
                .iter()
                .map(|&m| m == 1i64)
                .collect();

            let embedding = mean_pool_with_mask(hidden.view(), &real_mask);
            doc_embeddings.push(embedding);
        }

        results.push(doc_embeddings);
        debug!(doc_idx, "text embeddings generated for all chunks");
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Public API — Images
// ---------------------------------------------------------------------------

/// Generate 512-dim image embeddings using CLIP ViT-B/32 visual encoder.
///
/// **RAM lifecycle:** CLIP session opened, inference run, session dropped.
///
/// # Errors
/// - [`EmbedError::ModelNotFound`] if the `.onnx` file is absent.
/// - [`EmbedError::Ort`] for ONNX runtime errors.
pub fn embed_image_batch(
    images: &[DynamicImage],
    config: &Config,
) -> Result<Vec<Vec<f32>>, EmbedError> {
    if images.is_empty() {
        return Ok(Vec::new());
    }

    let session = crate::state::get_model_manager().get_clip_vision(config)?;
    let mut results = Vec::with_capacity(images.len());

    for (doc_idx, img) in images.iter().enumerate() {
        let resized = img.resize_exact(
            CLIP_IMAGE_SIZE,
            CLIP_IMAGE_SIZE,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();
        let pixel_values = image_to_clip_tensor(rgb)?;

        let inputs = ort::inputs![
            "pixel_values" => ort::value::Tensor::from_array(pixel_values).map_err(|e| EmbedError::Ort(e.to_string()))?,
        ];

        let mut session_guard = session.lock().map_err(|e| EmbedError::Ort(format!("Mutex lock error: {}", e)))?;
        let outputs = session_guard.run(inputs).map_err(|e| EmbedError::Ort(e.to_string()))?;

        let embeds = outputs["image_embeds"]
            .try_extract_array::<f32>()
            .map_err(|e| EmbedError::Ort(e.to_string()))?;

        let mut vec: Vec<f32> = embeds.iter().copied().collect();
        if vec.len() != 512 {
            return Err(EmbedError::Shape(format!(
                "expected CLIP output dim 512, got {}",
                vec.len()
            )));
        }

        l2_normalize(&mut vec);

        results.push(vec);
        debug!(doc_idx, "image embedding generated");
    }

    Ok(results)
}

/// Generate 512-dim text embeddings using CLIP text encoder for querying images.
pub fn embed_clip_text(query: &str, config: &Config) -> Result<Vec<f32>, EmbedError> {
    let model = crate::state::get_model_manager().get_clip_text(config)?;

    let encoding = model.tokenizer
        .encode(query, true)
        .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

    let (ids, _mask, _type_ids) = encode_to_tensors(&encoding, 77)?;

    let inputs = ort::inputs![
        "input_ids" => ort::value::Tensor::from_array(ids).map_err(|e| EmbedError::Ort(e.to_string()))?,
    ];

    let mut session_guard = model.session.lock().map_err(|e| EmbedError::Ort(format!("Mutex lock error: {}", e)))?;
    let outputs = session_guard.run(inputs).map_err(|e| EmbedError::Ort(e.to_string()))?;

    let embeds = outputs["text_embeds"]
        .try_extract_array::<f32>()
        .map_err(|e| EmbedError::Ort(e.to_string()))?;

    let mut vec: Vec<f32> = embeds.iter().copied().collect();
    if vec.len() != 512 {
        return Err(EmbedError::Shape(format!(
            "expected CLIP text output dim 512, got {}",
            vec.len()
        )));
    }

    l2_normalize(&mut vec);

    Ok(vec)
}

// ---------------------------------------------------------------------------
// Tokenizer helpers
// ---------------------------------------------------------------------------

/// Convert a HuggingFace [`tokenizers::Encoding`] into three
/// `[1, seq_len]` i64 ndarray tensors: `(input_ids, attention_mask, token_type_ids)`.
///
/// Truncates to `seq_len` tokens and pads shorter sequences with zeros.
/// The attention_mask is `1` for real tokens and `0` for padding.
fn encode_to_tensors(
    encoding: &tokenizers::Encoding,
    seq_len: usize,
) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>), EmbedError> {
    let ids_raw   = encoding.get_ids();
    let mask_raw  = encoding.get_attention_mask();
    let types_raw = encoding.get_type_ids();

    // Pre-allocate padded buffers.
    let mut ids   = vec![0i64; seq_len];
    let mut mask  = vec![0i64; seq_len];
    let mut types = vec![0i64; seq_len];

    let n = ids_raw.len().min(seq_len);
    for i in 0..n {
        ids[i]   = ids_raw[i]   as i64;
        mask[i]  = mask_raw[i]  as i64;
        types[i] = types_raw[i] as i64;
    }

    let make = |v: Vec<i64>| {
        Array2::from_shape_vec((1, seq_len), v)
            .map_err(|e| EmbedError::Shape(e.to_string()))
    };

    Ok((make(ids)?, make(mask)?, make(types)?))
}

// ---------------------------------------------------------------------------
// Pooling + normalisation
// ---------------------------------------------------------------------------

/// Attention-mask–aware mean pooling of `[1, seq_len, hidden_dim]` → `[hidden_dim]`.
///
/// Only positions where `mask[i] == true` are included in the average.
/// The result is L2-normalised so cosine similarity equals dot product.
fn mean_pool_with_mask(
    hidden: ndarray::ArrayViewD<f32>,
    mask: &[bool],
) -> Vec<f32> {
    let shape      = hidden.shape();
    let seq_len    = shape[1];
    let hidden_dim = shape[2];

    let mut pool  = vec![0.0f32; hidden_dim];
    let mut count = 0usize;

    for s in 0..seq_len {
        // Skip padding positions.
        if s >= mask.len() || !mask[s] {
            continue;
        }
        for h in 0..hidden_dim {
            pool[h] += hidden[[0, s, h]];
        }
        count += 1;
    }

    if count > 0 {
        let inv = 1.0 / count as f32;
        for v in &mut pool { *v *= inv; }
    }

    // L2-normalise.
    let norm: f32 = pool.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for v in &mut pool { *v /= norm; }
    }

    pool
}

// ---------------------------------------------------------------------------
// Text chunking
// ---------------------------------------------------------------------------

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![String::new()]; // At least one empty chunk to avoid dropping empty documents
    }
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_size).min(words.len());
        chunks.push(words[i..end].join(" "));
        if end == words.len() {
            break;
        }
        i += chunk_size - overlap;
    }
    chunks
}

// ---------------------------------------------------------------------------
// CLIP image tensor
// ---------------------------------------------------------------------------

/// Convert an RGB8 224×224 image to a `[1, 3, 224, 224]` f32 tensor
/// normalised with ImageNet statistics (as expected by CLIP ViT-B/32).
fn image_to_clip_tensor(rgb: image::RgbImage) -> Result<Array4<f32>, EmbedError> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD:  [f32; 3] = [0.229, 0.224, 0.225];

    let (w, h) = rgb.dimensions();
    debug_assert_eq!(w, CLIP_IMAGE_SIZE);
    debug_assert_eq!(h, CLIP_IMAGE_SIZE);

    let mut tensor = Array4::<f32>::zeros([1, 3, h as usize, w as usize]);

    for y in 0..h {
        for x in 0..w {
            let px = rgb.get_pixel(x, y);
            for c in 0..3usize {
                tensor[[0, c, y as usize, x as usize]] =
                    (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }

    Ok(tensor)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn check_file_exists(path: &PathBuf) -> Result<(), EmbedError> {
    if !path.exists() {
        warn!(path = %path.display(), "required model file not found");
        return Err(EmbedError::ModelNotFound(path.clone()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("model file not found: {0}")]
    ModelNotFound(PathBuf),

    #[error("ORT error: {0}")]
    Ort(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("tensor shape error: {0}")]
    Shape(String),
}
