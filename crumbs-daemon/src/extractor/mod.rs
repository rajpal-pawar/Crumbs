//! `extractor/mod.rs` — File-type router and content extraction.
//!
//! # Responsibilities
//!
//! Given a file path, this module decides:
//! 1. Is this file indexable at all? (size, extension, binary sniff)
//! 2. What *kind* of content does it contain? (text or image)
//! 3. Extract the appropriate payload for the embedding pipeline.
//!
//! # Text extraction
//! For text files the extractor reads at most [`Config::text_read_limit_bytes`]
//! bytes from the *start* of the file.  This cap is deliberate:
//! - MiniLM-L6-v2 has a 512-token context window (~2 KB of dense prose).
//! - Reading 128 KB gives us plenty of context without allocating a 15 MB
//!   `String` buffer that would blow the RAM budget on the i7-6500U.
//!
//! # Image extraction
//! For image files the extractor opens the file and decodes it to an RGB8
//! `DynamicImage`, ready to be resized to 224×224 and converted to a tensor
//! for CLIP.  The full resize/normalise step lives in `embed::image_embed`.
//!
//! # Binary detection
//! We use a two-layer heuristic:
//! 1. Extension allowlist — if the extension is in [`TEXT_EXTENSIONS`] or
//!    [`IMAGE_EXTENSIONS`] we accept immediately.
//! 2. Null-byte sniff — if the first 8 KB contains a null byte and the
//!    extension is unknown, we treat the file as binary and skip it.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use image::DynamicImage;
use sha2::{Digest, Sha256};
use tracing::debug;

use crate::config::Config;

// ---------------------------------------------------------------------------
// Stdout Silencer for rogue println! in dependencies
// ---------------------------------------------------------------------------
struct StdoutSilencer {
    #[cfg(target_os = "linux")]
    saved_stdout: i32,
}

impl StdoutSilencer {
    fn new() -> Self {
        #[cfg(target_os = "linux")]
        unsafe {
            let saved = libc::dup(libc::STDOUT_FILENO);
            libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO);
            Self { saved_stdout: saved }
        }
        #[cfg(not(target_os = "linux"))]
        Self {}
    }
}

impl Drop for StdoutSilencer {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        unsafe {
            libc::dup2(self.saved_stdout, libc::STDOUT_FILENO);
            libc::close(self.saved_stdout);
        }
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The payload extracted from a single file.
#[derive(Debug)]
pub enum Extracted {
    /// A text document ready for BM25 indexing + text embedding.
    Text {
        /// Chunked text content (approx 300 words per chunk with 50 word overlap).
        chunks: Vec<String>,
        /// SHA-256 hex digest of the **full file** (not the truncated body).
        checksum: String,
        /// Detected MIME type string (e.g. `"text/plain"`).
        mime_type: String,
    },
    /// An image ready for CLIP embedding.
    Image {
        /// Decoded image (RGB8).  Will be resized to 224×224 in the embedder.
        image: DynamicImage,
        /// SHA-256 hex digest of the raw file bytes.
        checksum: String,
        /// MIME type string (e.g. `"image/jpeg"`).
        mime_type: String,
    },
}

impl Extracted {
    pub fn checksum(&self) -> &str {
        match self {
            Extracted::Text  { checksum, .. } => checksum,
            Extracted::Image { checksum, .. } => checksum,
        }
    }

    pub fn mime_type(&self) -> &str {
        match self {
            Extracted::Text  { mime_type, .. } => mime_type,
            Extracted::Image { mime_type, .. } => mime_type,
        }
    }
}

// ---------------------------------------------------------------------------
// Extension lists
// ---------------------------------------------------------------------------

/// Extensions we treat as plain text and feed to MiniLM.
pub const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "rst", "adoc",
    "rs",  "py", "js",  "ts",  "jsx", "tsx",
    "go",  "c",  "cpp", "h",   "hpp", "java",
    "cs",  "rb", "php", "sh",  "bat", "ps1",
    "toml", "yaml", "yml", "json", "xml", "html", "htm", "css",
    "csv",  "tsv",  "log", "ini", "cfg", "conf",
    "tex", "bib", "pdf", "bash", "zsh",
];

/// Extensions we treat as images and route to the CLIP stub.
pub const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff", "tif",
];

// Sniff buffer: check the first N bytes for null bytes.
const SNIFF_BYTES: usize = 8 * 1024; // 8 KB

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Attempt to extract indexable content from `path`.
///
/// Returns:
/// - `Ok(Some(Extracted))` — successfully extracted content.
/// - `Ok(None)`            — file is binary / unsupported / should be skipped.
/// - `Err(_)`              — I/O or decode error (caller logs and skips).
///
/// This function is **synchronous** and must be called from within
/// `tokio::task::spawn_blocking`.
pub fn extract(path: &Path, config: &Config) -> Result<Option<Extracted>, ExtractError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if TEXT_EXTENSIONS.contains(&ext.as_str()) {
        extract_text(path, config)
    } else if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        extract_image(path).map(Some)
    } else {
        // Unsupported media/binary file or unknown extension.
        // Skip the AI pipeline entirely, but compute SHA-256 and MIME type.
        let file = File::open(path).map_err(ExtractError::Io)?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut chunk = vec![0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut chunk).map_err(ExtractError::Io)?;
            if n == 0 {
                break;
            }
            hasher.update(&chunk[..n]);
        }
        let checksum = hex::encode(hasher.finalize());

        let mime_type = mime_guess::from_path(path)
            .first_or(mime_guess::mime::APPLICATION_OCTET_STREAM)
            .to_string();

        Ok(Some(Extracted::Text {
            chunks: Vec::new(),
            checksum,
            mime_type,
        }))
    }
}

// ---------------------------------------------------------------------------
// Text extractor
// ---------------------------------------------------------------------------

fn chunk_text(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let chunk_size = 300;
    let overlap = 50;
    
    let mut i = 0;
    while i < words.len() {
        let end = std::cmp::min(i + chunk_size, words.len());
        chunks.push(words[i..end].join(" "));
        if end == words.len() {
            break;
        }
        i += chunk_size - overlap;
    }
    
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

fn extract_text(path: &Path, config: &Config) -> Result<Option<Extracted>, ExtractError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ext == "pdf" {
        return extract_pdf(path, config).map(Some);
    }

    let file = File::open(path).map_err(ExtractError::Io)?;
    let mut reader = BufReader::new(file);

    let mut hasher = Sha256::new();
    let mut body_buf = Vec::new();

    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut chunk).map_err(ExtractError::Io)?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
        body_buf.extend_from_slice(&chunk[..n]);
    }

    let checksum = hex::encode(hasher.finalize());

    let body = match String::from_utf8(body_buf) {
        Ok(s) => s,
        Err(_) => {
            debug!(path = %path.display(), "skipping file with invalid UTF-8");
            return Ok(None);
        }
    };

    let mut mime_type = mime_guess::from_path(path)
        .first_or_text_plain()
        .to_string();

    if mime_type == "application/octet-stream" && TEXT_EXTENSIONS.contains(&ext.as_str()) {
        mime_type = "text/plain".to_string();
    }

    debug!(
        path = %path.display(),
        body_len = body.len(),
        "text extracted"
    );

    let chunks = chunk_text(&body);

    Ok(Some(Extracted::Text { chunks, checksum, mime_type }))
}

// ---------------------------------------------------------------------------
// PDF extractor (Hybrid)
// ---------------------------------------------------------------------------

fn extract_pdf(path: &Path, config: &Config) -> Result<Extracted, ExtractError> {
    let raw = std::fs::read(path).map_err(ExtractError::Io)?;
    let checksum = hex::encode(Sha256::digest(&raw));
    let mime_type = mime_guess::from_path(path).first_or_text_plain().to_string();

    use pdfium_render::prelude::*;
    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
            &config.model_cache_dir()
        ))
        .or_else(|_| Pdfium::bind_to_system_library())
        .map_err(|e| ExtractError::PdfDecode(format!("Pdfium bind error: {}", e)))?
    );

    let document = pdfium.load_pdf_from_file(path, None)
        .map_err(|e| ExtractError::PdfDecode(format!("Pdfium load error: {}", e)))?;

    // CRITICAL FIX: Force a solid white background so the OCR doesn't read black-on-black!
    let render_config = PdfRenderConfig::new()
        .set_target_width(1500)
        .set_clear_color(PdfColor::new(255, 255, 255, 255));

    let mut body = String::new();
    let mut ocr_engine: Option<ocrs::OcrEngine> = None;
    let mut pages_ocred = 0;

    for (i, page) in document.pages().iter().enumerate() {
        let mut page_text = page.text().map(|t| t.all()).unwrap_or_default();

        // Low-Hardware OCR Fallback: If <50 chars, assume scanned
        if page_text.trim().len() < 50 {
            if pages_ocred < 5 { // GUARDRAIL: Max 5 pages of OCR to save CPU
                if ocr_engine.is_none() {
                    let detection_model_path = config.model_cache_dir().join("text-detection.rten");
                    let recognition_model_path = config.model_cache_dir().join("text-recognition.rten");

                    let detection_model = rten::Model::load_file(&detection_model_path)
                        .map_err(|e| ExtractError::PdfDecode(format!("Failed to load detection model: {}", e)))?;
                    let recognition_model = rten::Model::load_file(&recognition_model_path)
                        .map_err(|e| ExtractError::PdfDecode(format!("Failed to load recognition model: {}", e)))?;

                    let engine = ocrs::OcrEngine::new(ocrs::OcrEngineParams {
                        detection_model: Some(detection_model),
                        recognition_model: Some(recognition_model),
                        ..Default::default()
                    }).map_err(|e| ExtractError::PdfDecode(format!("Failed to init OCR engine: {}", e)))?;

                    ocr_engine = Some(engine);
                }

                // Render page and run through Neural Networks with explicit error tracking
                match page.render_with_config(&render_config) {
                    Ok(bitmap) => {
                        let dynamic_image = bitmap.as_image();
                        let img = dynamic_image.into_rgb8();
                        match ocrs::ImageSource::from_bytes(img.as_raw(), img.dimensions()) {
                            Ok(img_source) => {
                                let engine = ocr_engine.as_ref().unwrap();
                                match engine.prepare_input(img_source) {
                                    Ok(ocr_input) => {
                                        match engine.detect_words(&ocr_input) {
                                            Ok(word_rects) => {
                                                let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
                                                match engine.recognize_text(&ocr_input, &line_rects) {
                                                    Ok(lines) => {
                                                        let mut extracted_words = 0;
                                                        for line in lines.iter().flatten() {
                                                            page_text.push_str(&line.to_string());
                                                            page_text.push('\n');
                                                            extracted_words += 1;
                                                        }
                                                        tracing::info!(path = %path.display(), page = i, words = extracted_words, "OCR successful");
                                                    }
                                                    Err(e) => tracing::warn!("OCR recognize_text failed: {:?}", e),
                                                }
                                            }
                                            Err(e) => tracing::warn!("OCR detect_words failed: {:?}", e),
                                        }
                                    }
                                    Err(e) => tracing::warn!("OCR prepare_input failed: {:?}", e),
                                }
                            }
                            Err(e) => tracing::warn!("OCR ImageSource failed: {:?}", e),
                        }
                    }
                    Err(e) => tracing::warn!("PDFium render failed: {:?}", e),
                }
                pages_ocred += 1;
            }
        }

        body.push_str(&page_text);
        body.push('\n');
    }

    drop(ocr_engine);

    let chunks = chunk_text(&body);

    Ok(Extracted::Text { chunks, checksum, mime_type })
}
// ---------------------------------------------------------------------------
// Image extractor (CLIP stub)
// ---------------------------------------------------------------------------

/// Decode a supported image file to an RGB8 `DynamicImage`.
///
/// The decoded image is passed to `embed::image_embed()` which resizes it to
/// 224×224, normalises pixel values, and runs CLIP inference.
///
/// # Phase 3 scope
/// Decoding is complete.  The CLIP model path and normalisation constants are
/// wired in `embed/mod.rs`.  Full CLIP training / fine-tuning is out of scope.
fn extract_image(path: &Path) -> Result<Extracted, ExtractError> {
    // Hash the raw bytes first for change-detection.
    let raw = std::fs::read(path).map_err(ExtractError::Io)?;
    let checksum = hex::encode(Sha256::digest(&raw));

    // Decode to DynamicImage.
    let image = image::load_from_memory(&raw)
        .map_err(|e| ExtractError::ImageDecode(e.to_string()))?;

    let mime_type = mime_guess::from_path(path)
        .first_or(mime_guess::mime::IMAGE_JPEG)
        .to_string();

    debug!(
        path = %path.display(),
        width = image.width(),
        height = image.height(),
        "image decoded"
    );

    Ok(Extracted::Image { image, checksum, mime_type })
}

// ---------------------------------------------------------------------------
// Binary sniff
// ---------------------------------------------------------------------------

fn is_binary(path: &Path) -> Result<bool, ExtractError> {
    let mut file = File::open(path).map_err(ExtractError::Io)?;
    let mut buf = vec![0u8; SNIFF_BYTES];
    let n = file.read(&mut buf).map_err(ExtractError::Io)?;
    // Presence of a null byte is a strong indicator of binary content.
    Ok(buf[..n].contains(&0u8))
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("image decode error: {0}")]
    ImageDecode(String),

    #[error("pdf decode error: {0}")]
    PdfDecode(String),
}
