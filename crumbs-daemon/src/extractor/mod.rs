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
use tracing::{debug, warn};

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
        /// UTF-8 content, capped at [`Config::text_read_limit_bytes`].
        body: String,
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
const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "rst", "adoc",
    "rs",  "py", "js",  "ts",  "jsx", "tsx",
    "go",  "c",  "cpp", "h",   "hpp", "java",
    "cs",  "rb", "php", "sh",  "bat", "ps1",
    "toml", "yaml", "yml", "json", "xml", "html", "htm", "css",
    "csv",  "tsv",  "log", "ini", "cfg", "conf",
    "tex", "bib", "pdf",
];

/// Extensions we treat as images and route to the CLIP stub.
const IMAGE_EXTENSIONS: &[&str] = &[
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
        extract_text(path, config).map(Some)
    } else if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        extract_image(path).map(Some)
    } else {
        // Unknown extension — sniff the first 8 KB for null bytes.
        if is_binary(path)? {
            debug!(path = %path.display(), "skipping binary file (null-byte sniff)");
            Ok(None)
        } else {
            // Unknown text-like file — attempt text extraction.
            extract_text(path, config).map(Some)
        }
    }
}

// ---------------------------------------------------------------------------
// Text extractor
// ---------------------------------------------------------------------------

fn extract_text(path: &Path, config: &Config) -> Result<Extracted, ExtractError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ext == "pdf" {
        let raw = std::fs::read(path).map_err(ExtractError::Io)?;
        let checksum = hex::encode(Sha256::digest(&raw));
        let mime_type = mime_guess::from_path(path)
            .first_or_text_plain()
            .to_string();

        let body_result = {
            let _silencer = StdoutSilencer::new();
            pdf_extract::extract_text(path)
        };
        
        let mut body = match body_result {
            Ok(text) => text,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "pdf text extraction failed, indexing metadata only");
                String::new()
            }
        };

        if body.len() > config.text_read_limit_bytes {
            let mut end = config.text_read_limit_bytes;
            while end > 0 && !body.is_char_boundary(end) {
                end -= 1;
            }
            body.truncate(end);
        }

        debug!(path = %path.display(), body_len = body.len(), "PDF processed");
        return Ok(Extracted::Text { body, checksum, mime_type });
    }

    let file = File::open(path).map_err(ExtractError::Io)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut reader = BufReader::new(file);

    // ------------------------------------------------------------------
    // Compute SHA-256 while reading so we don't open the file twice.
    // We need to hash the full file for accurate change-detection, but
    // we only store `text_read_limit_bytes` worth of content.
    // ------------------------------------------------------------------
    let limit = config.text_read_limit_bytes;
    let mut hasher = Sha256::new();
    let mut body_buf = Vec::with_capacity(limit.min(file_len as usize + 1));
    let mut total_read: usize = 0;

    // Read in 64 KB chunks.  Hash everything, accumulate only up to `limit`.
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut chunk).map_err(ExtractError::Io)?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
        if total_read < limit {
            let take = n.min(limit - total_read);
            body_buf.extend_from_slice(&chunk[..take]);
        }
        total_read += n;
    }

    let checksum = hex::encode(hasher.finalize());

    // Convert to UTF-8, replacing invalid sequences rather than erroring.
    let body = String::from_utf8_lossy(&body_buf).into_owned();

    let mime_type = mime_guess::from_path(path)
        .first_or_text_plain()
        .to_string();

    debug!(
        path = %path.display(),
        body_len = body.len(),
        total_bytes = total_read,
        truncated = total_read > limit,
        "text extracted"
    );

    Ok(Extracted::Text { body, checksum, mime_type })
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
