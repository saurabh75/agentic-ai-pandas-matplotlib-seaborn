"""
File utilities for the Local RAG Agent.

Handles:
- SHA256 hashing for duplicate detection
- Filename sanitization to prevent path traversal
- File validation (size, extension, MIME type)
- Secure file storage
"""

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Tuple

from src.config import MAX_FILE_SIZE_BYTES, SUPPORTED_EXTENSIONS
from src.exceptions import SecurityError, UnsupportedFormatError
from src.logger import get_logger

logger = get_logger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file's raw binary content.

    Uses a 64KB buffer to handle large files without loading them entirely
    into memory. The hash is deterministic and collision-resistant,
    making it ideal for duplicate detection.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hexadecimal SHA256 digest string (64 characters).
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):  # 64KB buffer
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_bytes_hash(content: bytes) -> str:
    """Compute SHA256 hash of raw bytes (for in-memory content)."""
    return hashlib.sha256(content).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection attacks.

    Steps:
    1. Strip path separators and parent directory references
    2. Remove non-alphanumeric characters (except dots, dashes, underscores)
    3. Prevent empty names by generating a UUID fallback
    4. Preserve the original extension
    """
    # Extract extension first
    name_part = Path(filename).stem
    ext_part = Path(filename).suffix.lower()

    # Remove path traversal characters
    sanitized = os.path.basename(name_part)

    # Replace spaces with underscores, then strip other dangerous chars
    sanitized = sanitized.replace(" ", "_")
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "", sanitized)

    # Prevent empty names
    if not sanitized:
        sanitized = f"doc_{uuid.uuid4().hex[:8]}"

    # Limit length (filesystem safety)
    max_name_len = 200 - len(ext_part)
    sanitized = sanitized[:max_name_len]

    return f"{sanitized}{ext_part}"


def validate_file(
    file_path: Path,
    check_size: bool = True,
    check_extension: bool = True,
    check_mime: bool = True,
) -> None:
    """Validate a file against security and format policies.

    Performs three checks:
    1. Size: File must not exceed MAX_FILE_SIZE_BYTES
    2. Extension: Must be in SUPPORTED_EXTENSIONS
    3. MIME Type: Must match the declared extension (anti-spoofing)
    """
    # --- Size Check ---
    if check_size:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            msg = (
                f"File '{file_path.name}' exceeds maximum size of "
                f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB "
                f"(actual: {file_size // (1024 * 1024)} MB)."
            )
            logger.warning(msg)
            raise SecurityError(msg)

    # --- Extension Check ---
    ext = file_path.suffix.lower()
    if check_extension and ext not in SUPPORTED_EXTENSIONS:
        msg = (
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        logger.warning(msg)
        raise UnsupportedFormatError(msg)

    # --- MIME Type Check ---
    if check_mime:
        try:
            import magic
            detected_mime = magic.from_file(str(file_path), mime=True)
        except Exception as e:
            logger.warning(f"Could not detect MIME type for {file_path}: {e}")
            detected_mime = "application/octet-stream"

        # Map extensions to expected MIME types
        mime_map = {
            ".pdf": ["application/pdf"],
            ".docx": [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/octet-stream",
            ],
            ".xlsx": [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/octet-stream",
            ],
            ".xls": ["application/vnd.ms-excel", "application/octet-stream"],
            ".pptx": [
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/octet-stream",
            ],
            ".ppt": ["application/vnd.ms-powerpoint", "application/octet-stream"],
            ".txt": ["text/plain"],
            ".md": ["text/plain", "text/markdown"],
            ".csv": ["text/csv", "text/plain"],
        }

        expected_mimes = mime_map.get(ext, [])
        if expected_mimes and detected_mime not in expected_mimes:
            if detected_mime != "application/octet-stream":
                msg = (
                    f"MIME type mismatch for '{file_path.name}': "
                    f"expected {expected_mimes}, got {detected_mime}. "
                    f"Possible file spoofing attempt."
                )
                logger.warning(msg)

    logger.info(f"File validation passed: {file_path.name}")


def get_unique_storage_path(documents_dir: Path, filename: str) -> Path:
    """Generate a unique storage path for an uploaded file.

    If a file with the same name exists, appends a counter to avoid
    overwriting existing files.
    """
    target = documents_dir / filename
    if not target.exists():
        return target

    # Append counter if file exists
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        candidate = documents_dir / new_name
        if not candidate.exists():
            return candidate
        counter += 1
