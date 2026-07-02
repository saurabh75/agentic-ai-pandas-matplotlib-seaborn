"""
Document processing pipeline for the Local RAG Agent.

Handles:
- File loading via LangChain loaders
- Text chunking with configurable overlap
- Metadata enrichment
- Deduplication via SHA256 hashing
- Batch insertion into ChromaDB
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Type, Dict, Any

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import os as _os
import time
EMBED_BATCH_SIZE = int(_os.getenv("EMBED_BATCH_SIZE", "16"))
MAX_CHUNKS_PER_DOC = int(_os.getenv("MAX_CHUNKS_PER_DOC", "2000"))
TABULAR_AS_SCHEMA_ONLY = _os.getenv("TABULAR_AS_SCHEMA_ONLY", "true").lower() == "true"
TABULAR_EXTS = {".csv", ".xlsx", ".xls"}

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR
from src.models.document import DocumentMetadata, IngestedDocument
from src.services.vector_store import get_vector_store, is_document_indexed
from src.services import data_store  # Patch v5
from src.utils.file_utils import compute_file_hash, validate_file
from src.logger import get_logger
from src.exceptions import (
    DocumentProcessingError,
    EmptyDocumentError,
    DuplicateDocumentError,
)

logger = get_logger(__name__)

LOADER_MAP: Dict[str, Type] = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
}


def get_loader_for_file(file_path: Path) -> Optional[Any]:
    """Get the appropriate LangChain loader for a file."""
    ext = file_path.suffix.lower()
    loader_class = LOADER_MAP.get(ext)
    if not loader_class:
        return None
    if ext in {".txt", ".md", ".csv"}:
        return loader_class(str(file_path), encoding="utf-8")
    return loader_class(str(file_path))


def extract_page_number(doc: Document) -> Optional[int]:
    """Extract page number from document metadata."""
    meta = doc.metadata or {}
    if "page" in meta:
        return int(meta["page"]) + 1
    if "page_number" in meta:
        return int(meta["page_number"])
    return None


def process_single_file(file_path: Path) -> IngestedDocument:
    """Process a single document file through the full pipeline."""
    filename = file_path.name
    logger.info(f"Processing: {filename}")

    try:
        validate_file(file_path)
    except Exception as e:
        raise DocumentProcessingError(f"Validation failed for {filename}: {e}") from e

    document_hash = compute_file_hash(file_path)
    logger.debug(f"Hash for {filename}: {document_hash[:16]}...")

    if is_document_indexed(document_hash):
        logger.info(f"Skipping duplicate document: {filename}")
        raise DuplicateDocumentError(f"'{filename}' has already been indexed.")

    ext = file_path.suffix.lower()

    # -------------------------------------------------------------------
    # Patch v5: tabular files → parquet + single "schema" chunk
    # Avoids ballooning a 54 KB CSV into 2000+ text chunks that overload
    # Ollama's embedding worker on 6 GB VRAM.
    # -------------------------------------------------------------------
    if TABULAR_AS_SCHEMA_ONLY and ext in TABULAR_EXTS:
        entry = data_store.save_tabular(file_path, document_hash)
        if entry is None:
            raise DocumentProcessingError(f"Could not read tabular file: {filename}")

        summary_text = data_store.schema_summary(entry)
        upload_date = datetime.now().isoformat()
        schema_doc = Document(
            page_content=summary_text,
            metadata=DocumentMetadata(
                filename=filename,
                file_path=str(file_path.resolve()),
                page_number=None,
                upload_date=upload_date,
                document_hash=document_hash,
                chunk_index=0,
            ).model_dump(),
        )
        try:
            get_vector_store().add_documents([schema_doc])
        except Exception as e:
            raise DocumentProcessingError(f"Failed to index schema chunk for {filename}: {e}") from e
        logger.info(f"[tabular] {filename}: {entry['rows']} rows registered + 1 schema chunk")
        return IngestedDocument(
            filename=filename,
            document_hash=document_hash,
            total_chunks=1,
            upload_date=upload_date,
            status="success",
        )

    loader = get_loader_for_file(file_path)
    if not loader:
        raise DocumentProcessingError(f"No loader available for {filename}")

    try:
        documents = loader.load()
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        raise DocumentProcessingError(f"Could not parse {filename}: {e}") from e

    total_text = "".join(doc.page_content for doc in documents)
    if not total_text.strip():
        raise EmptyDocumentError(f"'{filename}' contains no extractable text.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        logger.warning(
            f"{filename}: {len(chunks)} chunks exceeds MAX_CHUNKS_PER_DOC "
            f"({MAX_CHUNKS_PER_DOC}); truncating."
        )
        chunks = chunks[:MAX_CHUNKS_PER_DOC]
    logger.info(f"Split {filename} into {len(chunks)} chunks")

    upload_date = datetime.now().isoformat()

    for i, chunk in enumerate(chunks):
        page_num = extract_page_number(chunk)
        chunk.metadata = DocumentMetadata(
            filename=filename,
            file_path=str(file_path.resolve()),
            page_number=page_num,
            upload_date=upload_date,
            document_hash=document_hash,
            chunk_index=i,
        ).model_dump()

    try:
        vectorstore = get_vector_store()
        total = len(chunks)
        batch_size = max(1, EMBED_BATCH_SIZE)
        indexed = 0
        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            attempts = 0
            while True:
                try:
                    vectorstore.add_documents(batch)
                    break
                except Exception as be:
                    attempts += 1
                    msg = str(be).lower()
                    transient = any(s in msg for s in (
                        "refused", "reset", "timeout", "timed out",
                        "connection", "eof", "broken pipe", "502", "503", "504",
                    ))
                    if attempts >= 3 or not transient:
                        raise
                    wait = 2 ** attempts
                    logger.warning(
                        f"Embedding batch {start//batch_size + 1} failed "
                        f"(attempt {attempts}/3): {be}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
            indexed += len(batch)
            if (start // batch_size) % 10 == 0 or indexed == total:
                logger.info(f"  embedded {indexed}/{total} chunks from {filename}")
        logger.info(f"Indexed {total} chunks from {filename}")
    except Exception as e:
        raise DocumentProcessingError(f"Failed to index {filename}: {e}") from e

    return IngestedDocument(
        filename=filename,
        document_hash=document_hash,
        total_chunks=len(chunks),
        upload_date=upload_date,
        status="success",
    )


def process_directory(directory: Optional[Path] = None) -> List[IngestedDocument]:
    """Process all supported documents in a directory."""
    target_dir = directory or DOCUMENTS_DIR
    if not target_dir.exists():
        logger.warning(f"Documents directory does not exist: {target_dir}")
        return []

    results: List[IngestedDocument] = []
    for root, _, files in os.walk(target_dir):
        for filename in sorted(files):
            file_path = Path(root) / filename
            if filename.startswith("."):
                continue
            ext = file_path.suffix.lower()
            if ext not in LOADER_MAP:
                continue
            try:
                result = process_single_file(file_path)
                results.append(result)
            except DuplicateDocumentError as e:
                logger.info(str(e))
            except EmptyDocumentError as e:
                logger.warning(str(e))
            except DocumentProcessingError as e:
                logger.error(str(e))
            except Exception as e:
                logger.error(f"Unexpected error processing {filename}: {e}")

    logger.info(f"Successfully processed {len(results)} documents")
    return results
