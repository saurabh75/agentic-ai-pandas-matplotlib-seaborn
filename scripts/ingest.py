#!/usr/bin/env python3
"""
CLI script for ingesting documents into the Local RAG Agent.

Usage:
    python scripts/ingest.py                    # Process documents/ directory
    python scripts/ingest.py --dir /path/to/docs # Process specific directory
    python scripts/ingest.py --file report.pdf   # Process single file
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.document_processor import process_directory, process_single_file
from src.config import DOCUMENTS_DIR
from src.logger import get_logger
from src.exceptions import DuplicateDocumentError, EmptyDocumentError, DocumentProcessingError

logger = get_logger("ingest")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Local RAG Agent vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Ingest all documents in ./documents/
  %(prog)s --dir ~/reports          # Ingest from ~/reports
  %(prog)s --file contract.pdf      # Ingest single file
        """,
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory containing documents to ingest (default: ./documents/)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Single file to ingest",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Single file mode
    if args.file:
        if not args.file.exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)

        try:
            result = process_single_file(args.file)
            print(f"\n✅ Successfully ingested: {result.filename}")
            print(f"   Chunks: {result.total_chunks}")
            print(f"   Hash: {result.document_hash[:16]}...")
        except DuplicateDocumentError as e:
            print(f"\n⚠️  {e}")
        except (EmptyDocumentError, DocumentProcessingError) as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
        return

    # Directory mode
    target_dir = args.dir or DOCUMENTS_DIR
    print(f"📁 Scanning directory: {target_dir}")

    results = process_directory(target_dir)

    # Print summary
    print(f"\n{'='*50}")
    print(f"INGESTION SUMMARY")
    print(f"{'='*50}")
    print(f"Documents successfully indexed: {len(results)}")

    if results:
        total_chunks = sum(r.total_chunks for r in results)
        print(f"Total chunks created: {total_chunks}")
        print("\nIndexed files:")
        for r in results:
            print(f"  ✅ {r.filename} ({r.total_chunks} chunks)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
