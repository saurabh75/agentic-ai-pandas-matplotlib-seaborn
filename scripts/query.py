#!/usr/bin/env python3
"""
CLI script for querying the Local RAG Agent.

Usage:
    python scripts/query.py "What is the main topic?"
    python scripts/query.py --session mysession "Follow-up question"
    python scripts/query.py --new-session "Start fresh conversation"
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.retriever import RAGRetriever
from src.services.llm_service import generate_answer, clear_session
from src.logger import get_logger

logger = get_logger("query")


def main():
    parser = argparse.ArgumentParser(
        description="Query the Local RAG Agent knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the key findings?"
  %(prog)s --session project1 "Who is the author?"
  %(prog)s --new-session "Start fresh conversation"
        """,
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to ask the agent",
    )
    parser.add_argument(
        "--session", "-s",
        default="cli_default",
        help="Conversation session ID (default: cli_default)",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Clear conversation history before querying",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source citations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of sources to retrieve (default: 5)",
    )

    args = parser.parse_args()
    question = " ".join(args.question)

    # Clear session if requested
    if args.new_session:
        clear_session(args.session)
        print("🗑️  Conversation history cleared.\n")

    print(f"❓ Question: {question}\n")

    # Initialize retriever
    try:
        retriever = RAGRetriever(retrieval_k=args.top_k)
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        sys.exit(1)

    # Retrieve
    start_time = time.time()
    try:
        results = retriever.retrieve(question)
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        sys.exit(1)

    if not results:
        print("⚠️  No relevant documents found. Try rephrasing your question.")
        sys.exit(0)

    # Generate answer
    try:
        response = generate_answer(
            question=question,
            retrieval_results=results,
            session_id=args.session,
        )
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        sys.exit(1)

    elapsed = (time.time() - start_time) * 1000

    # Display
    print(f"💡 Answer:\n{response.answer}\n")

    if not args.no_sources:
        print("📚 Sources:")
        print("-" * 60)
        for i, src in enumerate(response.sources, 1):
            meta = src.metadata
            page_info = f", page {meta.page_number}" if meta.page_number else ""
            print(f"  {i}. {meta.filename}{page_info}")
            print(f"     Similarity: {src.similarity_score:.3f}", end="")
            if src.reranker_score is not None:
                print(f" | Reranker: {src.reranker_score:.3f}")
            else:
                print()
            # Truncate content for display
            content_preview = src.content[:200].replace("\n", " ")
            if len(src.content) > 200:
                content_preview += "..."
            print(f'     "{content_preview}"')
            print()

    print(f"⏱️  Query time: {elapsed:.0f}ms | Model: {response.model_used}")


if __name__ == "__main__":
    main()
