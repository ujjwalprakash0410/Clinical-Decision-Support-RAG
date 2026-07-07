#!/usr/bin/env python
"""CLI: build (or rebuild) the FAISS index from configured sources.

Usage:
    python scripts/build_index.py --sources who cdc guideline --rebuild
    python scripts/build_index.py --sources pubmed --query-terms "atrial fibrillation" "anticoagulation"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.dependencies.di import get_indexing_service  # noqa: E402
from app.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clinical RAG vector index.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["who", "cdc", "guideline"],
        help="Loader source names to run (pubmed, who, cdc, guideline)",
    )
    parser.add_argument(
        "--query-terms",
        nargs="*",
        default=[],
        help="Search terms for online loaders such as PubMed",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the existing index and rebuild from scratch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexing_service = get_indexing_service()

    chunk_count, sources_used = indexing_service.index_sources(
        source_names=args.sources,
        query_terms=args.query_terms,
        rebuild=args.rebuild,
    )

    print(f"Indexed {chunk_count} chunks from sources: {sources_used}")
    if chunk_count == 0:
        print(
            "No documents were found. Place files under data/guidelines/<who|cdc|general>/ "
            "or pass --query-terms for the PubMed loader."
        )


if __name__ == "__main__":
    main()
