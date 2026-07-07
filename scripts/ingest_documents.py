#!/usr/bin/env python
"""CLI: copy/ingest local guideline files into the expected data layout.

Usage:
    python scripts/ingest_documents.py --file my_guideline.pdf --category general
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

_CATEGORY_DIRS = {
    "who": "data/guidelines/who",
    "cdc": "data/guidelines/cdc",
    "general": "data/guidelines/general",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a local guideline document into the ingestion directory."
    )
    parser.add_argument("--file", required=True, help="Path to the source file to ingest")
    parser.add_argument(
        "--category",
        choices=list(_CATEGORY_DIRS.keys()),
        default="general",
        help="Which loader category this document belongs to",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.file)
    if not source_path.exists():
        print(f"File not found: {source_path}")
        sys.exit(1)

    destination_dir = Path(_CATEGORY_DIRS[args.category])
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name

    shutil.copy2(source_path, destination_path)
    print(f"Copied {source_path} -> {destination_path}")
    print("Run scripts/build_index.py to index the new document(s).")


if __name__ == "__main__":
    main()
