"""Prints every chunk in the Chroma index with its ID, source file and text, for labelling gold chunks.

Usage: python evaluation/print_chunks.py > chunks.txt

Builds the index first if it is missing. Log messages go to stderr, so a redirected file holds
only the chunks.
"""
import logging
import os
import sys
import textwrap
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest import get_client, load_or_build_collection

WIDTH = 100


def chunk_number(chunk_id: str) -> int:
    """Sorts chunk_2 before chunk_10."""
    return int(chunk_id.rsplit("_", 1)[1])


def main():
    collection = load_or_build_collection(get_client(), None)
    result = collection.get(include=["documents", "metadatas"])
    rows = sorted(zip(result["ids"], result["metadatas"], result["documents"]), key=lambda row: chunk_number(row[0]))
    per_source = Counter(meta["source"] for _, meta, _ in rows)

    print(f"{len(rows)} chunks from {len(per_source)} documents\n")
    for chunk_id, meta, text in rows:
        print("=" * WIDTH)
        print(f"{chunk_id}  |  {meta['source']}  |  part {meta['chunk_index'] + 1} of {per_source[meta['source']]}")
        print("-" * WIDTH)
        print(textwrap.fill(text, width=WIDTH))
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        main()
    except BrokenPipeError:
        # The reader stopped early (e.g. `| head`, or quitting `less`); exit without a traceback.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
