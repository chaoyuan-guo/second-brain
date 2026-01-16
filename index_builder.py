"""Build a FAISS index for local Markdown notes using OpenAI embeddings."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
INDEX_DIR = DATA_DIR / "indexes"
DEFAULT_SOURCE_DIR = NOTES_DIR / "my_markdowns"
DEFAULT_INDEX_PATH = INDEX_DIR / "my_notes.index"
DEFAULT_METADATA_PATH = NOTES_DIR / "my_notes_metadata.json"
DEFAULT_API_BASE_URL = "https://space.ai-builders.com/backend/v1"


@dataclass
class Chunk:
    """Represents a chunked segment of a markdown document."""

    text: str
    source_path: str
    chunk_index: int
    heading_path: str
    document_title: str
    chunk_type: str


@dataclass
class Section:
    """A logical section derived from Markdown headings."""

    heading_path: str
    text: str
    level: int


def discover_markdown_files(root: Path) -> List[Path]:
    """Return all markdown files under ``root`` (recursive)."""

    if not root.exists():
        raise FileNotFoundError(f"Source directory {root} does not exist")
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Split ``text`` into chunks measured in words with overlap."""

    words = text.split()
    if not words:
        return []

    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: List[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))

        if start + chunk_size >= len(words):
            break

    return chunks


def summarize_text(text: str, word_limit: int) -> str:
    """Return the first ``word_limit`` words of ``text`` as a lightweight summary."""

    if word_limit <= 0:
        return ""
    words = text.split()
    if not words:
        return ""
    return " ".join(words[:word_limit])


def split_markdown_sections(text: str, fallback_title: str) -> Tuple[List[Section], str]:
    """Chunk markdown by heading hierarchy to preserve semantic boundaries."""

    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
    sections: List[Section] = []
    heading_stack: List[str] = []
    buffer: List[str] = []
    document_title = ""

    def current_path() -> str:
        return " > ".join(heading_stack) if heading_stack else fallback_title

    def flush_section() -> None:
        content = "\n".join(buffer).strip()
        if not content:
            buffer.clear()
            return
        sections.append(
            Section(
                heading_path=current_path(),
                text=content,
                level=len(heading_stack),
            )
        )
        buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = heading_pattern.match(line)
        if match:
            flush_section()
            level = len(match.group(1))
            title = match.group(2).strip() or "Untitled"
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            if level == 1 and not document_title:
                document_title = title
            continue
        buffer.append(raw_line)

    flush_section()

    if not sections and text.strip():
        sections.append(Section(heading_path=fallback_title, text=text.strip(), level=0))

    if not document_title:
        document_title = heading_stack[0] if heading_stack else fallback_title

    return sections, document_title


def load_chunks(
    files: Sequence[Path],
    root: Path,
    chunk_size: int,
    chunk_overlap: int,
    summary_word_limit: int,
) -> List[Chunk]:
    """Read all files and split them into summary/detail chunks per section."""

    collected: List[Chunk] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")

        relative_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        sections, document_title = split_markdown_sections(text, fallback_title=path.stem)
        chunk_counter = 0
        for section in sections:
            summary_text = summarize_text(section.text, word_limit=summary_word_limit)
            if summary_text:
                collected.append(
                    Chunk(
                        text=summary_text,
                        source_path=relative_path,
                        chunk_index=chunk_counter,
                        heading_path=section.heading_path,
                        document_title=document_title,
                        chunk_type="summary",
                    )
                )
                chunk_counter += 1

            detail_chunks = chunk_text(
                section.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ) or ([section.text] if section.text else [])

            for detail in detail_chunks:
                collected.append(
                    Chunk(
                        text=detail,
                        source_path=relative_path,
                        chunk_index=chunk_counter,
                        heading_path=section.heading_path,
                        document_title=document_title,
                        chunk_type="detail",
                    )
                )
                chunk_counter += 1

    return collected


def batched(iterable: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def embed_chunks(
    client: OpenAI,
    chunks: List[Chunk],
    model: str,
    batch_size: int,
) -> np.ndarray:
    """Convert text chunks into vectors using OpenAI embeddings API."""

    vectors: List[List[float]] = []
    texts = [chunk.text for chunk in chunks]

    for batch in batched(texts, batch_size):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    if not vectors:
        raise RuntimeError("No embeddings were generated. Ensure markdown files have content.")

    return np.array(vectors, dtype="float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Create an IndexFlatL2 FAISS index populated with ``vectors``."""

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index


def save_metadata(chunks: List[Chunk], path: Path) -> None:
    """Persist chunk metadata for downstream retrieval."""

    records = [
        {
            "vector_id": idx,
            "source_path": chunk.source_path,
            "chunk_index": chunk.chunk_index,
            "heading_path": chunk.heading_path,
            "document_title": chunk.document_title,
            "chunk_type": chunk.chunk_type,
            "text": chunk.text,
        }
        for idx, chunk in enumerate(chunks)
    ]
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from Markdown notes.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing markdown files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=350,
        help="Approximate number of words per detail chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Word overlap between consecutive detail chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks to embed per API call.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to save the FAISS index.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to store chunk metadata JSON.",
    )
    parser.add_argument(
        "--summary-word-limit",
        type=int,
        default=80,
        help="Number of words to keep in generated summary chunks per section.",
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("SUPER_MIND_API_KEY")
    if not api_key:
        raise EnvironmentError("SUPER_MIND_API_KEY is missing from environment/.env file")

    api_base_url = os.getenv("SUPER_MIND_API_BASE_URL", DEFAULT_API_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=api_base_url)

    source_dir = args.source_dir.expanduser().resolve()
    markdown_files = discover_markdown_files(source_dir)
    if not markdown_files:
        raise RuntimeError(f"No markdown files found in {source_dir}.")

    print(f"Discovered {len(markdown_files)} markdown files.")
    chunks = load_chunks(
        markdown_files,
        root=source_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        summary_word_limit=args.summary_word_limit,
    )

    if not chunks:
        raise RuntimeError("Markdown files produced zero chunks. Check chunk parameters.")

    print(f"Prepared {len(chunks)} chunks. Requesting embeddings from OpenAI API…")
    vectors = embed_chunks(
        client=client,
        chunks=chunks,
        model=args.model,
        batch_size=args.batch_size,
    )

    print(f"Received embeddings with dimension {vectors.shape[1]}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index = build_faiss_index(vectors)
    faiss.write_index(index, str(args.output))
    print(f"Saved FAISS index to {args.output}.")

    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        save_metadata(chunks, args.metadata_output)
        print(f"Saved metadata to {args.metadata_output}.")


if __name__ == "__main__":
    main()
