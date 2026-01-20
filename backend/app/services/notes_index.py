"""Utilities for updating markdown note indexes."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import faiss
import numpy as np

from ..core.config import settings
from ..core.logging import app_logger
from ..repositories.notes import reset_notes_cache
from .clients import client
from .exceptions import ToolExecutionError

logger = app_logger

DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_SUMMARY_WORD_LIMIT = 80
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


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


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into chunks measured in words with overlap."""

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
    """Return the first words of text as a lightweight summary."""

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


def build_chunks_for_text(
    text: str,
    source_path: str,
    fallback_title: str,
    chunk_size: int,
    chunk_overlap: int,
    summary_word_limit: int,
) -> List[Chunk]:
    """Split a markdown text into summary and detail chunks."""

    sections, document_title = split_markdown_sections(text, fallback_title=fallback_title)
    collected: List[Chunk] = []
    chunk_counter = 0

    for section in sections:
        summary_text = summarize_text(section.text, word_limit=summary_word_limit)
        if summary_text:
            collected.append(
                Chunk(
                    text=summary_text,
                    source_path=source_path,
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
                    source_path=source_path,
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


def embed_chunks(chunks: List[Chunk], model: str, batch_size: int) -> np.ndarray:
    """Convert text chunks into vectors using OpenAI embeddings API."""

    texts = [chunk.text for chunk in chunks]
    vectors: List[List[float]] = []

    for batch in batched(texts, batch_size):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    if not vectors:
        raise ToolExecutionError("No embeddings were generated for the upload.")

    return np.array(vectors, dtype="float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Create an IndexFlatL2 FAISS index populated with vectors."""

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index


def _load_existing_index() -> tuple[faiss.Index | None, List[dict[str, Any]]]:
    index_path = settings.faiss_index_path
    metadata_path = settings.faiss_metadata_path

    if not index_path.exists() and not metadata_path.exists():
        return None, []
    if not index_path.exists() or not metadata_path.exists():
        raise ToolExecutionError("Index or metadata file is missing. Rebuild the index first.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolExecutionError("Metadata file is corrupted.") from exc

    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:  # pragma: no cover - faiss raises non-typed errors
        raise ToolExecutionError("Failed to read the existing FAISS index.") from exc

    return index, metadata


def _write_atomic(path: Path, content: str) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def _write_index_atomic(index_path: Path, index: faiss.Index) -> None:
    temp_path = index_path.with_name(f"{index_path.name}.tmp")
    faiss.write_index(index, str(temp_path))
    os.replace(temp_path, index_path)


def update_notes_index_from_upload(file_name: str, raw_bytes: bytes) -> dict[str, Any]:
    """Replace a markdown file, update vectors, and refresh the index on disk."""

    safe_name = Path(file_name).name
    if not safe_name or safe_name != file_name:
        raise ToolExecutionError("Invalid file name.")
    if not safe_name.lower().endswith(".md"):
        raise ToolExecutionError("Only .md files are supported.")

    logger.info(
        "Notes upload received",
        extra={"file_name": safe_name, "bytes": len(raw_bytes)},
    )

    notes_root = settings.base_dir / "data" / "notes" / "my_markdowns"
    notes_root.mkdir(parents=True, exist_ok=True)
    target_path = notes_root / safe_name
    replaced = target_path.exists()

    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw_bytes.decode("utf-8", errors="ignore")

    target_path.write_text(decoded, encoding="utf-8")

    source_path = safe_name
    chunks = build_chunks_for_text(
        decoded,
        source_path=source_path,
        fallback_title=Path(safe_name).stem,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        summary_word_limit=DEFAULT_SUMMARY_WORD_LIMIT,
    )

    if not chunks:
        raise ToolExecutionError("Uploaded markdown produced zero chunks.")

    logger.info(
        "Notes upload chunked",
        extra={"file_name": safe_name, "chunks": len(chunks)},
    )

    new_vectors = embed_chunks(
        chunks,
        model=DEFAULT_EMBEDDING_MODEL,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    index, metadata = _load_existing_index()
    kept_metadata: List[dict[str, Any]] = []
    kept_vectors = np.empty((0, new_vectors.shape[1]), dtype="float32")
    removed_count = 0

    if index is not None and metadata:
        vector_count = min(index.ntotal, len(metadata))
        if index.ntotal != len(metadata):
            logger.warning(
                "Index/metadata length mismatch",
                extra={"index_total": index.ntotal, "metadata_total": len(metadata)},
            )
        if vector_count:
            all_vectors = index.reconstruct_n(0, vector_count)
            kept_positions: List[int] = []
            for position, record in enumerate(metadata[:vector_count]):
                if record.get("source_path") == source_path:
                    removed_count += 1
                    continue
                kept_metadata.append(record)
                kept_positions.append(position)
            if kept_positions:
                kept_vectors = all_vectors[kept_positions]
            else:
                kept_vectors = np.empty((0, all_vectors.shape[1]), dtype="float32")

    if kept_vectors.size:
        if kept_vectors.shape[1] != new_vectors.shape[1]:
            raise ToolExecutionError("Embedding dimension mismatch.")
        combined_vectors = np.concatenate([kept_vectors, new_vectors], axis=0)
    else:
        combined_vectors = new_vectors

    new_index = build_faiss_index(combined_vectors)

    records: List[dict[str, Any]] = []
    for idx, record in enumerate(kept_metadata):
        updated = dict(record)
        updated["vector_id"] = idx
        records.append(updated)

    start_id = len(records)
    for offset, chunk in enumerate(chunks):
        records.append(
            {
                "vector_id": start_id + offset,
                "source_path": chunk.source_path,
                "chunk_index": chunk.chunk_index,
                "heading_path": chunk.heading_path,
                "document_title": chunk.document_title,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
            }
        )

    settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.faiss_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    _write_index_atomic(settings.faiss_index_path, new_index)
    _write_atomic(
        settings.faiss_metadata_path,
        json.dumps(records, ensure_ascii=False, indent=2),
    )
    reset_notes_cache()

    logger.info(
        "Notes index updated from upload",
        extra={
            "file_name": safe_name,
            "replaced": replaced,
            "removed_vectors": removed_count,
            "new_chunks": len(chunks),
            "total_vectors": len(records),
        },
    )

    return {
        "message": f"Uploaded {safe_name} and updated {len(chunks)} chunks.",
        "file_name": safe_name,
        "chunks_added": len(chunks),
        "replaced": replaced,
        "removed_vectors": removed_count,
        "total_vectors": len(records),
    }


__all__ = ["update_notes_index_from_upload"]
