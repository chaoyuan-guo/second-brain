from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np


class DummyEmbeddingClient:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self.embeddings = self

    def create(self, model: str, input: list[str]) -> SimpleNamespace:
        data = []
        for index, _ in enumerate(input):
            vector = [float(index + 1)] * self._dimension
            data.append(SimpleNamespace(embedding=vector))
        return SimpleNamespace(data=data)


class FakeSettings:
    def __init__(self, base_dir: Path, index_path: Path, metadata_path: Path) -> None:
        self.base_dir = base_dir
        self.faiss_index_path = index_path
        self.faiss_metadata_path = metadata_path


def test_update_notes_index_replaces_vectors(monkeypatch, tmp_path: Path) -> None:
    from backend.app.services import notes_index

    dimension = 3
    notes_root = tmp_path / "data" / "notes" / "my_markdowns"
    notes_root.mkdir(parents=True, exist_ok=True)

    index_path = tmp_path / "data" / "indexes" / "my_notes.index"
    metadata_path = tmp_path / "data" / "notes" / "my_notes_metadata.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    metadata = [
        {
            "vector_id": 0,
            "source_path": "old.md",
            "chunk_index": 0,
            "heading_path": "Old",
            "document_title": "Old",
            "chunk_type": "detail",
            "text": "old",
        },
        {
            "vector_id": 1,
            "source_path": "keep.md",
            "chunk_index": 0,
            "heading_path": "Keep",
            "document_title": "Keep",
            "chunk_type": "detail",
            "text": "keep",
        },
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(
        notes_index,
        "settings",
        FakeSettings(tmp_path, index_path, metadata_path),
    )
    monkeypatch.setattr(notes_index, "client", DummyEmbeddingClient(dimension))

    result = notes_index.update_notes_index_from_upload("old.md", b"New content")

    assert result["replaced"] is False
    assert result["chunks_added"] == 2
    assert result["removed_vectors"] == 1

    updated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(updated_metadata) == result["total_vectors"]
    assert any(record["source_path"] == "keep.md" for record in updated_metadata)
    assert any(record["source_path"] == "old.md" for record in updated_metadata)
    assert [record["vector_id"] for record in updated_metadata] == list(
        range(len(updated_metadata))
    )

    updated_index = faiss.read_index(str(index_path))
    assert updated_index.ntotal == len(updated_metadata)
