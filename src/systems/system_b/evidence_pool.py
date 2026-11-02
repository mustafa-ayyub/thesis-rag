import re

from src.systems.system_b.types import EvidenceItem
from src.types import RetrievedChunk


class RepeatedQueryError(ValueError):
    pass


def normalize_query(query: str) -> str:
    return " ".join(re.findall(r"\w+", query.lower()))


class EvidencePool:
    def __init__(self, max_final_chunks: int) -> None:
        if max_final_chunks < 1:
            raise ValueError("max_final_chunks must be at least 1")
        self._max_final_chunks = max_final_chunks
        self._items_by_chunk_id: dict[str, EvidenceItem] = {}
        self._items_by_evidence_id: dict[str, EvidenceItem] = {}
        self._evidence_ids: list[str] = []
        self._queries: set[str] = set()
        self._query_history: list[str] = []

    @property
    def evidence_ids(self) -> list[str]:
        return list(self._evidence_ids)

    @property
    def queries(self) -> list[str]:
        return list(self._query_history)

    @property
    def all_chunks(self) -> list[RetrievedChunk]:
        return [self._items_by_evidence_id[evidence_id].chunk for evidence_id in self._evidence_ids]

    def mark_query(self, query: str) -> None:
        normalized = normalize_query(query)
        if not normalized:
            raise ValueError("query must not be empty")
        if normalized in self._queries:
            raise RepeatedQueryError(f"Query already used: {query}")
        self._queries.add(normalized)
        self._query_history.append(query.strip())

    def add_chunks(
        self,
        call_number: int,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[str], list[str]]:
        new_evidence_ids: list[str] = []
        duplicate_chunk_ids: list[str] = []

        for chunk in chunks:
            existing = self._items_by_chunk_id.get(chunk.chunk_id)
            if existing is not None:
                existing.duplicate_count += 1
                duplicate_chunk_ids.append(chunk.chunk_id)
                continue

            evidence_id = f"E{len(self._evidence_ids) + 1}"
            item = EvidenceItem(
                evidence_id=evidence_id,
                chunk=chunk,
                first_call_number=call_number,
            )
            self._items_by_chunk_id[chunk.chunk_id] = item
            self._items_by_evidence_id[evidence_id] = item
            self._evidence_ids.append(evidence_id)
            new_evidence_ids.append(evidence_id)

        return new_evidence_ids, duplicate_chunk_ids

    def get_chunk(self, evidence_id: str) -> RetrievedChunk:
        try:
            return self._items_by_evidence_id[evidence_id].chunk
        except KeyError as exc:
            raise ValueError(f"Unknown evidence ID: {evidence_id}") from exc

    def select_chunks(self, evidence_ids: list[str]) -> list[RetrievedChunk]:
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("selected evidence IDs must be unique")
        if len(evidence_ids) > self._max_final_chunks:
            raise ValueError(
                f"Selected {len(evidence_ids)} chunks; maximum is {self._max_final_chunks}"
            )
        return [self.get_chunk(evidence_id) for evidence_id in evidence_ids]

    def render(self) -> str:
        if not self._evidence_ids:
            return "No evidence has been retrieved yet."

        blocks: list[str] = []
        for evidence_id in self._evidence_ids:
            item = self._items_by_evidence_id[evidence_id]
            chunk = item.chunk
            blocks.append(
                f"{evidence_id} (call {item.first_call_number}, {chunk.doc_name}, "
                f"p.{chunk.page_number}, section: \"{chunk.section_title}\") "
                f"{chunk.text}"
            )
        return "\n\n".join(blocks)
