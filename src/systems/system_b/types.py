from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.types import RetrievedChunk


class RetrievalCallRecord(BaseModel):
    call_number: int
    query: str
    new_evidence_ids: list[str]
    duplicate_chunk_ids: list[str]
    latency_ms: int


class AgentDecision(BaseModel):
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    decision: Literal["retrieve_more", "stop"]
    next_query: str | None = None
    selected_evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def validate_decision(self) -> "AgentDecision":
        if self.decision == "retrieve_more":
            if not self.next_query or not self.next_query.strip():
                raise ValueError("retrieve_more requires next_query")
            return self
        if self.next_query:
            raise ValueError("stop must not include next_query")
        return self


@dataclass
class EvidenceItem:
    evidence_id: str
    chunk: RetrievedChunk
    first_call_number: int
    duplicate_count: int = 0


@dataclass
class GenerationResult:
    answer: str
    llm_call_count: int


@dataclass
class SystemBResult:
    question: str
    answer: str
    final_chunks: list[RetrievedChunk]
    all_chunks: list[RetrievedChunk]
    retrieval_calls: list[RetrievalCallRecord] = field(default_factory=list)
    agent_decisions: list[AgentDecision] = field(default_factory=list)
    latency_ms: int = 0
    retrieval_call_count: int = 0
    llm_call_count: int = 0
    warnings: list[str] = field(default_factory=list)
