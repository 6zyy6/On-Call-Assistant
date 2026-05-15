from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    id: str
    html: str


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryItem] = Field(default_factory=list)


class ReadFileArgs(BaseModel):
    fname: str


@dataclass
class RetrievalCandidate:
    id: str
    title: str
    snippet: str
    score: float
    source: str


@dataclass
class ToolTraceEvent:
    type: str
    name: str
    fname: str
    status: str
    message: str


@dataclass
class LoadedDocument:
    id: str
    title: str
    html: str
