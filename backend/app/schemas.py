"""Pydantic request/response schemas."""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


# ---------- Mistake Type ----------
class MistakeTypeOut(BaseModel):
    id: int
    code: str
    label: str
    description: Optional[str] = None

    model_config = {"from_attributes": True}


# ---------- Mistake ----------
class MistakeOut(BaseModel):
    id: int
    session_id: int
    mistake_type: MistakeTypeOut
    transcript_span: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    suggested_correction: Optional[str] = None
    explanation_short: Optional[str] = None
    confidence: Optional[float] = None
    canonical_example: Optional[str] = None
    stt_uncertain: bool = False
    uncertain: bool = False
    uncertain_reason: Optional[str] = None

    model_config = {"from_attributes": True}


# ---------- Transcript ----------
class TranscriptOut(BaseModel):
    id: int
    session_id: int
    raw_text: str
    tokens_with_timestamps: Optional[list] = None

    model_config = {"from_attributes": True}


# ---------- Session ----------
class SessionCreate(BaseModel):
    language: Optional[str] = "en"
    user_id: Optional[int] = None
    audio_meta: Optional[dict] = None
    transcript_text: Optional[str] = None  # if transcript already available


class SessionOut(BaseModel):
    id: int
    user_id: Optional[int] = None
    created_at: datetime
    language: str
    audio_meta: Optional[dict] = None
    stt_provider: Optional[str] = None
    stt_confidence_summary: Optional[float] = None
    status: str

    model_config = {"from_attributes": True}


class SessionDetailOut(SessionOut):
    transcript: Optional[TranscriptOut] = None
    mistakes: list[MistakeOut] = []


class SessionListOut(BaseModel):
    sessions: list[SessionOut]


# ---------- Analysis ----------
class AnalyzeRequest(BaseModel):
    session_id: int
    transcript_text: Optional[str] = None  # override transcript if needed


class LLMMistake(BaseModel):
    """Schema expected from the LLM JSON output."""
    type: str
    span_text: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    suggested_correction: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    stt_uncertain: bool = False
    uncertain: bool = False
    uncertain_reason: Optional[str] = None


class AnalysisResult(BaseModel):
    mistakes: list[LLMMistake]


class AnalyzeResponse(BaseModel):
    session_id: int
    mistakes: list[MistakeOut]


# ---------- Insights ----------
class MistakeCountItem(BaseModel):
    code: str
    label: str
    count: int
    description: Optional[str] = None
    recent_mistake_summary: Optional[str] = None


class TrendPoint(BaseModel):
    session_id: int
    date: str
    mistake_type_code: str
    count: int


class RecentMistakeItem(BaseModel):
    id: int
    session_id: int
    date: str
    mistake_type_code: str
    mistake_type_label: str
    transcript_span: Optional[str] = None
    suggested_correction: Optional[str] = None
    explanation_short: Optional[str] = None
    original_sentence: Optional[str] = None
    corrected_sentence: Optional[str] = None


class InsightsResponse(BaseModel):
    top_mistakes: list[MistakeCountItem]
    trends: list[TrendPoint]
    recent_mistakes: list[RecentMistakeItem]


# ---------- Practice ----------
class PracticeRequest(BaseModel):
    mistake_type_code: str
    count: int = 3


class PracticePrompt(BaseModel):
    prompt: str
    expected_answer: str


class PracticeResponse(BaseModel):
    mistake_type_code: str
    prompts: list[PracticePrompt]


# ---------- Topic Practice ----------
class TopicItem(BaseModel):
    key: str
    title: str
    prompt: str
    level: str


class TopicListResponse(BaseModel):
    estimated_level: str
    topics: list[TopicItem]


class TopicAttemptItem(BaseModel):
    id: int
    session_id: int
    date: str
    language: str
    topic_key: str
    topic_text: str
    is_free_talk: bool
    estimated_level: Optional[str] = None
    mistake_count: int
    transcript: str


class TopicHistoryResponse(BaseModel):
    topic_key: str
    attempts: list[TopicAttemptItem]
