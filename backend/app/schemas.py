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


# ---------- User Language Profile ----------
class UserLanguageProfileBase(BaseModel):
    language_code: str
    display_name: str


class UserLanguageProfileCreate(UserLanguageProfileBase):
    pass


class UserLanguageProfileOut(UserLanguageProfileBase):
    id: int
    user_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class UserLanguageProfileSetCurrent(BaseModel):
    profile_id: int


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
    allow_language_mismatch: bool = False


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


class ProgressPoint(BaseModel):
    session_id: int
    date: str
    word_count: int
    mistake_count: int
    error_rate_per_100_words: float


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
    progress: list[ProgressPoint]
    improvement_banners: list[str]


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


# ---------- Rewrite Training ----------
class RewriteExerciseResponse(BaseModel):
    source_mistake_id: int
    language: str
    mistake_type_code: str
    mistake_type_label: str
    original_sentence: str
    wrong_span: Optional[str] = None
    expected_correction: Optional[str] = None
    explanation_short: Optional[str] = None


class RewriteSubmitRequest(BaseModel):
    user_id: int = 1
    language_code: str
    source_mistake_id: int
    original_sentence: str
    wrong_span: Optional[str] = None
    expected_correction: Optional[str] = None
    user_rewrite: str


class RewriteSubmitResponse(BaseModel):
    is_correct: bool
    score: float
    feedback: str
    expected_correction: Optional[str] = None


class RewriteStatsItem(BaseModel):
    wrong_span: Optional[str] = None
    expected_correction: Optional[str] = None
    attempts: int
    correct_attempts: int
    accuracy: float
    latest_result: Optional[bool] = None
    latest_date: Optional[str] = None


class RewriteStatsResponse(BaseModel):
    total_attempts: int
    total_correct: int
    overall_accuracy: float
    recent_attempts: list[RewriteStatsItem]
