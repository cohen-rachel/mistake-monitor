"""SQLAlchemy ORM models."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    current_language_profile_id = Column(
        Integer, ForeignKey("user_language_profiles.id"), nullable=True
    )

    sessions = relationship("Session", back_populates="user", lazy="selectin")
    language_profiles = relationship("UserLanguageProfile", back_populates="user", lazy="selectin", foreign_keys="UserLanguageProfile.user_id")
    current_language_profile = relationship(
        "UserLanguageProfile", foreign_keys=[current_language_profile_id], post_update=True
    )
    practice_attempts = relationship("PracticeAttempt", back_populates="user", lazy="selectin")
    rewrite_attempts = relationship("RewriteAttempt", back_populates="user", lazy="selectin")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    language = Column(String(10), default="en", nullable=False)
    audio_meta = Column(JSON, nullable=True)
    stt_provider = Column(String(50), nullable=True)
    stt_confidence_summary = Column(Float, nullable=True)
    status = Column(String(20), default="pending", nullable=False)  # pending | transcribed | analyzed

    user = relationship("User", back_populates="sessions")
    transcript = relationship("Transcript", back_populates="session", uselist=False, lazy="selectin")
    mistakes = relationship("Mistake", back_populates="session", lazy="selectin")
    language_profile_link = relationship(
        "SessionLanguageProfile",
        back_populates="session",
        uselist=False,
        lazy="selectin",
    )
    practice_attempt = relationship(
        "PracticeAttempt",
        back_populates="session",
        uselist=False,
        lazy="selectin",
    )


class UserLanguageProfile(Base):
    """Language profile per user, so one user can track multiple learning languages."""
    __tablename__ = "user_language_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    language_code = Column(String(10), nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="language_profiles", foreign_keys="UserLanguageProfile.user_id")
    session_links = relationship("SessionLanguageProfile", back_populates="language_profile", lazy="selectin")

    __table_args__ = (UniqueConstraint("user_id", "language_code"),)


class SessionLanguageProfile(Base):
    """Join table linking a session to the user's language profile."""
    __tablename__ = "session_language_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, unique=True, index=True)
    language_profile_id = Column(Integer, ForeignKey("user_language_profiles.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    session = relationship("Session", back_populates="language_profile_link")
    language_profile = relationship("UserLanguageProfile", back_populates="session_links")


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)
    raw_text = Column(Text, nullable=False)
    tokens_with_timestamps = Column(JSON, nullable=True)

    session = relationship("Session", back_populates="transcript")


class MistakeType(Base):
    __tablename__ = "mistake_types"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    label = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)

    mistakes = relationship("Mistake", back_populates="mistake_type", lazy="selectin")

    __table_args__ = (UniqueConstraint("code"),)


class Mistake(Base):
    __tablename__ = "mistakes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    mistake_type_id = Column(Integer, ForeignKey("mistake_types.id"), nullable=False)
    transcript_span = Column(String(500), nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    suggested_correction = Column(String(500), nullable=True)
    explanation_short = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    canonical_example = Column(String(500), nullable=True)
    skill_family = Column(String(200), nullable=True, index=True)
    pattern_label = Column(String(200), nullable=True, index=True)
    canonical_wrong_example = Column(String(500), nullable=True)
    canonical_correct_example = Column(String(500), nullable=True)
    stt_uncertain = Column(Boolean, default=False, nullable=False)
    uncertain = Column(Boolean, default=False, nullable=False)
    uncertain_reason = Column(String(500), nullable=True)

    session = relationship("Session", back_populates="mistakes")
    mistake_type = relationship("MistakeType", back_populates="mistakes", lazy="selectin")
    rewrite_attempts = relationship("RewriteAttempt", back_populates="source_mistake", lazy="selectin")
    memories = relationship("MistakeMemory", back_populates="source_mistake", lazy="selectin")


class PracticeAttempt(Base):
    """Stores a user's answer for a specific practice topic/session for future comparison."""
    __tablename__ = "practice_attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, unique=True, index=True)
    language = Column(String(10), nullable=False, default="en", index=True)
    topic_key = Column(String(100), nullable=False, index=True)
    topic_text = Column(Text, nullable=False)
    is_free_talk = Column(Boolean, default=False, nullable=False)
    estimated_level = Column(String(20), nullable=True)  # beginner | intermediate | advanced
    created_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="practice_attempts")
    session = relationship("Session", back_populates="practice_attempt")


class RewriteAttempt(Base):
    """Tracks user rewrite-training attempts for specific mistake instances over time."""
    __tablename__ = "rewrite_attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    language = Column(String(10), nullable=False, default="en", index=True)
    language_code = Column(String(10), nullable=False, index=True)
    source_mistake_id = Column(Integer, ForeignKey("mistakes.id"), nullable=False, index=True)
    original_sentence = Column(Text, nullable=False)
    wrong_span = Column(String(500), nullable=True)
    expected_correction = Column(String(500), nullable=True)
    user_rewrite = Column(Text, nullable=False)
    is_correct = Column(Boolean, default=False, nullable=False, index=True)
    score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="rewrite_attempts")
    source_mistake = relationship("Mistake", back_populates="rewrite_attempts")


class MistakeMemory(Base):
    """Long-lived memory of a recurring speaking mistake pattern for a language profile."""
    __tablename__ = "mistake_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    language_profile_id = Column(Integer, ForeignKey("user_language_profiles.id"), nullable=False, index=True)
    source_mistake_id = Column(Integer, ForeignKey("mistakes.id"), nullable=False, index=True)
    mistake_type_code = Column(String(50), nullable=False, index=True)
    skill_family = Column(String(200), nullable=False, index=True)
    pattern_label = Column(String(200), nullable=False, index=True)
    wrong_form = Column(String(500), nullable=True)
    correct_form = Column(String(500), nullable=True)
    canonical_wrong_example = Column(String(500), nullable=True)
    canonical_correct_example = Column(String(500), nullable=True)
    explanation = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="open", index=True)  # open | repeated | improved
    occurrence_count = Column(Integer, nullable=False, default=1)
    improvement_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    last_seen_at = Column(DateTime, default=utcnow, nullable=False)

    language_profile = relationship("UserLanguageProfile", lazy="selectin")
    source_mistake = relationship("Mistake", back_populates="memories", lazy="selectin")
    events = relationship("ImprovementEvent", back_populates="memory", lazy="selectin")


class ImprovementEvent(Base):
    """Speaking-based improvement or repeat event linked to a prior mistake memory."""
    __tablename__ = "improvement_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    language_profile_id = Column(Integer, ForeignKey("user_language_profiles.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    memory_id = Column(Integer, ForeignKey("mistake_memories.id"), nullable=False, index=True)
    event_type = Column(String(20), nullable=False, index=True)  # win | repeat
    sentence_text = Column(Text, nullable=True)
    reason = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    language_profile = relationship("UserLanguageProfile", lazy="selectin")
    session = relationship("Session", lazy="selectin")
    memory = relationship("MistakeMemory", back_populates="events", lazy="selectin")
