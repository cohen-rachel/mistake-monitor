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

    sessions = relationship("Session", back_populates="user", lazy="selectin")
    language_profiles = relationship("UserLanguageProfile", back_populates="user", lazy="selectin")


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


class UserLanguageProfile(Base):
    """Language profile per user, so one user can track multiple learning languages."""
    __tablename__ = "user_language_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    language_code = Column(String(10), nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="language_profiles")
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
    stt_uncertain = Column(Boolean, default=False, nullable=False)
    uncertain = Column(Boolean, default=False, nullable=False)
    uncertain_reason = Column(String(500), nullable=True)

    session = relationship("Session", back_populates="mistakes")
    mistake_type = relationship("MistakeType", back_populates="mistakes", lazy="selectin")
