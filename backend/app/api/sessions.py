"""Session endpoints: create and retrieve sessions."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database import get_db
from app.config import settings
from app.models import (
    Session,
    Transcript,
    User,
    UserLanguageProfile,
    SessionLanguageProfile,
    PracticeAttempt,
)
from app.schemas import SessionCreate, SessionOut, SessionDetailOut, SessionListOut
from app.services.stt.factory import get_stt_provider
from app.services.analysis import analyze_transcript

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _language_display_name(language: str) -> str:
    names = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "ja": "Japanese",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
    }
    return names.get(language, language.upper())


@router.post("", response_model=SessionDetailOut)
async def create_session(
    audio_file: Optional[UploadFile] = File(None),
    language: str = Form("en"),
    user_id: Optional[int] = Form(None),
    transcript_text: Optional[str] = Form(None),
    practice_topic_key: Optional[str] = Form(None),
    practice_topic_text: Optional[str] = Form(None),
    is_free_talk: bool = Form(False),
    estimated_level: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Create a new session.

    Accepts either:
    - An audio file (will be transcribed via STT)
    - A pre-built transcript_text (from real-time transcription)
    """
    # Ensure user exists (create default user if needed)
    if user_id:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    else:
        # Use or create default user (id=1)
        result = await db.execute(select(User).where(User.id == 1))
        user = result.scalar_one_or_none()
        if not user:
            user = User(id=1)
            db.add(user)
            await db.flush()
        user_id = 1

    stt_provider_name = settings.stt_provider
    stt_confidence = None
    tokens_with_timestamps = None

    # If audio file provided, transcribe it
    if audio_file and not transcript_text:
        audio_bytes = await audio_file.read()
        stt = get_stt_provider()
        result = await stt.transcribe(audio_bytes, language)
        transcript_text = result.text
        stt_confidence = result.average_confidence
        tokens_with_timestamps = [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "confidence": seg.confidence,
            }
            for seg in result.segments
        ]

    if not transcript_text:
        raise HTTPException(
            status_code=400,
            detail="Either audio_file or transcript_text must be provided",
        )

    # Create session
    session = Session(
        user_id=user_id,
        language=language,
        stt_provider=stt_provider_name,
        stt_confidence_summary=stt_confidence,
        status="transcribed",
    )
    db.add(session)
    await db.flush()

    # Ensure per-user language profile exists, then link the session to it.
    profile_result = await db.execute(
        select(UserLanguageProfile).where(
            UserLanguageProfile.user_id == user_id,
            UserLanguageProfile.language_code == language,
        )
    )
    language_profile = profile_result.scalar_one_or_none()
    if language_profile is None:
        language_profile = UserLanguageProfile(
            user_id=user_id,
            language_code=language,
            display_name=_language_display_name(language),
        )
        db.add(language_profile)
        await db.flush()

    db.add(
        SessionLanguageProfile(
            session_id=session.id,
            language_profile_id=language_profile.id,
        )
    )

    # Create transcript
    transcript = Transcript(
        session_id=session.id,
        raw_text=transcript_text,
        tokens_with_timestamps=tokens_with_timestamps,
    )
    db.add(transcript)

    # Persist practice metadata for this spoken answer when provided.
    if practice_topic_key or is_free_talk:
        db.add(
            PracticeAttempt(
                user_id=user_id,
                session_id=session.id,
                language=language,
                topic_key=practice_topic_key or "free_talk",
                topic_text=practice_topic_text or "Speak freely about any topic.",
                is_free_talk=is_free_talk,
                estimated_level=estimated_level,
            )
        )
    await db.commit()

    # If audio file was uploaded, auto-analyze
    if audio_file:
        try:
            await analyze_transcript(db, session.id)
        except Exception:
            pass  # Analysis failure shouldn't block session creation

    await db.refresh(session)
    return session


@router.get("", response_model=SessionListOut)
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """List all sessions, most recent first."""
    result = await db.execute(
        select(Session).order_by(Session.created_at.desc())
    )
    sessions = result.scalars().all()
    return SessionListOut(sessions=sessions)


@router.get("/{session_id}", response_model=SessionDetailOut)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get a session with full transcript and analysis."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
